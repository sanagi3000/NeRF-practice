import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def load_data():
    # tiny_nerf用に準備されたデータがnpzファイルに格納されているので，ロードする
    data = np.load("./data/tiny_nerf_data.npz")
    images = data["images"] # 画像データ
    poses = data["poses"] # 4x4の行列，カメラの外部パラメータを表す
    focal = data["focal"] # スカラー値，焦点距離を表す

    return images, poses, focal

def calculate_ray(image_height, image_width, focal, pose):
    # カメラの外部パラメータから，各ピクセルに対応するレイ(原点o, 方向d)を計算する
    # まず全ピクセルの行列を作成
    x, y = np.meshgrid(np.arange(image_width), np.arange(image_height), indexing="xy")
    # 中心が0, 0になるように正規化
    x_normalized = x - image_width * 0.5
    y_normalized = y - image_height * 0.5
    # print("x_normalized:", x_normalized)
    # print("y_normalized:", y_normalized)

    # ピンホールの相似関係を利用すると，モデルから各レイは以下のように計算される．
    directions_original = np.stack([x_normalized / focal, -y_normalized / focal, -np.ones_like(x)], axis=-1)
    # この計算の結果，directionas_originalは，image_height x imageth_width x 3のテンソルとなる．

    # これに，カメラの回転行列の回転移動成分pose[:3, :3]を適用する．

    directions_ray = np.sum(directions_original[..., None, :] * pose[:3, :3], axis=-1)
    # 一回Noneを加えることで，image_height x imageth_width x 1 x 3のテンソルに変換し，
    # pose[:3, :3]との要素積を取り，最後の次元で和を取ることで，
    # 結果的にimage_height x imageth_width x 3のテンソルが得られる．
    # (行列積という演算を，ブロードキャストして積を求めた後に和をとったものと捉えている)．

    # 原点の座標は，カメラの並行移動成分であるpose[:3, -1]に他ならない．これは全てのピクセルで共通なので，direction_rayと同じ形状にbroadcastしてやれば，origins_rayを得られる．
    # つまり，origins_rayは全てのピクセルで同じ値を持つ
    origins_ray = np.broadcast_to(pose[:3, -1], directions_ray.shape) # レイの始点を計算

    return origins_ray, directions_ray
    

def volume_rendering(model, o, d, random_flag, near=2., far=6., N_samples=64):
    def batch_function(function, chunk_size=1024*32):
        # 与えられた関数（function）を，chankのかたまりごとに適用し，その結果を連結して返す
        return lambda x: torch.cat([function(x[i:i+chunk_size]) for i in range(0, x.size(0), chunk_size)], dim=0)

    sampling_t = torch.linspace(near, far, N_samples).to(o.device) # 0から1までのN_samples個の等間隔な数列を生成 
    # random性を付与
    # torch.rand_likeは与えられたテンソルと同じ形状で平均0, 分散1の乱数を生成するもの
    # ゆえに，これをノイズとして加えることで，区画内でランダムに揺らぐサンプリングが可能となる．
    if random_flag:
        noise = torch.rand_like(sampling_t) * (far - near) / N_samples
        sampling_t += noise
    point_list = o[..., None, :] + d[..., None, :] * sampling_t[..., :, None] # 各サンプリング点の座標を計算
    # ネットワークをバッチごとに適用
    point_list_converted = torch.reshape(point_list, (-1, 3))
    result = batch_function(model)(point_list_converted)

    # バッチごとに適用した結果を元の形状に戻す
    # points_listの最後の要素数は3つ(x, y, z)であったが，実行結果は4つ(sigma, r, g, b)なので，そのように変換
    result = torch.reshape(result, list(point_list.shape[:-1]) + [4])

    sigma = torch.relu(result[..., 3])
    rgb = torch.sigmoid(result[..., :3])
    
    faraway = torch.broadcast_to(torch.tensor([1e10]), sampling_t[..., :1].shape).to(o.device) # 遠方のサンプリング点を表す deviceに送る処理が必要
    distance_list = torch.cat([sampling_t[..., 1:] - sampling_t[..., :-1], faraway], dim=-1) # 各サンプリング点間の距離を計算
    alpha = 1.0 - torch.exp(-sigma * distance_list) # 各サンプリング点の透過率を計算
    # plt.hist(alpha.cpu().detach().numpy().flatten(), bins=100)
    # plt.savefig("./result/weights.png")
    # plt.close()

    # 累積積cumprodを用いて，各サンプリング点の重みを計算
    cumprod = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    # cumprod_shifted = torch.cat([torch.ones_like(cumprod[..., :1]), cumprod[..., :-1]], dim=-1)
    cumprod_shifted = torch.roll(cumprod, 1, -1) 
    cumprod_shifted[..., 0] = 1.
    weights = alpha * cumprod_shifted # 各サンプリング点の重みを計算


    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2) # 各サンプリング点の色を重み付けして足し合わせることで，最終的な色を計算

    return rgb_map
    

def positonal_encoding(x, L):
    # 実数xを入力として，R^2Lの高次元空間に写像するのがpositonal　encoding
    # ここではテンソルを受け取り，その各要素の実数それぞれに対して，positonal encodingを行い，最終結果をconcatenateして返す
    # 論文の式に従い,sinとcosを用いる
    result_list = []
    for l in range(L):
        result_list.append(torch.sin(2.0**l * np.pi * x))
        result_list.append(torch.cos(2.0**l * np.pi * x))
    
    return torch.cat(result_list, dim=-1) # ここではresult_list内のtensorはすべて一次元なので，全てを結合したtensorを返す


def weights_initialization(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

class nerf_model (torch.nn.Module):
    # nerfのニューラルネットワークを定義する
    def __init__(self):
        super(nerf_model, self).__init__() # スーパークラス (torch.nn.Module) のコンストラクタを呼び出す
        self.layer0 = torch.nn.Linear(3*2*6, 256)
        self.layer1 = torch.nn.Linear(256, 256)
        self.layer2 = torch.nn.Linear(256, 256)
        self.layer3 = torch.nn.Linear(256, 256)
        self.layer4 = torch.nn.Linear(256, 256)
        self.layer5 = torch.nn.Linear(256 + 3*2*6, 256) # inputをskip connectionで結合
        self.layer6 = torch.nn.Linear(256, 256)
        self.layer7 = torch.nn.Linear(256, 256)
        self.layer8 = torch.nn.Linear(256, 256)
        self.layer9 = torch.nn.Linear(256, 128)
        self.layer10 = torch.nn.Linear(128, 4)

        self.apply(weights_initialization)

    def forward(self, input):
        encoded_input = positonal_encoding(input, 6) # 3 * 2 * 6 = 36要素
        h = torch.relu(self.layer0(encoded_input))
        h = torch.relu(self.layer1(h))
        h = torch.relu(self.layer2(h))
        h = torch.relu(self.layer3(h))
        h = torch.relu(self.layer4(h))
        h = torch.relu(self.layer5(torch.cat([h, encoded_input], dim=-1))) # inputをskip connectionで結合
        h = torch.relu(self.layer6(h))
        h = torch.relu(self.layer7(h))
        h = torch.relu(self.layer8(h))
        h = torch.relu(self.layer9(h))
        output = self.layer10(h)

        return output


def main():

    # デバイスの設定 GPUが使える場合はGPUを使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nerf_model()
    model.to(device) # モデルをデバイスに転送

    # 最適化手法の設定
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 

    images, poses, focal = load_data()

    # ハイパーパラメータの設定
    N_sample = 64 # 画像ごとのサンプル数
    N_iter = 10000 # 学習回数
    plot_iter = 100 # 学習経過のプロット間隔

    # 開始時刻の取得
    start_time = time.time()
    

    num_images = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]

    PSNR_file = open("./result/PSNR.csv", "w")
    loss_file = open("./result/loss.csv", "w")

    tmp_list_x = []
    tmp_list_y = []
    # 学習のためのループ
    for i in range(N_iter):
        # 画像をランダムに選択
        target_index = np.random.randint(num_images)

        target_image = images[target_index]
        # rgbを保存してみる
        # plt.imshow(target_image)
        # plt.savefig("./result/target_image.png")
        # plt.close()

        target_pose = poses[target_index]

        # レイの計算
        o, d = calculate_ray(image_height, image_width, focal, target_pose)

        o = torch.tensor(o, dtype=torch.float32).to(device)
        d = torch.tensor(d, dtype=torch.float32).to(device)
        target_image = torch.tensor(target_image, dtype=torch.float32).to(device)
        
        # モデルの出力を取得
        rgb = volume_rendering(model, o, d, random_flag=True)
        # rgbを保存してみる
        # plt.imshow(rgb.cpu().detach().numpy())
        # plt.savefig("./result/rgb_test.png")
        # plt.close()


        # 損失関数はL2ノルムを用いる
        # pytorchの場合は，L2ノルムに対応する関数がそのまま用意されている
        loss = torch.nn.functional.mse_loss(rgb, target_image)
        loss_file.write(str(loss) + "\n")
        # 逆伝搬→パラメータの更新
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        PSNR = -10 * torch.log10(loss)

        # 現在時刻の取得
        elapsed_time = time.time() - start_time
        print("iteration: ", i, "elappsed_time", elapsed_time, "loss: ", loss.item(), "PSNR: ", PSNR.item())    
        PSNR_file.write(str(PSNR) + "\n")


        if i % plot_iter == 0:
            # 100回ごとにテスト画像の出力結果を保存
            for j in [101, 102, 103]:
                test_image = images[j]
                test_pose = poses[j]

                o, d = calculate_ray(image_height, image_width, focal, test_pose)
                o = torch.tensor(o, dtype=torch.float32).to(device)
                d = torch.tensor(d, dtype=torch.float32).to(device)
                test_image = torch.tensor(test_image, dtype=torch.float32).to(device)

                rgb_test = volume_rendering(model, o, d, random_flag=True)

                plt.imshow(rgb_test.cpu().detach().numpy())
                plt.savefig("./result/iter_" + str(i) + "_image_" + str(j) + ".png")
    
if __name__ == "__main__":
    main()