import numpy as np

from net import Net
from read_data import read_traindata, read_testdata


def main():

    # 畳み込み層を使うかどうか。今回は使わないので、False固定で行う。
    useConvolution = False

    # ノイズを加えるか
    addNoise = True
    d = 25

    # 画像をリサイズするための大きさ設定。
    if useConvolution == True:
        image_size = (16,16)
    else:
        image_size = (10,10)

    # 学習率を決定
    learning_rate = 0.01

    # 画像のカテゴリーを定義
    category_list = ['apple_green', 'apple_red', 'avocado', 'banana', 'blueberry', 'cabbage', 'carrot', 'cauliflower', 'cherry', 'corn']
    # category_list = ['apple_green', 'apple_red', 'avocado', 'banana', 'blueberry', 'cabbage', 'carrot', 'cauliflower']
    # category_list = ['apple_green', 'apple_red', 'avocado', 'banana', 'blueberry']
    # category_list = ['apple_green', 'banana']
    category_num = len(category_list)

    # 訓練データを得るための準備
    traindata_input = np.empty((0, image_size[0], image_size[1], 3))
    traindata_label_matrix = np.empty((0, category_num))

    # 各カテゴリにおいて、画像を取得し、入力と出力の正解の組み合わせを作成。
    for i in range(category_num):
        # 全ての訓練データ画像をまとめた行列traindata_inputを作成。
        train_image = read_traindata(category_list[i], image_size) # 形状は(画像の枚数, size[0], size[1], 3)
        # ノイズを加える
        if addNoise == True:
            shape = (train_image.shape[0], image_size[0], image_size[1], 3)
            mask = np.random.rand(*shape) < (d / 100)
            train_image[mask] = np.random.rand(np.sum(mask))
        traindata_input = np.concatenate((traindata_input, train_image), axis=0)
        # 訓練データの答えをまとめたベクトルをカテゴリーの数だけまとめた行列を作成。
        traindata_label_matrix_add = np.zeros((train_image.shape[0], category_num))
        traindata_label_matrix_add[:, i] = 1
        traindata_label_matrix = np.concatenate((traindata_label_matrix, traindata_label_matrix_add), axis=0)

    # ニューラルネットのインスタンスを作成
    net = Net(image_size[0], image_size[1], category_num, useConvolution) # category_numは分類するカテゴリーの数。
    net = net.Back_Prop(traindata_input.shape[0], traindata_input, traindata_label_matrix, learning_rate)

    # テストデータを得る。
    test_image, label_index = read_testdata(category_list, image_size)

    # 正解数を記録するための変数
    success_num = 0
    # 画像を識別する。
    for i in range(test_image.shape[0]):
        # テストデータをニューラルネットに入れ、出力ベクトルを得る。
        testdata_output = net.Forward_Prop(test_image[i])
        # 出力ベクトルから、カテゴリーを推定
        estimated_label_index = np.argmax(testdata_output)
        if estimated_label_index == label_index[i]:
            success_num += 1
            sign = '○'
        else:
            sign = '×'
        # 結果をプリント
        print("画像{}: {}.  実際→ {}, 推定→ {}, 出力ベクトル→ {}\n".format(i, sign, label_index[i], estimated_label_index, testdata_output))

    # 正解率を算出
    success_rate = success_num / test_image.shape[0] * 100
    print("正解率: {}%".format(success_rate))
    
# # 特徴量を抽出する関数。画像データの(番号,高さ,幅,RGB)の4次元配列を、情報を保持したまま(番号,特徴量)の2次元配列に変換。
# def extract_features(image):
#     n_image = image.shape[0]
#     n_feat = image[0].size
#     feat = np.zeros((n_image, n_feat), dtype=np.float32)
#     for i in range(n_image):
#         feat[i,:] = np.reshape(image[i],(1,n_feat))
#     feat /= 256 # 各画素値は0~1の浮動小数に変換して入力する．
#     return feat
    

if __name__ == '__main__':
    main()
