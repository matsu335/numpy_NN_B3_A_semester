import numpy as np

class Net():

    def __init__(self, image_size_h, image_size_w, category_num, useConvolution, fullyConnectionedlayer = 3):
        self.size_h = image_size_h
        self.size_w = image_size_w
        self.useConvolution = useConvolution # 畳み込み層を使うかどうか。
        if self.useConvolution == True:
            self.feat_num = int(self.size_h/4) * int(self.size_w/4) * 3
        else:
            self.feat_num = self.size_h * self.size_w * 3
        self.category_num = category_num # category_numは分類するカテゴリーの数。出力ベクトルの次元にする。

        # 畳み込み層、最大プーリング層について定義。今回は使いません。活性化関数も使えておらず、線形変換しているだけです。
        self.hidden_matrix_0 = np.zeros((self.size_h, self.size_w, 3))
        self.kernel_1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        self.hidden_matrix_1 = np.zeros((self.size_h, self.size_w, 3)) # 1つ目のConvolutionの後
        self.poolingArgMax_1 = np.zeros((self.size_h, self.size_w, 3))
        self.hidden_matrix_2 = np.zeros((int(self.size_h/2), int(self.size_w/2), 3))
        self.kernel_2 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        self.hidden_matrix_3 = np.zeros((int(self.size_h/2), int(self.size_w/2), 3))
        self.poolingArgMax_2 = np.zeros((int(self.size_h/2), int(self.size_w/2), 3))
        self.hidden_matrix_4 = np.zeros((int(self.size_h/4), int(self.size_w/4), 3))
        self.delta_conv_4 = np.zeros((int(self.size_h/4), int(self.size_w/4), 3)) # 2つ目のPoolingの後
        self.delta_conv_3 = np.zeros((int(self.size_h/2), int(self.size_w/2), 3)) # 2つ目のConvolutionと2つ目のPoolingの間
        self.delta_conv_2 = np.zeros((int(self.size_h/2), int(self.size_w/2), 3)) # 1つ目のPoolingと2つ目のConvolutionの間
        self.delta_conv_1 = np.zeros((self.size_h, self.size_w, 3)) # 1つ目のConvolutionと1つ目のPoolingの間
        self.delta_conv_0 = np.zeros((self.size_h, self.size_w, 3)) # 1つ目のConvolutionの前

        # 全結合層について定義
        self.fullyConnectionedLayer = fullyConnectionedlayer # layerは層の数
        self.weight = np.random.randn(self.fullyConnectionedLayer-2, self.feat_num, self.feat_num) * 0.01 # 重みはランダムな値で初期化
        self.weight_last = np.random.randn(self.category_num, self.feat_num) * 0.01
        self.u = np.zeros((self.fullyConnectionedLayer-1, self.feat_num)) # 内部の状態を定義。各層への入力。状態ベクトル(特徴量次元)を層の数だけ作る。ただし、最終層を除くため、(層の数-1)個。
        self.u_last = np.zeros(self.category_num) # 最終層への入力
        self.x = np.zeros((self.fullyConnectionedLayer-1, self.feat_num)) # 内部の状態を定義。各層からの出力。状態ベクトル(特徴量次元)を層の数だけ作る。ただし、最終層を除くため、(層の数-1)個。
        self.x_last = np.zeros(self.category_num) # 最終層からの出力
        self.delta = np.zeros((self.fullyConnectionedLayer-1, self.feat_num)) # 損失関数の勾配ベクトル(特徴量次元)を層の数だけ作る。ただし、最終層を除くため、(層の数-1)個。
        self.delta_last = np.zeros(self.category_num) # 最終層の損失関数の勾配ベクトル
        # 許容する損失関数の値を定義
        self.lossMaximum = 0.03
        self.finish_permitted_times = 1
        # 使用する活性化関数を定義
        self.activation_function = self.ReLU
        self.activation_function_derivative = self.ReLU_derivative

    # FeedForward処理を実装
    def Forward_Prop(self, input_image): # input_image: (size[0], size[1], 3)の次元の行列
        # コンボリューションですが、使いません。
        if self.useConvolution == True:
            self.hidden_matrix_0 = input_image
            self.hidden_matrix_1 = self.Convolution(self.hidden_matrix_0, self.kernel_1)
            self.hidden_matrix_2 = self.Pooling(self.hidden_matrix_1, self.poolingArgMax_1)
            self.hidden_matrix_3 = self.Convolution(self.hidden_matrix_2, self.kernel_2)
            self.hidden_matrix_4 = self.Pooling(self.hidden_matrix_3, self.poolingArgMax_2)
            self.u[0, :] = self.Extract_features(self.hidden_matrix_4) # self.uは特徴量を格納する1次元のベクトル。
        else:
            self.u[0, :] = self.Extract_features(input_image) # self.uは特徴量を格納する1次元のベクトル。
        self.x[0, :] = self.activation_function(self.u[0, :])
        for i in range(self.fullyConnectionedLayer-2):
            self.u[i+1, :] = np.dot(self.weight[i, :, :], self.x[i, :])
            self.x[i+1, :] = self.activation_function(self.u[i+1, :])
        self.u_last = np.dot(self.weight_last, self.x[self.fullyConnectionedLayer-2])
        self.x_last = self.softmax(self.u_last)
        output = self.x_last
        return output

    # BackPropagation処理を実装
    def Back_Prop(self, P, xP, yP, eta):
        loss_p = np.ones(self.category_num)
        permitted_times = 0
        while (permitted_times < self.finish_permitted_times):  # 出力誤差が一定値以下になるまで
            # ランダムなp番目の訓練データの入力画像データxpと出力ベクトルypを抽出。分類器を作りたいので、カテゴリーの数次元のベクトルを出力とする。
            p = np.random.randint(0, P) 
            xp = xP[p] # p番目の画像データを得る。
            yp = yP[p] # p番目の画像の出力ベクトルを得る。
            
            # 現在の重み関数で入力xpをFeedForward処理。
            y = self.Forward_Prop(xp)

            # p番目の訓練データで出力誤差ベクトル(特徴量次元)を計算
            loss_p = y - yp

            # 最終層の損失関数の勾配を計算。
            self.delta_last = np.multiply(loss_p, self.softmax_derivative(self.u_last))

            # 最終層直前の重み関数を計算。
            self.weight_last = self.weight_last - eta * np.outer(self.delta_last, self.x[self.fullyConnectionedLayer-2, :])

            # 誤差逆伝播で重み関数を更新していく。
            for i in range(self.fullyConnectionedLayer-2):
                if i == 0:
                    self.delta[self.fullyConnectionedLayer-2, :] = np.multiply(np.dot(self.weight_last.T, self.delta_last), self.activation_function_derivative(self.u[self.fullyConnectionedLayer-2, :]))
                else:
                    self.delta[self.fullyConnectionedLayer-2-i, :] = np.multiply(np.dot(self.weight[self.fullyConnectionedLayer-2-i,:, :].T, self.delta[self.fullyConnectionedLayer-1-i, :]), self.activation_function_derivative(self.u[self.fullyConnectionedLayer-2-i, :]))
                self.weight[self.fullyConnectionedLayer-3-i, :, :] = self.weight[self.fullyConnectionedLayer-3-i, :, :] - eta * np.outer(self.delta[self.fullyConnectionedLayer-2-i, :], self.x[self.fullyConnectionedLayer-3-i, :])
            # 畳み込み層のバックプロパゲーションを書いているが、使わない。
            if self.useConvolution == True:
                self.delta[0, :] = np.multiply(np.dot(self.weight[0,:, :].T, self.delta[1, :]), self.activation_function_derivative(self.u[0, :]))
                self.delta_conv_4 = self.Undo_Extract_features(self.delta[0, :]) # Undoで特徴量を画像に戻す
                self.delta_conv_3 = self.Back_Pooling(self.delta_conv_4, self.poolingArgMax_2, self.hidden_matrix_3)
                self.delta_conv_2 = self.Back_Convolution(self.delta_conv_3, self.hidden_matrix_2, self.kernel_2, eta)
                self.delta_conv_1 = self.Back_Pooling(self.delta_conv_2, self.poolingArgMax_1, self.hidden_matrix_1)
                self.delta_conv_0 = self.Back_Convolution(self.delta_conv_1, self.hidden_matrix_0, self.kernel_1, eta)

            if np.linalg.norm(loss_p) > self.lossMaximum:
                permitted_times = 0
            else:
                permitted_times += 1

            # self.delta[0, :]は使わないため更新していない
            print("一つのpについて逆伝播が完了しました。loss_p = {}, p = {}".format(loss_p, p))

        return self

    # 使いません
    def Convolution(self, input_image, kernel): # input_image: (高さ, 幅, 3)の次元の行列
        padded_image = np.pad(input_image, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        output_image = np.zeros((input_image.shape[0], input_image.shape[1], 3))
        for i in range(input_image.shape[0]):
            for j in range(input_image.shape[1]):
                output_image[i, j, :] = np.sum(kernel * padded_image[i:i+3, j:j+3, :], axis=(0,1))
        
        return output_image
    # 使いません
    def Back_Convolution(self, delta, hidden_matrix, kernel, eta):
        padded_hidden_matrix = np.pad(hidden_matrix, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        for i in range(3):
            for j in range(3):
                kernel[i,j] = kernel[i,j] - eta * np.sum(padded_hidden_matrix[i:padded_hidden_matrix.shape[0]-2+i, j:padded_hidden_matrix.shape[1]-2+j, :] * delta[:, :, :])/(delta.shape[0]*delta.shape[1]*3)
        flipped_kernel = np.flipud(np.fliplr(kernel))
        return self.Convolution(delta, flipped_kernel) * self.activation_function_derivative(hidden_matrix)
    # 使いません
    def Pooling(self, input_image, record_matrix): # input_image: (高さ, 幅, 3)の次元の行列
        output_image = np.zeros((int(input_image.shape[0]/2), int(input_image.shape[1]/2), 3))
        for i in range(int(input_image.shape[0]/2)):
            for j in range(int(input_image.shape[1]/2)):
                section = input_image[i*2:i*2+2, j*2:j*2+2, :]
                for k in range(3):
                    # 最大値を求め、そのインデックスを記録
                    max_value = np.max(section[:, :, k])
                    max_index = np.unravel_index(np.argmax(section[:, :, k]), section[:, :, k].shape)
                    
                    # 最大値のインデックスをrecord_matrixに記録
                    record_matrix[i*2 + max_index[0], j*2 + max_index[1], k] = 1
                    
                    # output_imageに最大値を設定
                    output_image[i, j, k] = max_value
        
        return output_image
    # 使いません
    def Back_Pooling(self, delta, record_matrix, hidden_matrix):
        enlarged_delta = np.repeat(delta, 2, axis=0)
        enlarged_delta = np.repeat(enlarged_delta, 2, axis=1)
        return record_matrix * enlarged_delta * self.activation_function_derivative(hidden_matrix) 
                        
    # 画像を1次元の特徴量ベクトルに変換。全結合層に入れるため。
    def Extract_features(self, image_matrix):
        image_feat = np.reshape(image_matrix, (1,self.feat_num))
        return image_feat
    
    def Undo_Extract_features(self, image_feat):
        image_matrix = np.reshape(image_feat, (int(self.size_h/4), int(self.size_w/4), 3))
        return image_matrix


    # 活性化関数を定義
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2
    
    def softmax(self, x):
        sum_probability = np.sum(np.exp(x))
        return np.exp(x)/sum_probability
    
    def softmax_derivative(self, x):
        return self.softmax(x) * (1 - self.softmax(x))

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return np.where(x > 0, 1, 0)
