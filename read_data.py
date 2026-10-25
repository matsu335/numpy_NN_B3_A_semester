import numpy as np
import os
import cv2 as cv
# from scipy.signal import convolve2d

# 訓練データを読み込むための関数
def read_traindata(category, size):
    path = os.listdir('data/traindata/' + category)
    n = len(path)
    images = np.zeros((n, size[0], size[1], 3), dtype=np.float32)
    for i in range(n):
        img = cv.imread('data/traindata/' + category + '/' + path[i])
        images[i] = cv.resize(img, (size[1],size[0]))
    
    images /= 255
    return images # 返り値の形状はimages → (画像の枚数, size[0], size[1], 3)

# テストデータを読み込むための関数
def read_testdata(category_list, size):
    images_all = np.empty((0, size[0], size[1], 3))
    label_indexes_all = np.empty(0, dtype=np.int32)
    for i in range(len(category_list)):
        path = os.listdir('data/testdata/images/' + category_list[i])
        n = len(path)
        images = np.zeros((n, size[0], size[1], 3), dtype=np.float32)
        label_indexes = np.ones(n, dtype=np.int32) * i
        for j in range(n):
            img = cv.imread('data/testdata/images/' + category_list[i] + '/' + path[j])
            images[j] = cv.resize(img, (size[1],size[0]))
        images_all = np.concatenate((images_all, images), axis=0)
        label_indexes_all = np.concatenate((label_indexes_all, label_indexes))

    images_all /= 255
    return images_all, label_indexes_all # 返り値の形状はimages_all → (画像の枚数, size[0], size[1], 3), label_indexes_all → 一次元配列
