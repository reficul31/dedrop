import os
import cv2
import h5py
import numpy as np

def get_patches(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def preprocess_train_data(data_path, patch_size, stride):
    rain_folder_path = os.path.join(data_path, 'data')
    clean_folder_path = os.path.join(data_path, 'gt')

    rain_image_list = os.listdir(rain_folder_path)
    clean_image_list = os.listdir(clean_folder_path)

    rain_image_list.sort()
    clean_image_list.sort()

    save_clean_path = os.path.join(data_path, 'train_rain.h5')
    save_rain_path = os.path.join(data_path, 'train_clean.h5')

    clean_h5f = h5py.File(save_clean_path, 'w')
    rain_h5f = h5py.File(save_rain_path, 'w')

    assert len(rain_image_list) == len(clean_image_list)

    train_num = 0
    for i in range(len(rain_image_list)):
        rain_path = os.path.join(rain_folder_path, rain_image_list[i])
        clean_path = os.path.join(clean_folder_path, clean_image_list[i])
        
        if not os.path.exists(rain_path) or not os.path.exists(clean_path):
            continue

        clean = cv2.imread(clean_path)
        b, g, r = cv2.split(clean)
        clean = cv2.merge([r, g, b])

        rain = cv2.imread(rain_path)
        b, g, r = cv2.split(rain)
        rain = cv2.merge([r, g, b])

        clean = np.float32(clean) / 255
        rain = np.float32(rain) / 255
        
        clean_patches = get_patches(clean.transpose(2,0,1), win=patch_size, stride=stride)
        rain_patches = get_patches(rain.transpose(2, 0, 1), win=patch_size, stride=stride)

        for n in range(rain_patches.shape[3]):
            clean_data = clean_patches[:, :, :, n].copy()
            clean_h5f.create_dataset(str(train_num), data=clean_data)

            rain_data = rain_patches[:, :, :, n].copy()
            rain_h5f.create_dataset(str(train_num), data=rain_data)

            train_num += 1

    rain_h5f.close()
    clean_h5f.close()

if __name__ == '__main__':
    file_path = os.path.dirname(os.path.realpath(__file__))
    preprocess_train_data(os.path.join(file_path, 'train'), 100, 80)
