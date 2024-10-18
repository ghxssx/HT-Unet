import os
import glob
import random
import torch
# import cv2
import numpy as np
from torch.utils.data import Dataset

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'bj_train/*.npy'))
        self.count = 0  # 记录已读取的图像数量

    def augment(self, image, flipCode):
        # 数据增强函数，根据flipCode进行翻转操作
        flip = np.flip(image, flipCode)
        return flip

    def normalize(self, image):
        # 归一化函数，将图像像素值归一化到 [0, 1] 范围
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return image

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        label_path = image_path[:16] + "fenxi_train" + image_path[24:28] + "rst" + image_path[31:]
        # 读取训练图片和标签图片
        image = np.load(image_path)
        image = np.squeeze(image)
        label = np.load(label_path)
        label = np.squeeze(label)
        
        label = (label-np.min(image)) / (np.max(image) - np.min(image)) # 归一化图像


        self.count += 1

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)
    

if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("/data3/gh/chang")
    ISBI_Loader("/data3/gh/chang")
    #print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=8, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        #print(label.shape)
        #print(np.max(image),np.min(image))
        #print(np.max(label),np.min(label))
