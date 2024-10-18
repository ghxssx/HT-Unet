import torch
from torch.utils.data import DataLoader
import timm
# from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from HTUnet_model import HTUnet
from utilst import *

# from engine import *
import os
import sys

# from utils import *
# from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

# from Unet_attention import U_Net_v1
from dataset_alldata import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
# from res_Unet import *

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train_net(net, device, data_path, epochs=300, batch_size=8, lr=0.01, test_ratio=0.15):

    # print_interval = 20
    
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    dataset_size = len(isbi_dataset)
    test_size = int(test_ratio * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(isbi_dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              shuffle=False)

    # train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
    #                                            batch_size=batch_size, 
    #                                            shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=0, momentum=0,alpha = 0.99,eps = 1e-8,centered = False,)
    #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
    #定义梯度下降
    scheduler = lr_scheduler.StepLR(optimizer, step_size=(epochs/5), gamma=0.5)
    # 定义Loss算法
    # criterion = nn.BCEWithLogitsLoss()

    # criterion = nn.MSELoss()   #回归任务，修改loss为mse
    criterion = GT_BceDiceLoss(wb=1, wd=0)

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    train_losses = []
    test_losses = []
    clip_value = 1.0
    #print("len(train_dataset)",len(train_dataset))  #5554
    #print("len(test_dataset)",len(test_dataset))   #979
    #print("len(train_loader)",len(train_loader))   #695
    #print("len(test_loader)",len(test_loader))    #123
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        train_loss = 0.0
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            #print(image,label)
            # 使用网络参数，输出预测结果
            gt_pre, out = net(image)
            # 计算loss
            # print(type(label))
            # print(label.size())
            # print(type(pred))
            # print(gt_pre[0].size(),gt_pre[1].size(),gt_pre[2].size(),gt_pre[3].size(),gt_pre[4].size(),out.size())
            loss = criterion(gt_pre, out, label)
            #print("loss",loss)
            train_loss += loss.item() * image.size(0)
            #print("train_loss",train_loss)
            # print('epoch:'+str(epoch)+'-Loss/train:', loss.item())
            # # 保存loss值最小的网络参数
            # if loss < best_loss:
            #     best_loss = loss
            #     torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            #nn.utils.clip_grad_norm_(net.parameters(), clip_value)
            optimizer.step()
        
        train_loss /= len(train_dataset)
        print('epoch:', epoch, 'Loss/train:', train_loss)

        # 调用学习率调度器进行学习率衰减
        scheduler.step()

        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_image, test_label in test_loader:
                test_image = test_image.to(device=device, dtype=torch.float32)
                test_label = test_label.to(device=device, dtype=torch.float32)
                test_gt_pre, test_out = net(test_image)
                test_loss += criterion(test_gt_pre, test_out, test_label).item() * test_image.size(0)

        test_loss /= len(test_dataset)
        print('epoch:', epoch, 'Loss/test:', test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(net.state_dict(), '/home/gh/Unet/save_model/HTUnet.pth')

        # 保存训练和测试loss
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
        # 绘制loss变化曲线
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curve')
    plt.legend()
    plt.savefig('/home/gh/Unet/log/egeunet_5_alldata0_modify_hdm_5_transformer.png')
    plt.show()


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:3')
    # 加载网络，图片单通道1，分类为1。
    # net = U_Net_v1(img_ch=35, output_ch=34)
    net = HTUnet(num_classes=34, 
                        input_channels=35, 
                        c_list=[8,16,24,32,48,64], 
                        bridge=True,
                        gt_ds=True,)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "/data3/gh/chang"
    import torch

    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        print("GPU is available")
        # 输出当前GPU设备的名称
        print("Current GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU is not available, using CPU")

    train_net(net, device, data_path)

