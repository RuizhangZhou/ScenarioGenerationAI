import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #优化
import numpy as np
import matplotlib.pyplot as plt #绘图
import torchvision #加载图片
from torchvision import transforms #图片变换
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os


#设备的配置
device='cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

#对数据做归一化（-1，1）
img_transform=transforms.Compose([
    #将shanpe为（H,W,C）的数组或img转为shape为（C,H,W）的tensor
    transforms.ToTensor(), #转为张量并归一化到【0，1】；数据只是范围变了，并没有改变分布
    transforms.Normalize(0.5,0.5)#数据归一化处理，将数据整理到[-1,1]之间；可让数据呈正态分布
])

#下载数据到指定的文件夹
# train_ds = torchvision.datasets.MNIST('data',
#                                       train=True,
#                                      transform=transform,
#                                      download=True)

# dataloader=torch.utils.data.DataLoader(train_ds,batch_size=64,shuffle=True)

# Configure data loader
# img_transform = transforms.Compose([
#     # transforms.ToPILImage(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
# ])

# 输入图片所在文件夹
# 读取CSV文件
# def read_csv_file(file_path):
#     df = pd.read_csv(file_path)
#     return df

# def get_all_file_paths(folder_path):
#     all_file_paths = []
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             file_path = os.path.abspath(os.path.join(root, file))
#             all_file_paths.append(file_path)
#     return all_file_paths

# folder_path = '/DATA1/rzhou/ika/testcases/inD/full_cases'  # 替换为你的文件夹路径
# file_paths = get_all_file_paths(folder_path)
# print(file_paths)


# # 读取所有CSV文件
# n = len(file_paths)  # 假设有n个文件
# datasets = [read_csv_file(file_path) for file_path in file_paths]
# # 合并数据集
# combined_data = torch.cat([torch.tensor(df.values) for df in datasets])
# # 重塑数据集维度
# combined_data = combined_data.view(n, 45030, 4)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


combined_data = torch.load('/DATA1/rzhou/ika/testcases/inD/inD_full_cases.pt')
# 创建自定义Dataset实例
custom_dataset = CustomDataset(combined_data)

# 创建DataLoader
batch_size = 64
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# 验证数据集维度
# sample_batch = next(iter(dataloader))
# print("数据集维度:", sample_batch.shape)  # 应该输出 (n, 45030, 4)

class Generator(nn.Module):
    #45030
    def __init__(self):
        super(Generator,self).__init__()
        self.main=nn.Sequential(
        nn.Linear(100,512),
        nn.ReLU(),
        nn.Linear(512,2048),
        nn.ReLU(),
        nn.Linear(2048,8192),
        nn.ReLU(),
        nn.Linear(8192,32768),
        nn.ReLU(),
        nn.Linear(32768,45030),
        nn.Tanh()#对于生成器，最后一个激活函数是tanh,值域：-1到1
        )
    #定义前向传播 
    def forward(self,x):  #x表示长度为100的noise输入
        img = self.main(x)
        img=img.view(-1,30,1501)#转换成图片的形式
        return img
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
        nn.Linear(45030,32768),
        nn.LeakyReLU(),
        nn.Linear(32768,8192),
        nn.LeakyReLU(),
        nn.Linear(8192,2048),
        nn.LeakyReLU(),
        nn.Linear(2048,512),
        nn.LeakyReLU(),
        nn.Linear(512,128),
        nn.LeakyReLU(),
        nn.Linear(128,1),
        nn.Sigmoid()
        )
    def forward(self,x):
        x =x.view(-1,45030) #展平
        x =self.main(x.float())
        return x
    

#初始化生成器和判别器把他们放到相应的设备上
gen = Generator().to(device)
dis = Discriminator().to(device)

#判别器的优化器
d_optim = torch.optim.Adam(dis.parameters(),lr=0.0001)
#生成器的优化器
g_optim = torch.optim.Adam(gen.parameters(),lr=0.0001)

#交叉熵损失函数
loss_fn = torch.nn.BCELoss()

# def gen_img_plot(model,test_input):
#     prediction = np.squeeze(model(test_input).detach().cpu().numpy())
#     fig = plt.figure(figsize=(4,4))
#     for i in range(16):
#         plt.subplot(4,4,i+1)
#         plt.imshow((prediction[i]+1)/2)
#         plt.axis('off')
#     plt.show()
    
test_input = torch.randn(16,100 ,device=device) #16个长度为100的随机数

D_loss = []
G_loss = []

#训练循环
for epoch in range(5):
    #初始化损失值
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader) #返回批次数 60000/64=938 batches
    #对数据集进行迭代
    for step,img in enumerate(dataloader):
        img =img.to(device) #把数据放到设备上
        size = img.size(0) #img的第一位是size,获取批次的大小
        random_noise = torch.randn(size,100,device=device)
        
        #判别器训练(真实图片的损失和生成图片的损失),损失的构建和优化
        d_optim.zero_grad()#梯度归零
        #判别器对于真实图片产生的损失
        real_output = dis(img) #判别器输入真实的图片，real_output对真实图片的预测结果
        d_real_loss = loss_fn(real_output,
                              torch.ones_like(real_output)
                              )
        d_real_loss.backward()#计算梯度
        
        #在生成器上去计算生成器的损失，优化目标是判别器上的参数
        gen_img = gen(random_noise) #得到生成的图片
        #因为优化目标是判别器，所以对生成器上的优化目标进行截断
        fake_output = dis(gen_img.detach()) #判别器输入生成的图片，fake_output对生成图片的预测;detach会截断梯度，梯度就不会再传递到gen模型中了
        #判别器在生成图像上产生的损失
        d_fake_loss = loss_fn(fake_output,
                              torch.zeros_like(fake_output)
                              )
        d_fake_loss.backward()
        #判别器损失
        d_loss = d_real_loss + d_fake_loss
        #判别器优化
        d_optim.step()
        
        
        #生成器上损失的构建和优化
        g_optim.zero_grad() #先将生成器上的梯度置零
        fake_output = dis(gen_img)
        g_loss = loss_fn(fake_output,
                              torch.ones_like(fake_output)
                          )  #生成器损失
        g_loss.backward()
        g_optim.step()
        #累计每一个批次的loss
        with torch.no_grad():
            d_epoch_loss +=d_loss
            g_epoch_loss +=g_loss
    #求平均损失
    with torch.no_grad():
        d_epoch_loss /=count
        g_epoch_loss /=count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch:',epoch)
        print('dLoss:',d_epoch_loss)
        print('gLoss:',g_epoch_loss)
        print(gen_img)
            