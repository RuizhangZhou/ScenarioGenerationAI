#import statements
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import LeakyReLU

#模型输入维度
img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)#输入图片的维度

z_dim = 100#噪声向量的大小用作生成器的输入
img_shape

def build_generator(img_shape, z_dim):
    model = Sequential([
        Dense(128, input_dim=z_dim),#全连接层
        LeakyReLU(alpha=0.01),
        Dense(28*28*1, activation='tanh'),
        Reshape(img_shape)#生成器的输出改变为图像尺寸
    ])
    return model

build_generator(img_shape, z_dim).summary()

def build_discrimination(img_shape):
    model = Sequential([
        Flatten(input_shape=img_shape),#输入图像展平
        Dense(128),
        LeakyReLU(alpha=0.01),
        Dense(1, activation='sigmoid')
    ])
    return model

build_discrimination(img_shape).summary()


def build_gan(generator, discriminator):
    model = Sequential()

    # 生成器模型和判别器模型结合到一起
    model.add(generator)
    model.add(discriminator)

    return model


discriminator = build_discrimination(img_shape)  # 构建并编译判别器
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)
generator = build_generator(img_shape, z_dim)  # 构建生成器
discriminator.trainable = False  # 训练生成器时保持判别器的参数固定

# 构建并编译判别器固定的GAN模型，以生成训练器
gan = build_gan(generator, discriminator)
gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam()
)

losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):
    (x_train, _), (_, _) = mnist.load_data()  # 加载mnist数据集
    x_train = x_train / 127.5 - 1.0  # 灰度像素值[0,255]缩放到[-1,1]
    x_train = np.expand_dims(x_train, axis=3)
    real = np.ones((batch_size, 1))  # 真实图像的标签都是1
    fake = np.zeros((batch_size, 1))  # 伪图像的标签都是0
    for iteration in range(iterations):
        idx = np.random.randint(0, x_train.shape[0], batch_size)  # 随机噪声采样
        imgs = x_train[idx]

        z = np.random.normal(0, 1, (batch_size, 100))  # 获取随机的一批真实图像
        gen_imgs = generator.predict(z)

        # 图像像素缩放到[0,1]
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 1, (batch_size, 100))  # 生成一批伪图像
        gen_imgs = generator.predict(z)

        g_loss = gan.train_on_batch(z, real)  # 训练判别器

        if (iteration + 1) % sample_interval == 0:
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
            iteration + 1, d_loss, 100.0 * accuracy, g_loss))  # 输出训练过程

            sample_images(generator)  # 输出生成图像的采样


def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))  # 样本随机噪声

    gen_imgs = generator.predict(z)  # 从随机噪声生成图像

    gen_imgs = 0.5 * gen_imgs + 0.5  # 将图像像素重置缩放至[0, 1]内

    # 设置图像网格
    fig, axs = plt.subplots(
        image_grid_rows,
        image_grid_columns,
        figsize=(4, 4),
        sharex=True,
        sharey=True
    )

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')  # 输出一个图像网格
            axs[i, j].axis('off')
            cnt += 1


#设置训练超参数
iterations = 20000
batch_size = 128
sample_interval = 1000

train(iterations, batch_size, sample_interval)
