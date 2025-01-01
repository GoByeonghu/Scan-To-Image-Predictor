import random
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from os.path import join
from os import listdir
#from torchsummary import summary
import time
import zipfile
#import tensorflow as tf
#import tensorflow.contrib.slim as slim


# 하이퍼파라미터 설정
num_epochs = 1
batch_size = 64
learning_rate = 0.001

# 랜덤 시드 고정
np.random.seed(42)

# 시작 시간 기록
start_time = time.time()

# 이미지 로드 함수 정의
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 커스텀 데이터셋 클래스 정의
class CustomDataset(data.Dataset):
    def __init__(self, noisy_image_paths, clean_image_paths, patch_size = 128, transform=None):
        self.clean_image_paths = [join(clean_image_paths, x) for x in listdir(clean_image_paths)]
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_paths[index])
        clean_image = load_img(self.clean_image_paths[index])

        H, W, _ = clean_image.shape

        # 이미지 랜덤 크롭
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        noisy_image = noisy_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        clean_image = clean_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        
        # transform 적용
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image
    
# 데이터셋 경로
noisy_image_paths = 'content/dataset/train/scan'
clean_image_paths = 'content/dataset/train/clean'

# 데이터셋 로드 및 전처리
train_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 커스텀 데이터셋 인스턴스 생성
train_dataset = CustomDataset(noisy_image_paths, clean_image_paths, transform=train_transform)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델 정의

#########################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, num_layers=17, num_channels=64):
        super(Network, self).__init__()

    def down_sample(self, x, scale_factor_h, scale_factor_w):
        _, _, h, w = x.size()
        new_size = (int(h / scale_factor_h), int(w / scale_factor_w))
        return F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

    def resBlock(self, x, channels=64, kernel_size=(3, 3), scale=1):
        tmp = nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False)(x)
        tmp = F.relu(tmp)
        tmp = nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False)(tmp)
        tmp *= scale
        return x + tmp

    def res_upsample_and_sum(self, x1, x2, output_channels, in_channels, scope=None):
        pool_size = 2
        x2 = self.resBlock(x2, output_channels)
        deconv_filter = nn.Parameter(torch.Tensor(pool_size, pool_size, output_channels, in_channels))
        deconv_filter.data.normal_(0, 0.02)
        deconv = F.conv_transpose2d(x1, deconv_filter, stride=pool_size)

        deconv_output = deconv + x2
        return deconv_output

    def param_free_norm(self, x, epsilon=1e-5):
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x_std = torch.sqrt(x.var(dim=(2, 3), keepdim=True) + epsilon)
        return (x - x_mean) / x_std

    def ain(self, noise_map, x_init, channels, scope='AIN'):
        x_init_shape = x_init.size()
        noise_map_down = F.interpolate(noise_map, size=(x_init_shape[2], x_init_shape[3]), mode='bilinear', align_corners=False)
        x = self.param_free_norm(x_init)
        tmp = nn.Conv2d(1, 64, kernel_size=5, padding=2)(noise_map_down)
        tmp = F.relu(tmp)
        noisemap_gamma = nn.Conv2d(64, channels, kernel_size=3, padding=1)(tmp)
        noisemap_beta = nn.Conv2d(64, channels, kernel_size=3, padding=1)(tmp)
        x = x * (1 + noisemap_gamma) + noisemap_beta
        return x

    def ain_resblock(self, noisemap, x_init, channels, scope='AINRes'):
        x = self.ain(noisemap, x_init, channels, scope='AIN_1')
        x = F.leaky_relu(x, negative_slope=0.02)
        x = nn.Conv2d(channels, channels, kernel_size=3, padding=1)(x)

        x = self.ain(noisemap, x, channels, scope='AIN_2')
        x = F.leaky_relu(x, negative_slope=0.02)
        x = nn.Conv2d(channels, channels, kernel_size=3, padding=1)(x)

        return x + x_init

    def FCN_Avg(self, input):
        x = nn.Conv2d(3, 32, kernel_size=3, padding=1)(input)
        x = F.relu(x)
        x = nn.Conv2d(32, 32, kernel_size=3, padding=1)(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=4, stride=4)
        x = nn.Conv2d(32, 32, kernel_size=3, padding=1)(x)
        x = F.relu(x)
        x = nn.Conv2d(32, 32, kernel_size=3, padding=1)(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = nn.Conv2d(32, 3, kernel_size=3, padding=1)(x)
        image_shape = input.size()
        y = F.interpolate(x, size=(image_shape[2], image_shape[3]), mode='bilinear', align_corners=False)
        y = nn.Conv2d(3, 3, kernel_size=3, padding=1)(y)
        y = nn.Conv2d(3, 3, kernel_size=3, padding=1)(y)
        return x, y

    def AINDNet_recon(self, input, noise_map):
        conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)(input)
        conv1 = self.ain_resblock(noise_map, conv1, 64, 'AINRes1_1')
        conv1 = self.ain_resblock(noise_map, conv1, 64, 'AINRes1_2')

        pool1 = F.avg_pool2d(conv1, kernel_size=2, stride=2)
        conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)(pool1)
        conv2 = self.ain_resblock(noise_map, conv2, 128, 'AINRes2_1')
        conv2 = self.ain_resblock(noise_map, conv2, 128, 'AINRes2_2')

        pool2 = F.avg_pool2d(conv2, kernel_size=2, stride=2)
        conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)(pool2)
        conv3 = self.ain_resblock(noise_map, conv3, 256, 'AINRes3_1')
        conv3 = self.ain_resblock(noise_map, conv3, 256, 'AINRes3_2')
        conv3 = self.ain_resblock(noise_map, conv3, 256, 'AINRes3_3')
        conv3 = self.ain_resblock(noise_map, conv3, 256, 'AINRes3_4')
        conv3 = self.ain_resblock(noise_map, conv3, 256, 'AINRes3_5')

        up4 = self.res_upsample_and_sum(conv3, conv2, 128, 256, scope='deconv4')
        conv4 = self.ain_resblock(noise_map, up4, 128, 'AINRes4_1')
        conv4 = self.ain_resblock(noise_map, conv4, 128, 'AINRes4_2')
        conv4 = self.ain_resblock(noise_map, conv4, 128, 'AINRes4_3')

        up5 = self.res_upsample_and_sum(conv4, conv1, 64, 128, scope='deconv5')
        conv5 = self.ain_resblock(noise_map, up5, 64, 'AINRes5_1')
        conv5 = self.ain_resblock(noise_map, conv5, 64, 'AINRes5_2')
        out = nn.Conv2d(64, 3, kernel_size=1)(conv5)

        return out

    def AINDNet(self, input):
        down_noise_map, noise_map = self.FCN_Avg(input)
        image_shape = input.size()
        upsample_noise_map = F.interpolate(down_noise_map, size=(image_shape[2], image_shape[3]), mode='bilinear', align_corners=False)
        noise_map = 0.8 * upsample_noise_map + 0.2 * noise_map
        out = self.AINDNet_recon(input, noise_map) + input

        return out

    def forward(self, x):
        out = self.AINDNet(x)
        return out
    

class Model(nn.Module):
    def __init__(self, num_layers=17, num_channels=64):
        super(Model, self).__init__()

        self.num_layers = num_layers
        self.num_channels = num_channels

        # Define layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=1)

        self.ain_conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.ain_conv2_gamma = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        self.ain_conv2_beta = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        self.ain_conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.ain_conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        self.deconv_filter = nn.Parameter(torch.Tensor(2, 2, num_channels, 128))

        self.fc_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.fc_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fc_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fc_conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.fc_conv5 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.fc_conv6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def down_sample(self, x, scale_factor_h, scale_factor_w):
        _, _, h, w = x.size()
        new_size = (int(h / scale_factor_h), int(w / scale_factor_w))
        return F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

    def resBlock(self, x, channels=64, kernel_size=(3, 3), scale=1):
        tmp = nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False)(x)
        tmp = F.relu(tmp)
        tmp = nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False)(tmp)
        tmp *= scale
        return x + tmp

    def res_upsample_and_sum(self, x1, x2, output_channels, in_channels, scope=None):
        pool_size = 2
        x2 = self.resBlock(x2, output_channels)
        deconv = F.conv_transpose2d(x1, self.deconv_filter, stride=pool_size)

        deconv_output = deconv + x2
        return deconv_output

    def param_free_norm(self, x, epsilon=1e-5):
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x_std = torch.sqrt(x.var(dim=(2, 3), keepdim=True) + epsilon)
        return (x - x_mean) / x_std

    def ain(self, noise_map, x_init, channels, scope='AIN'):
        x_init_shape = x_init.size()
        noise_map_down = F.interpolate(noise_map, size=(x_init_shape[2], x_init_shape[3]), mode='bilinear', align_corners=False)
        x = self.param_free_norm(x_init)
        tmp = self.ain_conv1(noise_map_down)
        tmp = F.relu(tmp)
        noisemap_gamma = self.ain_conv2_gamma(tmp)
        noisemap_beta = self.ain_conv2_beta(tmp)
        x = x * (1 + noisemap_gamma) + noisemap_beta
        return x

    def ain_resblock(self, noisemap, x_init, channels, scope='AINRes'):
        x = self.ain(noisemap, x_init, channels, scope='AIN_1')
        x = F.leaky_relu(x, negative_slope=0.02)
        x = self.ain_conv3(x)

        x = self.ain(noisemap, x, channels, scope='AIN_2')
        x = F.leaky_relu(x, negative_slope=0.02)
        x = self.ain_conv4(x)

        return x + x_init

    def FCN_Avg(self, input):
        x = self.fc_conv1(input)
        x = F.relu(x)
        x = self.fc_conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=4, stride=4)
        x = self.fc_conv3(x)
        x = F.relu(x)
        x = self.fc_conv4(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.fc_conv5(x)
        image_shape = input.size()
        y = F.interpolate(x, size=(image_shape[2], image_shape[3]), mode='bilinear', align_corners=False)
        y = self.fc_conv6(y)
        y = self.fc_conv6(y)
        return x, y

    def AINDNet_recon(self, input, noise_map):
        conv1 = self.conv1(input)
        conv1 = self.ain_resblock(noise_map, conv1, 64, 'AINRes1_1')
        conv1 = self.ain_resblock(noise_map, conv1, 64, 'AINRes1_2')

        pool1 = F.avg_pool2d(conv1, kernel_size=2, stride=2)
        conv2 = self.conv2(pool1)
        conv2 = self.ain_resblock(noise_map, conv2, 128, 'AINRes2_1')
        conv2 = self.ain_resblock(noise_map, conv2, 128, 'AINRes2_2')

        pool2 = F.avg_pool2d(conv2, kernel_size=2, stride=2)
        conv3 = self.conv3(pool2)
        conv3 = self.ain_resblock(noise_map, conv3, 256, 'AINRes3_1')
        conv3 = self.ain_resblock(noise_map, conv3, 256, 'AINRes3_2')

        pool3 = F.avg_pool2d(conv3, kernel_size=2, stride=2)
        conv4 = self.conv4(pool3)
        conv4 = self.ain_resblock(noise_map, conv4, 128, 'AINRes4_1')
        conv4 = self.ain_resblock(noise_map, conv4, 128, 'AINRes4_2')

        pool4 = F.avg_pool2d(conv4, kernel_size=2, stride=2)
        conv5 = self.conv5(pool4)
        conv5 = self.ain_resblock(noise_map, conv5, 64, 'AINRes5_1')
        conv5 = self.ain_resblock(noise_map, conv5, 64, 'AINRes5_2')

        pool5 = F.avg_pool2d(conv5, kernel_size=2, stride=2)
        conv6 = self.conv6(pool5)

        return conv6

    def forward(self, input):
        '''
        input_shape = input.size()
        noise_map_down = F.interpolate(noise_map, size=(input_shape[2], input_shape[3]), mode='bilinear', align_corners=False)

        recon = self.AINDNet_recon(input, noise_map_down)

        conv6 = self.down_sample(recon, 8, 8)
        _, feature_map = self.FCN_Avg(conv6)

        return feature_map
        '''
        down_noise_map, noise_map = self.FCN_Avg(input)
        image_shape = input.size()
        upsample_noise_map = F.interpolate(down_noise_map, size=(image_shape[2], image_shape[3]), mode='bilinear', align_corners=False)
        noise_map = 0.8 * upsample_noise_map + 0.2 * noise_map
        out = self.AINDNet_recon(input, noise_map) + input
        return out



##########################################################################################################################
    
# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DnCNN 모델 인스턴스 생성 및 GPU로 이동
model = Network().to(device)
#print(summary(model, (3, 128, 128)))

# 손실 함수와 최적화 알고리즘 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
model.train()
best_loss = 9999.0
for epoch in range(num_epochs):
    running_loss = 0.0
    for noisy_images, clean_images in train_loader:
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, noisy_images-clean_images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * noisy_images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 현재 epoch의 loss가 최소 loss보다 작으면 모델 갱신
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_dncnn_model.pth')
        print(f"{epoch+1}epoch 모델 저장 완료")

# 종료 시간 기록
end_time = time.time()

# 소요 시간 계산
training_time = end_time - start_time

# 시, 분, 초로 변환
minutes = int(training_time // 60)
seconds = int(training_time % 60)
hours = int(minutes // 60)
minutes = int(minutes % 60)

# 결과 출력
print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")
