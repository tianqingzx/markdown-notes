import csv
import glob
import os
import random
import time

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root, resize, num_clazz, mode):
        super(MyDataset, self).__init__()
        self.root = root
        self.resize = resize
        self.num_clazz = num_clazz

        self.clazz2idx = {}
        # 在这里的train训练集的目录路径还需要修改，完善
        for clazz in sorted(os.listdir(os.path.join(root)))[:num_clazz]:
            if not os.path.isdir(os.path.join(root, clazz)):
                continue
            self.clazz2idx[clazz] = len(self.clazz2idx.keys())

        self.images, self.idxs = self.load_csv('images.csv')
        
        if mode == 'train':  # 80%
            self.images = self.images[:int(0.8 * len(self.images))]
            self.idxs = self.idxs[:int(0.8 * len(self.idxs))]
        elif mode == 'val':  # 20%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.idxs = self.idxs[int(0.8 * len(self.idxs)):]

    def load_csv(self, filename):
        """ 生成图片、标签对应的csv文件 """
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for clazz in self.clazz2idx.keys():
                images += glob.glob(os.path.join(str(self.root), str(clazz), '*.png'))
            print(len(images))

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    clazz = img.split(os.sep)[-2]
                    idx = self.clazz2idx[clazz]
                    # 路径，类别
                    writer.writerow([img, idx])
                print('writen in csv file:', filename)

        images, idxs, num_idx = [], [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, idx = row
                idx = int(idx)
                if idx not in num_idx:
                    num_idx.append(idx)
                images.append(img)
                idxs.append(idx)
        assert len(images) == len(idxs), '图片标签数量不匹配'
        assert self.num_clazz == len(num_idx), '类数量与csv文件中类数量不匹配'
        return images, idxs

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        """ 对图片进行正则化处理 """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):
        img, idx = self.images[idx], self.idxs[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path => image data
            transforms.Resize((int(self.resize), int(self.resize))),
#             transforms.RandomRotation(15),  # 旋转15度
#             transforms.CenterCrop(self.resize),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  # 图片都可以使用这个数据，这个数据来源于ImageNet
        ])

        img = tf(img)
        idx = torch.tensor(idx)

        return img, idx