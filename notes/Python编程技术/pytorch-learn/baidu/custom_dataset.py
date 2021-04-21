import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms, datasets
import torch.optim as optim

from tqdm import tqdm
from visdom import Visdom

from video.googlenet.model import GoogLeNet
from mydataset import MyDataset


num_clazz = 20
batch_size = 10
lr = 1e-3
epochs = 10
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 6])  # number of workers


def main():
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_db = MyDataset('E:\\ai_learning_resource\\hwdb\\HWDB1\\train', 224, num_clazz=num_clazz, mode='train')
    val_db = MyDataset('E:\\ai_learning_resource\\hwdb\\HWDB1\\train', 224, num_clazz=num_clazz, mode='val')

    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=nw)

    val_loader = DataLoader(val_db, batch_size=batch_size, num_workers=nw // 2)

    print("using {} images for training, {} images for validation.".format(len(train_db), len(val_db)))

    # net = GoogLeNet(num_classes=num_clazz, aux_logits=True, init_weights=True)
    net = models.densenet121(pretrained=True, bn_size=batch_size, drop_rate=0.6, num_classes=num_clazz)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss().to(device)

    start = time.time()
    print('start time: ', start)
    train_steps = len(train_loader)
    # best_acc, best_epoch = 0, 0
    for epoch in range(epochs):
        net.train()
        print('start train')
        x, y = next(iter(train_loader))
        print(x[0].numpy(), y[0])
        running_loss = 0.0
        #     train_bar = tqdm(train_loader)
        for step, data in enumerate(train_loader):
            print('training')
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(x)
            loss0 = criteon(logits, y)
            loss1 = criteon(aux_logits1, y)
            loss2 = criteon(aux_logits2, y)
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            print('Step {}/{}'.format(step, len(train_loader)))

        # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            # val_bar = tqdm(val_loader)
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                outputs = net(val_x)  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_y).sum().item()

        val_accurate = acc / len(val_db)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

    print('Finished Training')
    print('\n{} epoch cost time {:f}s'.format(epochs, time.time() - start))


if __name__ == '__main__':
    main()
