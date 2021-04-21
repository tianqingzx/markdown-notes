import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from mydataset import MyDataset
from model import GoogLeNet

batchsz = 20
lr = 1e-3
epochs = 10
num_clazz = 200

device = torch.device('cuda')

root = "E:\\ai_learning_resource\\hwdb\\HWDB1\\train"
root_ = "E:\\ai_learning_resource\\hwdb\\HWDB1\\test"
train_db = MyDataset(root, 224, num_clazz, mode='train')
val_db = MyDataset(root, 224, num_clazz, mode='val')
test_db = MyDataset(root_, 224, num_clazz, mode='test')

train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=8)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=8)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=8)


def evalute(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def main():

    model = GoogLeNet(num_classes=num_clazz, aux_logits=True, init_weights=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    validation_acc = []
    best_acc, best_epoch = 0, 0
    global_step = 0

    for epoch in range(epochs):
        model.train()  # 训练模式
        total_batch_loss = 0
        print('start')
        # print(train_loader[:1])
        for step, data in enumerate(train_loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            logits, aux_logits2, aux_logits1 = model(x)
            loss0 = criterion(logits, y)
            loss1 = criterion(aux_logits1, y)
            loss2 = criterion(aux_logits2, y)
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            total_batch_loss += loss.item()
            # 梯度清零
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            if step % 200 == 0:
                print('Step {}/{} \t loss: {}'.format(step, len(train_loader), loss))

        # eval模式
        model.eval()
        val_acc = evalute(model, val_loader)
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), 'best.mdl')

        scheduler.step()  # 调整学习率
        print("epoch: ", epoch, "epoch_loss: ", total_batch_loss, "epoch_acc:", val_acc)

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)


if __name__ == '__main__':
    main()
