import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ae.auto_encoder import AE
from ae.vae import VAE
from visdom import Visdom


def main():
    mnist_train = datasets.MNIST('E:/ai_learning_resource/mnist', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST('E:/ai_learning_resource/mnist', False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

    x, _ = next(iter(mnist_train))
    print('x:', x.shape)

    device = torch.device('cuda')
    model = VAE().to(device)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    viz = Visdom()

    for epoch in range(1000):
        for batchidx, (x, _) in enumerate(mnist_train):
            x = x.to(device)

            x_hat, kld = model(x)
            loss = criteon(x_hat, x)

            if kld is not None:
                elbo = -loss - 1.0 * kld
                loss = -elbo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, 'loss', loss.item(), 'kld:', kld.item())

        x, _ = next(iter(mnist_test))
        x = x.to(device)
        with torch.no_grad():
            x_hat = model(x)
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    main()
