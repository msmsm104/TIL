import torch

from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset):  # Custom dataset.

    def __init__(self, data, labels, flatten=True):
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        # |x| = (28, 28)
        y = self.labels[idx]
        # |y| = (1, )
        if self.flatten:
            x = x.view(-1)
            # |x| = (784, )

        return x, y
        # DataLoader에 의해 불러지면 batch_size만큼 concat이 이루어짐
        # |x| = (batch_size, 784) || (batch_size, 28, 28)
        # |y| = (batch_size, 1)


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def get_loaders(config):  # argparse로부터 받은 config
    x, y = load_mnist(is_train=True, flatten=False)

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(x, dim=0, index=indices).split([
        train_cnt, valid_cnt], dim=0)
    # |train_x| = (train_cnt, 28, 28)
    train_y, valid_y = torch.index_select(y, dim=0, index=indices).split([
        train_cnt, valid_cnt], dim=0)
    # |train_y| = (train_cnt, 1)

    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_x, test_y = load_mnist(is_train=False, flatten=False)
    # |test_x| = (10000, 28, 28)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
