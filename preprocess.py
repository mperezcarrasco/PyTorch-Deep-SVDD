import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset


def get_mnist(args, data_dir='./data/mnist/'):
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    x_train = train.train_data.float()/255.
    y_train = train.train_labels

    x_train = x_train[np.where(y_train==args.normal_class)].unsqueeze(1)
    y_train = y_train[np.where(y_train==args.normal_class)]
    dataloader_train = DataLoader(TensorDataset(x_train,y_train), batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    
    x_test = test.test_data.float().view(-1,784)/255.
    y_test = test.test_labels
    dataloader_test = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)

    return dataloader_train, dataloader_test