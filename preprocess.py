import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset


def get_mnist(args, data_dir='./data/mnist/'):
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    x_train = train.data.float()/255.
    y_train = train.targets

    x_train = x_train[np.where(y_train==args.normal_class)].unsqueeze(1)
    y_train = y_train[np.where(y_train==args.normal_class)]
    dataloader_train = DataLoader(TensorDataset(x_train,y_train), batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    
    x_test = (test.data.float()/255.).unsqueeze(1)
    y_test = test.targets
    y_test[np.where(y_test!=args.normal_class)[0]] = 20
    y_test[np.where(y_test==args.normal_class)[0]] = 10
    y_test[np.where(y_test==10)[0]] = 0
    y_test[np.where(y_test==20)[0]] = 1
    dataloader_test = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    return dataloader_train, dataloader_test