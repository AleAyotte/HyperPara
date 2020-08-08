import math
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils


def dataset_to_loader(dataset, b_size=12, shuffle=False, pin_memory=False):

    """
    Transforms a torch dataset into a torch dataloader who provide an iterable over the dataset

    :param dataset: A torch dataset
    :param b_size: The batch size (Default: 12)
    :param shuffle: If the dataset is shuffle at each epoch (Default: False)
    :param pin_memory: If true put the fetched data Tensors in pinned memory (Default: False)
    :return: A torch data_loader that contain the features and the labels.
    """
    data_loader = utils.DataLoader(dataset,
                                   batch_size=b_size,
                                   shuffle=shuffle,
                                   drop_last=True,
                                   pin_memory=pin_memory)

    return data_loader


def validation_split(dtset=None, valid_size=0.1, seed=None):

    """
    Splits a torch dataset into two torch dataset.

    :param dtset: A torch dataset which contain our train data points and labels
    :param valid_size: Proportion of the dataset that will be use as validation data
    :param seed: The seed that will be used to split randomly the trainset into train and valid set.
    :return: train and valid features as numpy arrays and train and valid labels as numpy arrays if features and labels
             numpy arrays are given but no torch dataset. Train and valid torch datasets if a torch dataset is given.
    """
    if dtset is None:
        raise Exception("Dataset is missing. dtset is None: {}".format(dtset is None))
    else:
        num_data = len(dtset)
        num_valid = math.floor(num_data * valid_size)
        num_train = num_data - num_valid

        seed = 0 if seed is None else seed

        torch.manual_seed(seed)

        return utils.dataset.random_split(dtset, [num_train, num_valid])


def load_cifar10(valid_size=0.10, valid_aug=False, seed=None):

    """
    Loads the CIFAR10 dataset using pytorch
    inspired by pytorch tutorial "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

    :param valid_size: Proportion of the train set that will be use as validation data
    :param valid_aug: If true, use data_augmentation on the validation set.
    :param seed: The seed that will be used to split randomly the trainset into train and valid set.
    :return: The train set, validation set and test set of the CIFAR10 dataset as pytorch Dataset
    """

    assert 1 > valid_size > 0, 'Validation set proportion is suppose to be a real value between 0 and 1.'

    # Data augmentation for training
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                          torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.50),
                                                                               ratio=(0.25, 2.25), value=0)])

    # For the test set, we just want to normalize it
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # If we want to create a validation set with base data augmentation
    if valid_aug:
        transform_valid = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(0.5),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        transform_valid = transform_test

    # We get download the dataset if needed
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_valid)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # We split the train set into training set and validation set.
    trainset, _ = validation_split(trainset, valid_size, seed)
    _, validset = validation_split(validset, valid_size, seed)

    return trainset, validset, testset


def load_cifar100(valid_size=0.10, valid_aug=False, seed=None):

    """
    Loads the CIFAR100 dataset using pytorch
    inspired by pytorch tutorial "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

    :param valid_size: Proportion of the train set that will be use as validation data
    :param valid_aug: If true, use data_augmentation on the validation set.
    :param seed: The seed that will be used to split randomly the trainset into train and valid set.
    :return: The train set, validation set and test set of the CIFAR100 dataset as pytorch Dataset
    """

    # Data augmentation for training
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                          torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.50),
                                                                               ratio=(0.25, 2.25), value=0)])

    # For the test set, we just want to normalize it
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # If we want to create a validation set with base data augmentation
    if valid_aug:
        transform_valid = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(0.5),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        transform_valid = transform_test

    # We get download the dataset if needed
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    validset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_valid)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # We split the train set into training set and validation set.
    trainset, _ = validation_split(trainset, valid_size, seed)
    _, validset = validation_split(validset, valid_size, seed)

    return trainset, validset, testset
