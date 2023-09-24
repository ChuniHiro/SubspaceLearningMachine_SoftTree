"""Data loader"""
import torch
import torchvision
from torchvision import datasets, transforms
from ops import ChunkSampler
import os
import numpy as np

def get_dataloaders(
        dataset='mnist',
        batch_size=128,
        augmentation_on=False,
        cuda=False, num_workers=0,
):
    # TODO: move the dataloader to data.py
    kwargs = {
        'num_workers': num_workers, 'pin_memory': True,
    } if cuda else {}

    if dataset == 'mnist' :
        if augmentation_on:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )

        mnist_train = datasets.MNIST(
            '../data', train=True, download=True, transform=transform_train,
        )
        mnist_valid = datasets.MNIST(
            '../data', train=True, download=True, transform=transform_test,
        )
        mnist_test = datasets.MNIST(
            '../data', train=False, transform=transform_test,
        )

        TOTAL_NUM = 60000
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))

        train_loader = torch.utils.data.DataLoader(
            mnist_train,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)

        valid_loader = torch.utils.data.DataLoader(
            mnist_valid,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)

        test_loader = torch.utils.data.DataLoader(
            mnist_test,
            batch_size=1000,
            shuffle=False,
            **kwargs)
        
    elif dataset == 'fashion_mnist':

        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ],
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ],
        )

        mnist_train = datasets.FashionMNIST(
            '../data', train=True, download=True, transform=transform_train,
        )
        mnist_valid = datasets.FashionMNIST(
            '../data', train=True, download=True, transform=transform_test,
        )
        mnist_test = datasets.FashionMNIST(
            '../data', train=False, transform=transform_test,
        )

        TOTAL_NUM = 60000
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))

        train_loader = torch.utils.data.DataLoader(
            mnist_train,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)

        valid_loader = torch.utils.data.DataLoader(
            mnist_valid,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)

        test_loader = torch.utils.data.DataLoader(
            mnist_test,
            batch_size=1000,
            shuffle=False,
            **kwargs)


    elif dataset == 'cifar10':
        if augmentation_on:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )

        cifar10_train = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True,
            transform=transform_train,
        )
        cifar10_valid = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform_test,
        )
        cifar10_test = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True,
            transform=transform_test,
        )

        TOTAL_NUM = 50000
        NUM_VALID = int(round(TOTAL_NUM * 0.02))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        train_loader = torch.utils.data.DataLoader(
            cifar10_train,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            cifar10_valid,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            cifar10_test,
            batch_size=1000,
            shuffle=False,
            **kwargs)
        
    
    elif dataset == 'cifar100':
        if augmentation_on:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )

        cifar100_train = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True,
            transform=transform_train,
        )
        cifar100_valid = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_test,
        )
        cifar100_test = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True,
            transform=transform_test,
        )

        TOTAL_NUM = 50000
        NUM_VALID = int(round(TOTAL_NUM * 0.02))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        train_loader = torch.utils.data.DataLoader(
            cifar100_train,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            cifar100_valid,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            cifar100_test,
            batch_size=1000,
            shuffle=False,
            **kwargs)
        
    elif dataset == 'tiny-imagenet':

        if augmentation_on:
            transform_train = transforms.Compose(
                [
                    transforms.RandomRotation(20),
                    transforms.RandomCrop(64, padding=16),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262),
                    ),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262),
                    ),
                ],
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262),),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262),),
                ],
            )

        # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
        #           for x in ['train', 'val','test']}
        # dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x])
        #                 for x in ['train', 'val', 'test']}

        # cifar100_train = torchvision.datasets.CIFAR100(
        #     root='./data', train=True, download=True,
        #     transform=transform_train,
        # )

        # cifar100_valid = torchvision.datasets.CIFAR100(
        #     root='./data', train=True, download=True, transform=transform_test,
        # )
        # cifar100_test = torchvision.datasets.CIFAR100(
        #     root='./data', train=False, download=True,
        #     transform=transform_test,
        # )

        # data_dir = "../data/tiny-imagenet-200/"
        data_dir = "./tiny_imagenet/tiny-imagenet-200/"
        tiny_imagenet_train = datasets.ImageFolder(os.path.join(data_dir, 'train'),transform=transform_train) 
        # tiny_imagenet_val = datasets.ImageFolder(os.path.join(data_dir, 'val'),transform=transform_test)
        # tiny_imagenet_test = datasets.ImageFolder(os.path.join(data_dir, 'test'),transform=transform_test)
        tiny_imagenet_val = datasets.ImageFolder(os.path.join(data_dir, 'val'),transform=transform_test)
        tiny_imagenet_test = datasets.ImageFolder(os.path.join(data_dir, 'val'),transform=transform_test)

        TOTAL_NUM = 100000
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))

        train_loader = torch.utils.data.DataLoader(
            tiny_imagenet_train,
            batch_size=batch_size,
            # sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            tiny_imagenet_val,
            batch_size=batch_size,
            # sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            tiny_imagenet_test,
            batch_size=100,
            shuffle=False,
            **kwargs)

        # train_loader = torch.utils.data.DataLoader(tiny_imagenet_train, batch_size=batch_size, shuffle=True)
        # valid_loader = torch.utils.data.DataLoader(tiny_imagenet_val, batch_size=batch_size)
        # test_loader = torch.utils.data.DataLoader(tiny_imagenet_test, batch_size=1000, shuffle=False)
        
    else:
        raise NotImplementedError("Specified data set is not available.")

    return train_loader, valid_loader, test_loader, NUM_TRAIN, NUM_VALID


def get_dataset_details(dataset):

    if dataset == 'mnist' or  dataset == 'fashion_mnist':
        input_nc, input_width, input_height = 1, 28, 28
        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    elif dataset == 'fashion_mnist':
        input_nc, input_width, input_height = 1, 28, 28
        classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

    elif dataset == 'cifar10':
        input_nc, input_width, input_height = 3, 32, 32
        classes = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck',
        )
        
    elif dataset == 'cifar100':
        input_nc, input_width, input_height = 3, 32, 32
        classes = (' apple', ' aquarium_fish', ' baby', ' bear', ' beaver', ' bed', ' bee', ' beetle', ' bicycle', ' bottle', ' bowl', ' boy', ' bridge', ' bus', ' butterfly', ' camel', ' can', ' castle', ' caterpillar', ' cattle', ' chair', ' chimpanzee', ' clock', ' cloud', ' cockroach', ' couch', ' cra', ' crocodile', ' cup', ' dinosaur', ' dolphin', ' elephant', ' flatfish', ' forest', ' fox', ' girl', ' hamster', ' house', ' kangaroo', ' keyboard', ' lamp', ' lawn_mower', ' leopard', ' lion', ' lizard', ' lobster', ' man', ' maple_tree', ' motorcycle', ' mountain', ' mouse', ' mushroom', ' oak_tree', ' orange', ' orchid', ' otter', ' palm_tree', ' pear', ' pickup_truck', ' pine_tree', ' plain', ' plate', ' poppy', ' porcupine', ' possum', ' rabbit', ' raccoon', ' ray', ' road', ' rocket', ' rose', ' sea', ' seal', ' shark', ' shrew', ' skunk', ' skyscraper', ' snail', ' snake', ' spider', ' squirrel', ' streetcar', ' sunflower', ' sweet_pepper', ' table', ' tank', ' telephone', ' television', ' tiger', ' tractor', ' train', ' trout', ' tulip', ' turtle', ' wardrobe', ' whale', ' willow_tree', ' wolf', ' woman', ' worm')
    elif dataset == 'tiny-imagenet':
        input_nc, input_width, input_height = 3, 64, 64
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199')

    else:
        raise NotImplementedError("Specified data set is not available.")

    return input_nc, input_width, input_height, classes
