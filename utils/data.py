import torchvision.transforms as T
import torchvision.datasets as dset


def load_data(conf, split='train'):
    """Keys in conf: 'name', 'root', 'img_size'."""

    if conf.name.lower() == 'mnist':
        transform = T.Compose([
            T.Resize((conf.img_size, conf.img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        dataset = dset.MNIST(root=conf.root, train=(split == 'train'), transform=transform)

    elif conf.name.lower() in ['cifar10', 'cifar-10']:
        flip_p = 0.5 if split == 'train' else 0.0
        transform = T.Compose([
            T.Resize((conf.img_size, conf.img_size), antialias=True),
            T.RandomHorizontalFlip(p=flip_p),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset = dset.CIFAR10(root=conf.root, train=(split == 'train'), transform=transform)

    else:
        raise NotImplementedError(f'Unknown dataset: {conf.name}')

    return dataset
