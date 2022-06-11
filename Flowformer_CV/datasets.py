import os
import json

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import os
from PIL import Image
import pickle

def build_dataset(is_train, args, infer_no_resize=False):
    transform = build_transform(is_train, args, infer_no_resize)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = 1000
    return dataset, nb_classes


def build_transform(is_train, args, infer_no_resize=False):
    if hasattr(args, 'arch'):
        if 'cait' in args.arch and not is_train:
            print('# using cait eval transform')
            transformations = {}
            transformations = transforms.Compose(
                [transforms.Resize(args.input_size, interpolation=3),
                 transforms.CenterCrop(args.input_size),
                 transforms.ToTensor(),
                 transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
            return transformations

    if infer_no_resize:
        print('# using cait eval transform')
        transformations = {}
        transformations = transforms.Compose(
            [transforms.Resize(args.input_size, interpolation=3),
             transforms.CenterCrop(args.input_size),
             transforms.ToTensor(),
             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        return transformations

    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
