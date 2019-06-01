import random
import math

from PIL import Image, ImageOps
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomRotation, RandomVerticalFlip, RandomOrder, RandomApply,
    RandomHorizontalFlip, RandomResizedCrop, ColorJitter)

class RandomRotate:
    def __init__(self):
        self.degrees = [0, 90, 180, 270]
    def __call__(self, img):
        degree = random.choice(self.degrees)
        return img.rotate(degree)

class KeepAsepctResize:
    def __init__(self, target_size=(288, 288)):
        self.target_size = target_size
    
    def __call__(self, img):
        width, height = img.size
        long_side = max(width, height)
        
        delta_w = long_side - width
        delta_h = long_side - height
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        img = ImageOps.expand(img, padding)
        img = img.resize(self.target_size)
        return img



tensor_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_transform(
        target_size=(288,288),
        transform_list='random_crop, horizontal_flip', # random_crop | keep_aspect
        augment_ratio=0.5,
        is_train=True,
        ):
    transform = list()
    transform_list = transform_list.split(', ')
    augments = list()

    for transform_name in transform_list:
        if transform_name == 'random_crop':
            scale = (0.5, 1.0) if is_train else (0.8, 1.0)
            transform.append(RandomResizedCrop(target_size, scale=(0.8, 1.0)))
        elif transform_name == 'keep_aspect':
            transform.append(KeepAsepctResize(target_size))
        elif transform_name == 'horizontal_flip':
            augments.append(RandomHorizontalFlip())
        elif transform_name == 'vertical_flip':
            augments.append(RandomVerticalFlip())
        elif transform_name == 'random_rotate':
            augments.append(RandomRotate())
        elif transform_name == 'color_jitter':
            brightness = 0.1 if is_train else 0.05
            contrast = 0.1 if is_train else 0.05
            augments.append(ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=0,
                hue=0,
            ))
    transform.append(RandomApply(augments, p=augment_ratio))    
    
    return Compose(transform)