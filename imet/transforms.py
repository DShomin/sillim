import random
import math

from PIL import Image, ImageOps
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomRotation, RandomVerticalFlip, RandomOrder, RandomApply,
    RandomHorizontalFlip)
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_gamma,
    adjust_hue,
    adjust_saturation
)

class RandomSizedCrop:
    """Random crop the given PIL.Image to a random size
    of the original size and and a random aspect ratio
    of the original aspect ratio.
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR,
                 min_aspect=4/5, max_aspect=5/4,
                 min_area=0.25, max_area=1):
        self.size = size
        self.interpolation = interpolation
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.min_area, self.max_area) * area
            aspect_ratio = random.uniform(self.min_aspect, self.max_aspect)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Resize(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))

class RandomRotate:
    def __init__(self):
        self.degrees = [0, 90, 180, 270]
    def __call__(self, img):
        degree = random.choice(self.degrees)
        return img.transpose(degree)

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
        augment_ratio=0.5
        ):
    transform = list()
    transform_list = transform_list.split(', ')
    augments = list()

    for transform_name in transform_list:
        if transform_name == 'random_crop':
            transform.append(RandomCrop(target_size))
        elif transform_name == 'keep_aspect':
            transform.append(KeepAsepctResize(target_size))
        elif transform_name == 'horizontal_flip':
            augments.append(RandomHorizontalFlip())
        elif transform_name == 'vertical_flip':
            augments.append(RandomVerticalFlip())
        elif transform_name == 'random_rotate':
            augments.append(RandomRotate())
        elif transform_name == 'random_shift':
            pass
        elif transform_name == 'random_scale':
            pass
        elif transform_name == 'color_jitter':
            pass
    transform.append(RandomApply(augments, p=augment_ratio))    
    
    return Compose(transform)