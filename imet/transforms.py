import random
import math

from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomRotation, RandomVerticalFlip, RandomOrder,
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


class RandomeFunctional:

    def __init__(self, augmentations=[
        'hue',
        'contrast',
        'brightness',
        'gamma',
        'saturation'
    ]):
        self.transformations = []
        self.contrast_factor_range = (0.5, 1.5) # 0: gray, 1: original, 2: 2increased
        self.brightness_factor_range = (0.5, 1.5)
        self.gamma_range = (0.5, 1.5)
        self.hue_factor_range = (-0.5, 0.5)
        self.saturation_range = (0.5, 1.5) 

        for augments in augmentations:
            if augments == 'hue':
                self.transformations.append(
                    self._hue_transform
                )
            elif augments == 'contrast':
                self.transformations.append(
                    self._contrast_transform
                )
            elif augments == 'brightness':
                self.transformations.append(
                    self._contrast_transform
                )
            elif augments == 'gamma':
                self.transformations.append(
                    self._contrast_transform
                )
            elif augments == 'saturation':
                self.transformations.append(
                    self._contrast_transform
                )
    def _brightness_transform(self, img):
        return adjust_brightness(img, random.uniform(self.brightness_factor_range))
        
    def _contrast_transform(self, img):
        return adjust_contrast(img, random.uniform(self.contrast_factor_range))
    
    def _hue_transform(self, img):
        return adjust_hue(img, random.uniform(self.hue_factor_range))
    
    def _saturation_transform(self, img):
        return adjust_saturation(img, random.uniform(self.saturation_range))
    
    def _saturation_transform(self, img):
        return adjust_saturation(img, random.uniform(self.saturation_range))
    
    def _gamma_transform(self, img):
        return adjust_gamma(img, random.uniform(self.gamma_range))

    def __call__(self, img):
        for transform in self.transformations:
            img = transform(img)
        return img 


train_transform = RandomOrder(Compose([
    RandomCrop(288),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    Scale(),
    RandomRotation((-50,50)),
    RandomeFunctional
]))


test_transform = Compose([
    RandomCrop(288),
    RandomHorizontalFlip(),
    RandomRotation((-50,50))
])


tensor_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
