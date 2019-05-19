
import pandas as pd
import numpy as np
import os

from pathlib import Path
from imet.dataset import TrainDataset, TTADataset, get_ids, N_CLASSES, DATA_ROOT
from imet.transforms import get_transform
from torchvision import transforms
from imet.utils import ( ThreadingDataLoader as DataLoader)
from PIL import Image


inv_normalize = transforms.Compose(
        [
            transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
        ]
    )

def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)

def test_dataloader():
    run('python -m imet.make_folds')
    folds = pd.read_csv('./folds.csv')
    print(folds)


def test_augmentation():
    folds = pd.read_csv('./folds.csv')
    train_root = Path('./data/train')
    output_dir = Path('./output')
    train_transform = get_transform(
        transform_list='keep_aspect, horizontal_flip, random_rotate',
    )
    
    def make_loader(df: pd.DataFrame, image_transform) -> DataLoader:
        return DataLoader(
            TrainDataset(train_root, df, image_transform, debug=False),
            shuffle=True,
            batch_size=1,
            num_workers=1,
        )
    loader = make_loader(folds, train_transform)
    output_dir.mkdir(exist_ok=True, parents=True)

    for i in range(10):
        img, label = loader._get_item(i)
        img = np.transpose(inv_normalize(img.cpu()).numpy()*255,(1,2,0))
        img = Image.fromarray(np.uint8(img))
        img.save(f'{output_dir}/{str(i).zfill(3)}.png')
        


if __name__ == '__main__':
    #test_dataloader()
    test_augmentation()