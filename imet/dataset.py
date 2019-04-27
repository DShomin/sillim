import torch
import pandas as pd
import torch.utils.data as data

from PIL import Image
from pathlib import Path


# This loader is to be used for serving image tensors (ex img - y1[tag], y2[culture])
# culture label range ( 0 ~ 397 )
# tag label range( 398 ~ 1103 )
class IMetDataset(data.Dataset):

    """
    csv_path : pathlib.Path 
    root_dir : pathlib.Path
    device : string [example gpu -> 'cuda:0' cpu -> 'cpu:0']
    mode : string [example 'train', 'val', 'test'] val & test not yet
    double_label : bool [True -> return tag label & culture label, False -> return label]
    """

    def __init__(self, csv_path : Path, root_dir: Path, device="cuda:0", mode='train' , 
                    double_label=False, transform=None):
        df = pd.read_csv(csv_path)
        self.img_name = df.id.map('{}.png'.format).values
        if 'attribute_ids' in df.columns:
            self.img_label = df.attribute_ids.map(lambda x: x.split()).values
            self.img_label_flag = True
        else:
            self.img_label = None
            self.img_label_flag = False
        self.root_dir = root_dir
        self.device = device
        self.double_label = double_label
        self.df_len = df.shape[0]
        if transform != None:
            self.transform = transform[mode]
        else:
            self.transform = transform
        
    def __len__(self):
        return self.df_len
    
    def __getitem__(self, idx):
        img_id = self.img_name[idx]
        file_name = self.root_dir / img_id
        img = Image.open(file_name)
        label = self.img_label[idx]
        if self.double_label and self.img_label_flag != False:
            label_cul_tensor = torch.zeros((398))
            label_tag_tensor = torch.zeros((705))
            for i in label:
                if int(i) <= 397:
                    label_cul_tensor[int(i)] = 1
                else:
                    label_tag_tensor[int(i) - 398] = 1
            label_cul_tensor = label_cul_tensor.to(self.device)
            label_tag_tensor = label_tag_tensor.to(self.device)
            if self.transform:
                img = self.transform(img)
            img = img.to(self.device)
            return [img, label_cul_tensor, label_tag_tensor]
        else:
            label_tensor = torch.zeros((1103))
            for i in label:
                label_tensor[int(i)] = 1
            label_tensor = label_tensor.to(self.device)
            if self.transform:
                img = self.transform(img)
            img = img.to(self.device)
            return [img, label_tensor]

import torchvision

class IMetDataLoader:
    def __init__(self, comfig):
        """
        param config (All String)
        device : 'gpu' or anything(cpu)
        mode : 'train' or 'val' or 'test'
        csv_path : csv file path
        root_dir : image directory path
        double_label : True or False
        batch_size : 32
        """

        self.config = config
        if config.device == 'gpu' and torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        shuffle = False
        if config.mode == 'train':
            data_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225])
            ])
            shuffle = True
        elif config.mode == 'val':
            data_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225])
            ])
        elif config.mode == 'test':
            raise NotImplementedError("This mode is not implemented YET")
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

        if config.double_label == 'True':
            flag = True
        else:
            flag = False

        dataset = IMetDataset(csv_path=Path(config.csv_path), root_dir=Path(config.root_dir), device=device, 
                    mode=config.mode, transform=data_transforms, double_label=flag)

        data_loader = data.DataLoader(dataset=dataset, batch_size=int(config.batch_size), shuffle=shuffle)
        return data_loader