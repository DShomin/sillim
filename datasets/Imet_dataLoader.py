import torch
from PIL import Image
import torch.utils.data as data
from pathlib import Path

# This loader is to be used for serving image tensors (ex img - y1[tag], y2[culture])
# culture label range ( 0 ~ 397 )
# tag label range( 398 ~ 1103 )
class IMetDataset(data.Dataset):
    def __init__(self, csv_path : Path, root_dir: Path, device="cuda:0", mode='train' , double_label=False, transform=None):
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
            label_cul_tensor = torch.zeros((1, 398))
            label_tag_tensor = torch.zeros((1, 705))
            for i in label:
                if int(i) <= 397:
                    label_cul_tensor[0, int(i)] = 1
                else:
                    label_tag_tensor[0, int(i) - 398] = 1
            label_cul_tensor = label_cul_tensor.to(self.device)
            label_tag_tensor = label_tag_tensor.to(self.device)
            if self.transform:
                img = self.transform(img)
            return [img, label_cul_tensor, label_tag_tensor]
        else:
            label_tensor = torch.zeros((1,1103))
            for i in label:
                label_tensor[0, int(i)] = 1
            label_tensor = label_tensor.to(self.device)
            if self.transform:
                img = self.transform(img)
            return [img, label_tensor]