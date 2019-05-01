import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import PIL
from tqdm import tqdm
from pathlib import Path

import os

def get_img_size(img_id, path):
            
    img = PIL.Image.open(f'{path}{img_id}.png')
    
    return img.size

def split_shape_col(df:pd.DataFrame,):
    assert df.shape == None, "don't have shape column"
    
    width_list = [x[0] for x in df['shape'].values]
    height_list = [x[1] for x in df['shape'].values]
    
    df['width'] = width_list
    df['height'] = height_list

    del df['shape']
    
    return df

def add_aspect_ratio_col(df:pd.DataFrame):
    assert df.height == None or df.width == None, "don't have height or width column"
    df['aspect_ratio'] = df.height / df.width

    return df

def main():
    # define path csv & file directory
    train_csv_path = Path('../data/train.csv')
    
    train_file_path = Path('../data/train')
    test_file_path = Path('../data/test')

    # define train set & test set
    df_train = pd.read_csv(train_csv_path)
    df_train['shape'] = df_train['id'].progress_apply(lambda x : get_img_size(x, train_file_path))
    df_train = split_shape_col(df_train)
    df_train = add_aspect_ratio_col(df_train)

    df_test = pd.DataFrame()
    df_test['id'] = [x.split('.')[0] for x in os.listdir(test_file_path)]
    df_test['shape'] = df_test['id'].progress_apply(lambda x : get_img_size(x, test_file_path))
    df_test = add_aspect_ratio_col(df_test)

    # lowest ratio : 0.079428
    # maximum ratio : 9.620000
    # make range
    aspect_ratio_range = np.arange(0.05, 9.66, 0.05)

    # range ratio value count
    range_count = df_test.groupby(pd.cut(df_test.aspect_ratio, aspect_ratio_range)).aspect_ratio.count()

    # create validation set csv
    df_val = pd.DataFrame()
    for idx, count_val in enumerate(range_count):
    
        low_bound = aspect_ratio_range[idx]
        high_bound = low_bound + 0.05
        
        append_idx = df_train.loc[(df_train.aspect_ratio >= low_bound) & (df_train.aspect_ratio < high_bound)].index
        
        if count_val == 0 or append_idx.shape[0] == 0:
            continue
        else:
            append_df = df_train.loc[(df_train.aspect_ratio >= low_bound) & (df_train.aspect_ratio < high_bound)]
            df_train = df_train.drop(index=append_idx)
            df_val = df_val.append(append_df.sample(count_val, random_state=42))
    # clear columns
    df_val = df_val.drop(columns=['width', 'height', 'aspect_ratio'])
    df_val.to_csv('../data/validation.csv')



if __name__ == '__main__':
    main()