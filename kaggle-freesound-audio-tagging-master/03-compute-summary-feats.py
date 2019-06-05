"""Compute summary metrics of various spectral and time based features"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from summary_feats_funcs import all_feats

# number of cores to use
num_cores = int(sys.argv[1]) if len(sys.argv) > 1 else 1

print('Train...')
train = pd.read_csv('./data/train_curated.csv')

train_feats = Parallel(n_jobs=num_cores)(
    delayed(all_feats)('./data/audio_train_trim/' + x + '.npy')
    for x in tqdm(train.fname.values))

train_feats_df = pd.DataFrame(np.vstack(train_feats))
train_feats_df['fname'] = pd.Series(train.fname.values, index=train_feats_df.index)
train_feats_df.to_pickle('./data/train_tab_feats.pkl')


print('Test...')
samp_subm = pd.read_csv("./data/sample_submission.csv")

test_feats = Parallel(n_jobs=num_cores)(
    delayed(all_feats)('./data/audio_test_trim/' + x + '.npy')
    for x in tqdm(samp_subm
                  .fname.values))


test_feats_df = pd.DataFrame(np.vstack(test_feats))
test_feats_df['fname'] = pd.Series(samp_subm.fname.values,
                                   index=test_feats_df.index)
test_feats_df.to_pickle('./data/test_tab_feats.pkl')
