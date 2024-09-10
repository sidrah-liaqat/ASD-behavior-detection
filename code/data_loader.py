from __future__ import print_function, division
import os
import torch
import pandas as pd

import numpy as np

from torch.utils.data import Dataset
from pathlib import Path
from clf import debug, action, fold, subfold

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file_dir, transform=None, split=None, win=1):
        """
        Args:
            csv_file_dir (string): Path to the csv files with annotations.
            fname_file (string): File with all the video file names.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # initialized to 0
        self.win = win

        ### for pre-defined data splits

        basetemp = pd.read_csv('/home/sidrah/DL/synchrony/data_splits/fold{}/behaviortrain_fold{}.txt'.format(fold,fold), header=None)
        #basetemp = basetemp[0].str.replace('.mpg', '')
        self.train = pd.DataFrame(columns=['filename'])
        self.train.filename = basetemp
        basetemp = pd.read_csv('/home/sidrah/DL/synchrony/data_splits/fold{}/asdtrain_fold{}_subfold{}.txt'.format(fold,fold,subfold), header=None)
        #basetemp = basetemp[0].str.replace('.mpg', '')
        self.test = pd.DataFrame(columns=['filename'])
        self.test.filename = basetemp
        basetemp = pd.read_csv('/home/sidrah/DL/synchrony/data_splits/fold{}/test_fold{}_subfold{}.txt'.format(fold,fold,subfold), header=None)
        #basetemp = basetemp[0].str.replace('.mpg', '')
        self.val = pd.DataFrame(columns=['filename'])
        self.val.filename = basetemp

        self.uselist = getattr(self, split)
        self.uselist.reset_index(drop=True, inplace=True)

        # check the self.uselist.filename against forbidden ADHD subject ids and remove those
        ADHD_ids = [str(x) for x in [76065, 76071, 76104, 76199, 76200, 76229, 76271, 76298, 76313, 76319, 76339, 76350, 76367, 76373,
                    76374, 76386, 76391, 76423, 76426, 76427, 76429, 79075, 79090, 79100, 79139, 79155, 79159, 79162,
                    79213, 79275, 79308, 79311, 79327, 79341, 79354, 79357, 79358, 79375, 79377, 79380, 79384, 79403,
                    79404, 79412, 79419, 79422, 79434]]
        split_subjectid = self.uselist.filename.str.split(pat='_', expand=True)
        s = split_subjectid[0].isin(ADHD_ids)
        del_idx = s[s].index.values

        print('Deleting {} files'.format(len(del_idx)))

        self.uselist.drop(del_idx, inplace=True)
        self.uselist.reset_index(drop=True, inplace=True)
        self.uselist.dropna(inplace=True)
        self.uselist.reset_index(drop=True, inplace=True)


        self.label = ['look_face', 'look_object', 'smile', 'vocal']
        self.C_x_fc = []
        for i in range(68): self.C_x_fc.append('C_ x_{n}'.format(n=i))
        self.C_y_fc = []
        for i in range(68): self.C_y_fc.append('C_ y_{n}'.format(n=i))

        self.C_x_eye = []
        for i in range(56): self.C_x_eye.append('C_ eye_lmk_x_{n}'.format(n=i))
        self.C_y_eye = []
        for i in range(56): self.C_y_eye.append('C_ eye_lmk_y_{n}'.format(n=i))

        self.head_cols = ['tl_x', 'tl_y', 'br_x', 'br_y']
        self.gaze_angle_cols = ['C_ gaze_angle_x', 'C_ gaze_angle_y']

        self.AU = ['C_ AU01_r', 'C_ AU02_r', 'C_ AU04_r', 'C_ AU05_r', 'C_ AU06_r', 'C_ AU07_r', 'C_ AU09_r',
                   'C_ AU10_r', 'C_ AU12_r', 'C_ AU14_r', 'C_ AU15_r', 'C_ AU17_r', 'C_ AU20_r', 'C_ AU23_r',
                   'C_ AU25_r', 'C_ AU26_r', 'C_ AU45_r',
                   'C_ AU01_c', 'C_ AU02_c', 'C_ AU04_c', 'C_ AU05_c', 'C_ AU06_c', 'C_ AU07_c', 'C_ AU09_c',
                   'C_ AU10_c', 'C_ AU12_c', 'C_ AU14_c', 'C_ AU15_c', 'C_ AU17_c', 'C_ AU20_c', 'C_ AU23_c',
                   'C_ AU25_c', 'C_ AU26_c', 'C_ AU28_c', 'C_ AU45_c']
        # file exists check
        my_file = Path(csv_file_dir + self.uselist.filename[0] + '.csv')
        if my_file.is_file() == False:
            print('First file {} is not in the feature directory'.format(str(self.uselist.filename[0] + '.csv')))

        # read all files to store length for indexing later
        self.landmarks_frame = pd.read_csv(csv_file_dir + self.uselist.filename[0] + '.csv',
                                           header=0, index_col=['frame'],
                                           # usecols= ['frame', 'valid']+self.label + self.C_x_fc + self.C_y_fc +
                                           #         self.C_x_eye + self.C_y_eye + self.head_cols + self.AU)
                                           usecols=['frame', 'valid', 'C_ confidence'] + self.label +
                                                   self.C_x_fc + self.C_y_fc + self.C_x_eye + self.C_y_eye +
                                                   self.AU + self.head_cols + self.gaze_angle_cols
                                           )
        self.landmarks_frame.drop(self.landmarks_frame[self.landmarks_frame.valid == 0].index, inplace=True)

        #print(str(self.uselist.filename[testind]))
        # tried class balancing approach of dropping negative examples to make equal positive and negative
        # It didn't work as good as using class weights during training
        #pos = (self.landmarks_frame[action] == 1).sum()
        #if pos < self.landmarks_frame.shape[0] / 2:
        #    neg_df = self.landmarks_frame[self.landmarks_frame[action] == 0].sample(n=pos)
        #else:
        #    neg_df = self.landmarks_frame[self.landmarks_frame[action] == 0]
        # dropping negative examples to make them equal to positive examples per file
        #self.landmarks_frame.drop(self.landmarks_frame.loc[~self.landmarks_frame.index.isin(
        #    neg_df.index | self.landmarks_frame[self.landmarks_frame[action] == 1].index)].index, inplace=True)

        print('# of {} set files: {}'.format(split,str(self.uselist.filename.shape[0])))
        if (debug):
            offset = 50
            if(split=='train'):
                ex_cnt = 3
            elif(split=='val'):
                ex_cnt = 3
            print('{} examples in split : {}'.format(ex_cnt, split))
            starttr = offset; endtr = offset + ex_cnt
        else:
            starttr = 1; endtr = len(self.uselist.filename)
            print('{} examples in split : {}'.format(len(self.uselist.filename), split))
        for i in range(starttr, endtr):
            my_file = Path(csv_file_dir + self.uselist.filename[i] + '.csv')
            if my_file.is_file() == False:
                print('{} is not in the feature directory'.format(str(self.uselist.filename[i] + '.csv')))
                continue
            abc = pd.read_csv(csv_file_dir + self.uselist.filename[i]+'.csv',
                              header=0, index_col=['frame'],
                              usecols=['frame', 'valid', 'C_ confidence'] + self.label + self.C_x_fc + self.C_y_fc +
                                      self.C_x_eye + self.C_y_eye +
                                      self.AU + self.head_cols + self.gaze_angle_cols)
            abc.drop(abc[abc.valid == 0].index, inplace=True)
            #pos = (abc[action] == 1).sum()
            #if pos < abc.shape[0] / 2:
            #    neg_df = abc[abc[action] == 0].sample(n=pos)
            #else:
            #    neg_df = abc[abc[action]== 0]

            #abc.drop(abc.loc[~abc.index.isin(
            #    neg_df.index | abc[abc[action] == 1].index)].index, inplace=True)

            self.landmarks_frame = self.landmarks_frame.append(abc)
        self.landmarks_frame.reset_index(drop=True, inplace=True)
        print('For task {}, Positive : {}, Negative : {}'.format(
                        self.label[action],
                        str((self.landmarks_frame[self.label[action]]==1).sum()),
                        str((self.landmarks_frame[self.label[action]]==0).sum())))
        self.transform = transform

    def __len__(self):
        return int(np.ceil(self.landmarks_frame.shape[0]/self.win))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx * self.win
        #print('{}'.format(idx))
        # in case idx is such that there aren't enough frames for this batch, shift idx back as needed
        if (idx+self.win) > self.landmarks_frame.shape[0]: idx = self.landmarks_frame.shape[0]-self.win

        # there is no provision to load image in this code
        lmk_x = np.asarray(self.landmarks_frame.loc[idx:idx+self.win-1, self.C_x_fc], dtype=float)
        lmk_y = np.asarray(self.landmarks_frame.loc[idx:idx+self.win-1, self.C_y_fc], dtype=float)
        eye_lmk_x = np.asarray(self.landmarks_frame.loc[idx:idx+self.win-1, self.C_x_eye], dtype=float)
        eye_lmk_y = np.asarray(self.landmarks_frame.loc[idx:idx+self.win-1, self.C_y_eye], dtype=float)
        headpose = np.asarray(self.landmarks_frame.loc[idx:idx+self.win-1, self.head_cols], dtype=float)
        gazeangle = np.asarray(self.landmarks_frame.loc[idx:idx + self.win - 1, self.gaze_angle_cols], dtype=float)
        conf = np.asarray(self.landmarks_frame.loc[idx:idx + self.win - 1, 'C_ confidence'], dtype=float)


        au = np.asarray(self.landmarks_frame.loc[idx:idx+self.win-1, self.AU], dtype=float)
        # reshaping operations
        headpose = headpose.reshape(-1, 2)
        landmarks = np.vstack((lmk_x, lmk_y))
        landmarks = landmarks.transpose((1,0))
        eyelandmarks = np.vstack((eye_lmk_x, eye_lmk_y))
        eyelandmarks = eyelandmarks.transpose((1,0))

        # label for training
        sep_action = np.asarray(self.landmarks_frame.loc[idx:idx+self.win-1, self.label], dtype=int)


        #sample = {'image' : image, 'landmarks' : landmarks,
        sample = {
                  'landmarks': landmarks,
                  'eyelandmarks' : eyelandmarks,
                  'headpose' : headpose,
                  'gazeangle' : gazeangle,
                  'au' : au,
                  'action' : sep_action,
                  'conf': conf
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
