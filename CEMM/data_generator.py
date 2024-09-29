# -*- coding: utf-8 -*-
import os
import itertools
import glob
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


# Base data generator with common methods and variables
class BaseDataGenerator(Dataset):
    def __init__(self, data_df, data_root, data_transform=None, default_size=64, ap_th=44, pp_th=80, vp_th=36):
        df = data_df.copy()
        df['Ki67max'] *= 100
        df['Ki67Label'] = df['Ki67max'].apply(lambda x: 0 if x < 20 else 1)
        df['ID'] = df['ID'].astype(str)

        self.case_dict = {str(id): [event] for id, event in zip(df['ID'], df['Ki67Label'])}
        self.data_root = data_root
        self.th = {'ap': ap_th, 'pp': pp_th, 'vp': vp_th}
        self.default_size = default_size
        self.transform = data_transform
        self.file_list = self.create_file_list(data_root)

    def create_file_list(self, data_root):
        raise NotImplementedError("Subclasses should implement this method")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_info = self.file_list[index]
        key = file_info[0]
        label = int(self.case_dict[key][0])

        # Load image data
        ap_img = self.load_data(file_info[1], 'ap')
        pp_img = self.load_data(file_info[2], 'pp')
        vp_img = self.load_data(file_info[3], 'vp')

        # Apply transformations
        if self.transform:
            [ap_img, pp_img, vp_img] = self.transform([ap_img, pp_img, vp_img])

        # Convert to tensor
        (ap_img, pp_img, vp_img) = self.to_tensor([ap_img, pp_img, vp_img])

        return [ap_img, pp_img, vp_img], label, key

    def load_data(self, path, phase):
        depth = self.th[phase]
        if path is None:
            return np.zeros([depth, self.default_size, self.default_size], dtype=np.float32)

        img_3d = loadmat(path)['subset_data'].astype(np.float32)

        if np.ndim(img_3d) == 2:
            img_3d = np.expand_dims(img_3d, axis=2)

        img = img_3d.transpose(2, 0, 1)

        if img.shape[1] != self.default_size or img.shape[2] != self.default_size:
            img = np.zeros((depth, self.default_size, self.default_size))

        # Adjust image depth based on phase
        if phase == 'ap':
            img = self.adjust_image_depth(img, depth, pad=True)
        else:
            img = self.adjust_image_depth(img, depth, pad=False)

        return img

    def adjust_image_depth(self, img, depth, pad=True):
        if img.shape[0] < depth:
            img = np.pad(img, ((0, depth - img.shape[0]), (0, 0), (0, 0)), mode='edge' if not pad else 'constant')
        elif img.shape[0] > depth:
            start_p = random.randint(0, img.shape[0] - depth)
            img = img[start_p:start_p + depth, :, :]
        return img

    def to_tensor(self, inputs):
        return [torch.from_numpy(i).unsqueeze(0).float() if i.ndim == 3 else torch.from_numpy(i).float() for i in
                inputs]

    def get_usable_files(self, dir_root, use_num=100):
        all_files = glob.glob(os.path.join(dir_root, '*.mat'))
        return all_files[:use_num] if len(all_files) >= use_num else all_files or [None]


# Training data generator
class Train_Data_Generator(BaseDataGenerator):
    def create_file_list(self, data_root):
        file_list = []
        for each_case in os.listdir(data_root):
            key = each_case
            if key not in self.case_dict.keys():
                continue

            case_path = os.path.join(data_root, each_case)
            ap_list = self.get_usable_files(os.path.join(case_path, 'arterial_phase'))
            pp_list = self.get_usable_files(os.path.join(case_path, 'portal_phase'))
            vp_list = self.get_usable_files(os.path.join(case_path, 'venous_phase'))

            for i in range(min(len(ap_list), len(pp_list), len(vp_list))):
                file_list.append([key, ap_list[i], pp_list[i], vp_list[i]])

        random.shuffle(file_list)  # Shuffle for randomness
        return file_list


# Validation/Test data generator
class Test_Data_Generator(BaseDataGenerator):
    def __init__(self, data_df, data_root, data_transform=None, default_size=64, ap_th=44, pp_th=80, vp_th=36,
                 use_num=1):
        self.use_num = use_num
        super().__init__(data_df, data_root, data_transform, default_size, ap_th, pp_th, vp_th)

    def create_file_list(self, data_root):
        file_list = []
        for each_case in os.listdir(data_root):
            key = each_case
            if key not in self.case_dict.keys():
                continue

            case_path = os.path.join(data_root, each_case)
            ap_list = self.get_usable_files(os.path.join(case_path, 'arterial_phase'), self.use_num)
            pp_list = self.get_usable_files(os.path.join(case_path, 'portal_phase'), self.use_num)
            vp_list = self.get_usable_files(os.path.join(case_path, 'venous_phase'), self.use_num)

            for ap, pp, vp in itertools.product(ap_list, pp_list, vp_list):
                file_list.append([key, ap, pp, vp])
        return file_list