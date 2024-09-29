# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import torch
from data_generator import Train_Data_Generator, Test_Data_Generator
from torch.utils.data import DataLoader
from augment import Augment
from torchvision import transforms
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.model_selection import train_test_split
import yaml
import pickle

# Set experiment name
EXP = 'RA_EXP'

# Set seed for reproducibility
def set_seed(seed=48):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()  # Set random seed

# Parameters
num_epochs = 100
batch_size = 1
val_interval = 1

# Dataset paths (relative paths for portability)
df_path = './data/train+valid.csv'
df_root = './data/dataset'
df = pd.read_csv(df_path, encoding='utf-8')
test_df = pd.read_csv('./data/456-168_ex-valid.csv', encoding='utf-8')
df_train, df_valid = train_test_split(df, test_size=0.3, random_state=48)

# Data augmentation and preprocessing
Aug = Augment()
train_transform = transforms.Compose([transforms.Lambda(lambda x: Aug.z_score_normalization(x))])
valid_transform = transforms.Compose([transforms.Lambda(lambda x: Aug.z_score_normalization(x))])

# Data loaders
kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = DataLoader(Test_Data_Generator(df_train, df_root, train_transform), batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
test_loader = DataLoader(Test_Data_Generator(df_valid, df_root, valid_transform), batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
ex_test_loader = DataLoader(Test_Data_Generator(test_df, df_root, valid_transform), batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

# PyRadiomics feature extractor setup (read YAML config)
with open('./config/ra_config.yaml', 'r') as f:
    params = yaml.safe_load(f)
extractor = featureextractor.RadiomicsFeatureExtractor(params)

# Function to extract features from a single 2D slice
def extract_features_single_slice(image_3d, mask_2d, feature_names):
    middle_slice_idx = image_3d.shape[2] // 2
    image_2d = image_3d[:, :, middle_slice_idx]
    sitk_image_2d = sitk.GetImageFromArray(image_2d)
    feature_vector = extractor.execute(sitk_image_2d, mask_2d)
    return {key: feature_vector[key] for key in feature_names}

# Prepare mask and feature list
dataiter = iter(train_loader)
data, labels, patient_id = next(dataiter)
example_image = sitk.GetImageFromArray(data[0, 0, :, :, :].numpy())
mask_slice = np.ones(data[0, 0, :, :, :].shape[-2:], dtype=np.int8)
mask_slice[-1, :] = 0
sitk_mask = sitk.GetImageFromArray(mask_slice)
example_features = extractor.execute(example_image, sitk_mask)
feature_names = [key for key, value in example_features.items() if isinstance(value, (int, float, np.ndarray))]
pd.DataFrame(feature_names, columns=["Column_Name"]).to_csv("feature_names_2D.csv", index=False)

# Store features, labels, and IDs
def store_data(loader, filename_prefix):
    features, labels, ids = [], [], []
    for data, target, patient_id in loader:
        features_list = [extract_features_single_slice(data[i].numpy(), sitk_mask, feature_names) for i in range(len(data))]
        features.append(features_list)
        labels.append(target.numpy())
        ids.append(patient_id)
    with open(f'{filename_prefix}_features.pkl', 'wb') as f:
        pickle.dump(features, f)
    with open(f'{filename_prefix}_labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open(f'{filename_prefix}_ids.pkl', 'wb') as f:
        pickle.dump(ids, f)

# Store training, validation, and external test data
store_data(train_loader, 'train')
store_data(test_loader, 'test')
store_data(ex_test_loader, 'external_test')