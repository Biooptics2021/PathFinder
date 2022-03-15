import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


INFO_PATH = ''#clinical info
HEATMAP_PATH = ''#heatmaps .npy file

########################
#His-seg Heatmap Loader(MacroNet)
########################
class SegHeatmapDatasetLoader(Dataset):
    def __init__(self, seg_filepaths, transform=None):
        super(SegHeatmapDatasetLoader, self).__init__()
        self.seg_filepaths = seg_filepaths
        self.transform = transform


    def __len__(self):
        return len(self.seg_filepaths)


    def __getitem__(self, idx):
        seg_filepath = self.seg_filepaths[idx]
        seg = np.load(seg_filepath)

        if self.transform is not None:
            seg = self.transform(image=seg)["image"]

        hospital = seg_filepath.split('/')[-2]  #The training set may from different hospital
        base_dir = INFO_PATH
        self.data = pd.read_csv(base_dir + hospital + '.csv')

        if hospital == 'TCGA':
            ID = seg_filepath.split('/')[-1][:12]
            pd_index = self.data[self.data['WSIs'].isin([ID])].index.values[0]
            T = (self.data['days_to_death'][pd_index] + self.data['days_to_last_follow_up'][pd_index])/30
            O = (~self.data['vital_status'][pd_index].astype(bool)).astype(int)
        else:
            ID = seg_filepath.split('/')[-1][:-7]
            pd_index = self.data[self.data['WSIs'].isin([ID])].index.values[0]
            T = self.data['death_time'][pd_index]
            O = self.data['death_status'][pd_index]

        O = torch.tensor(O).type(torch.FloatTensor)
        T = torch.tensor(T).type(torch.FloatTensor)

        return seg, T, O, seg_filepath


########################
#Tumour patches Loader(MicroNet)
########################
class PatchesImageDatasetLoader(Dataset):
    def __init__(self, patches_filepaths, transform=None):
        super(PatchesImageDatasetLoader, self).__init__()
        self.images_filepaths = patches_filepaths
        self.transform = transform


    def get_files(self, path, rule=".png"):
        all = []
        for fpathe,dirs,fs in os.walk(path):
            for f in fs:
                filename = os.path.join(fpathe,f)
                if filename.endswith(rule):
                    all.append(filename)
        return all    


    def __len__(self):
        return len(self.images_filepaths)


    def __getitem__(self, idx):
        patches_filepath = self.images_filepaths[idx]
        patches_paths = self.get_files(patches_filepath)[:16]
        hospital = patches_filepath.split('/')[-2]
        base_dir = INFO_PATH
        self.data = pd.read_csv(base_dir+hospital+'.csv')
        
        Imgs = []
        for path in patches_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                img = self.transform(image=img)["image"]
            img = np.array(img)

            Imgs.append(img)
        Imgs = np.array(Imgs).astype('float64')
        Imgs = Imgs.reshape((16*3,512,512))

        if hospital == 'TCGA':
            ID = patches_filepath.split('/')[-1][:12]
            pd_index = self.data[self.data['WSIs'].isin([ID])].index.values[0]
            T = (self.data['days_to_death'][pd_index] + self.data['days_to_last_follow_up'][pd_index])/30
            O = (~self.data['vital_status'][pd_index].astype(bool)).astype(int)
        else:
            ID = patches_filepath.split('/')[-1][:-3]
            pd_index = self.data[self.data['WSIs'].isin([ID])].index.values[0]
            T = self.data['death_time'][pd_index]
            O = self.data['death_status'][pd_index]

        return Imgs, T, O, patches_paths


########################
#Fusion dataloader(M2MNet)
########################
class FusionDatasetLoader(Dataset):

    def __init__(self, patches_filepaths, transform_patch=None, transform_seg=None):
        super(FusionDatasetLoader, self).__init__()

        self.images_filepaths = patches_filepaths
        self.transform_patch = transform_patch
        self.transform_seg = transform_seg


    def get_files(self, path, rule=".png"):
        all = []
        for fpathe,dirs,fs in os.walk(path):
            for f in fs:
                filename = os.path.join(fpathe,f)
                if filename.endswith(rule):
                    all.append(filename)
        return all    


    def __len__(self):
        return len(self.images_filepaths)


    def __getitem__(self, idx):
        ########################patches#########################
        patches_filepath = self.images_filepaths[idx]
        patches_paths = self.get_files(patches_filepath)[:16]
        hospital = patches_filepath.split('/')[-2]
        base_dir = INFO_PATH
        self.data = pd.read_csv(base_dir+hospital+'.csv')
        
        Imgs = []
        for path in patches_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform_patch is not None:
                img = self.transform_patch(image=img)["image"]
            img = np.array(img)

            Imgs.append(img)
        Imgs = np.array(Imgs).astype('float64')
        Imgs = Imgs.reshape((16*3,512,512))

        if hospital == 'TCGA':
            ID = patches_filepath.split('/')[-1][:12]
            pd_index = self.data[self.data['WSIs'].isin([ID])].index.values[0]
            T = (self.data['days_to_death'][pd_index] + self.data['days_to_last_follow_up'][pd_index])/30
            O = (~self.data['vital_status'][pd_index].astype(bool)).astype(int)
        else:
            ID = patches_filepath.split('/')[-1][:-3]
            pd_index = self.data[self.data['WSIs'].isin([ID])].index.values[0]
            T = self.data['death_time'][pd_index]
            O = self.data['death_status'][pd_index]
        
        ########################segmaps#########################
        base_seg_path = HEATMAP_PATH
        if hospital == 'TCGA':
            seg_filepath = base_seg_path + patches_filepath.split('/')[-2] + '/' + patches_filepath.split('/')[-1][:-1] + '.npy'
        else:
            seg_filepath = base_seg_path + patches_filepath.split('/')[-2] + '/' + patches_filepath.split('/')[-1] + '.npy'
        seg = np.load(seg_filepath)

        if self.transform_seg is not None:
            seg = self.transform_seg(image=seg)["image"]

        return Imgs, seg, T, O, patches_paths, seg_filepath

