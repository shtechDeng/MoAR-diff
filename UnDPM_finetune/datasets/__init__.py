import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Subset
import numpy as np

from skimage.transform import resize
from skimage.exposure import rescale_intensity
import nibabel as nib
from torch.utils.data import Dataset

import shutil
import logging
import random
import time

def proc_month(month):
    bins = [4, 7, 13, 25, 120]
    for idx in range(len(bins)):
        if float(month) <= bins[idx]:
            return idx
        
class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def proc_nib_data(nib_data):
    p10 = np.percentile(nib_data, 10)
    p99 = np.percentile(nib_data, 99.9)

    nib_data[nib_data<p10] = p10
    nib_data[nib_data>p99] = p99

    nib_data = (nib_data - nib_data.min()) / (nib_data.max() - nib_data.min())

    nib_data = torch.tensor(nib_data, dtype=torch.float32)

    return nib_data

def nifti_rotate(src_img, xrot, yrot, zrot, save_img=None):
    home_path = os.getenv('HOME')
    # home_path = "home_path"
    temp_fold = os.path.join(home_path, '.nifti_rotate', str(int(time.time() * 1000000)) + '_' + str(random.randint(0, 1000000)))
    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)

    xrot = xrot*np.pi/180
    yrot = yrot*np.pi/180
    zrot = zrot*np.pi/180

    xmat = np.eye(3)
    ymat = np.eye(3)
    zmat = np.eye(3)

    xmat[1, 1] =  np.cos(xrot)
    xmat[1, 2] = -np.sin(xrot)
    xmat[2, 1] =  np.sin(xrot)
    xmat[2, 2] =  np.cos(xrot)

    ymat[0, 0] =  np.cos(yrot)
    ymat[0, 2] =  np.sin(yrot)
    ymat[2, 0] = -np.sin(yrot)
    ymat[2, 2] =  np.cos(yrot)

    zmat[0, 0] =  np.cos(zrot)
    zmat[0, 1] = -np.sin(zrot)
    zmat[1, 0] =  np.sin(zrot)
    zmat[1, 1] =  np.cos(zrot)

    mat = np.dot(np.dot(zmat, ymat), xmat)
    mat_all = np.eye(4)
    mat_all[:3, :3] = mat

    np.savetxt(os.path.join(temp_fold, 'rotate.mat'), mat_all, fmt="%.10f", delimiter="  ")

    if not save_img:
        save_img = os.path.join(temp_fold, 'rotate.nii.gz')

    cmd_applywarp = f"applywarp -i {src_img} -r {src_img} -o {save_img} --premat={os.path.join(temp_fold, 'rotate.mat')} --interp=spline"
    os.system(cmd_applywarp)

    ret_data = nib.load(save_img).get_fdata()
    shutil.rmtree(temp_fold)

    return ret_data
 
class MRI_Data(Dataset):
    def __init__(
        self, 
        data_path,
        data_file_name,
    ):
        self.data_path = data_path
        self.data_file_name = data_file_name

        self.data_list = []
        self.data_argu_main = {}
        self.data_argu_ref = {}
        self.sub2month = {}
        for subj in os.listdir(os.path.join(data_path, "SKD_T1w")):
            for age in os.listdir(os.path.join(data_path, "SKD_T1w", subj)):
                for nii in os.listdir(os.path.join(data_path, "SKD_T1w", subj, age)):
                    if('T1_artifact1' in nii): 
                        T1_path = os.path.join(data_path, "SKD_T1w", subj, age, "T1skull.nii.gz")
                        aT1_path = os.path.join(data_path, "SKD_T1w", subj, age, nii)
                        self.data_list.append((T1_path, aT1_path, age))
                    if('T1_artifact2' in nii): 
                        T1_path = os.path.join(data_path, "SKD_T1w", subj, age, "T1skull.nii.gz")
                        aT1_path = os.path.join(data_path, "SKD_T1w", subj, age, nii)
                        self.data_list.append((T1_path, aT1_path, age))
        for subj in os.listdir(os.path.join(data_path, "XM_T1w")):
            for age in os.listdir(os.path.join(data_path, "XM_T1w", subj)):
                for nii in os.listdir(os.path.join(data_path, "XM_T1w", subj, age)):
                    if('T1_artifact1' in nii): 
                        T1_path = os.path.join(data_path, "XM_T1w", subj, age, "T1skull.nii.gz")
                        aT1_path = os.path.join(data_path, "XM_T1w", subj, age, nii)
                        self.data_list.append((T1_path, aT1_path, age))
                    if('T1_artifact2' in nii): 
                        T1_path = os.path.join(data_path, "XM_T1w", subj, age, "T1skull.nii.gz")
                        aT1_path = os.path.join(data_path, "XM_T1w", subj, age, nii)
                        self.data_list.append((T1_path, aT1_path, age))

    def __getitem__(self, index):
        sub1, sub2, month = self.data_list[index]
        # ref_month = random.choice(self.sub2month[main_sub])

        data_path1 = sub1
        data1 = nib.load(data_path1).get_fdata()
        data1 = proc_nib_data(data1)

        data_path2 = sub2
        data2 = nib.load(data_path2).get_fdata()
        data2 = proc_nib_data(data2)
        
        return data1.unsqueeze(0), data2.unsqueeze(0), proc_month(month), data_path2

    def __len__(self):
        return len(self.data_list)
class MRI_Data_T2(Dataset):
    def __init__(
        self, 
        data_path,
        data_file_name,
    ):
        self.data_path = data_path
        self.data_file_name = data_file_name# 不需要

        self.data_list = []
        self.data_argu_main = {}
        self.data_argu_ref = {}
        self.sub2month = {}
        for subj in os.listdir(os.path.join(data_path, "SKD_T1w")):
            for age in os.listdir(os.path.join(data_path, "SKD_T1w", subj)):
                for nii in os.listdir(os.path.join(data_path, "SKD_T1w", subj, age)):
                    if('T2_masked.nii.gz' in nii): 
                        T1_path = os.path.join(data_path, "SKD_T1w", subj, age, "T1_masked.nii.gz")
                        aT1_path = os.path.join(data_path, "SKD_T1w", subj, age, nii)
                        self.data_list.append((T1_path, aT1_path, age))
        for subj in os.listdir(os.path.join(data_path, "XM_T1w")):
            for age in os.listdir(os.path.join(data_path, "XM_T1w", subj)):
                for nii in os.listdir(os.path.join(data_path, "XM_T1w", subj, age)):
                    if('T2_masked.nii.gz' in nii): 
                        T1_path = os.path.join(data_path, "XM_T1w", subj, age, "T1_masked.nii.gz")
                        aT1_path = os.path.join(data_path, "XM_T1w", subj, age, nii)
                        self.data_list.append((T1_path, aT1_path, age))
        logging.info(f"total:{len(self.data_list)}data!")

    def __getitem__(self, index):
        sub1, sub2, month = self.data_list[index]

        data_path1 = sub1
        data1 = nib.load(data_path1).get_fdata()
        data1 = proc_nib_data(data1)

        data_path2 = sub2
        data2 = nib.load(data_path2).get_fdata()
        data2 = proc_nib_data(data2)
        
        return data1.unsqueeze(0), data2.unsqueeze(0), proc_month(month), data_path2

    def __len__(self):
        return len(self.data_list)
   
class MRI_Data_inference(Dataset):
    def __init__(
            self,
            data_path,
            data_file_name,
    ):
        self.data_path = data_path
        self.data_file_name = data_file_name

        self.data_list = []
        self.data_argu_main = {}
        self.data_argu_ref = {}
        self.sub2month = {}
        for subj in os.listdir(data_path):
            for age in os.listdir(os.path.join(data_path, subj)):
                for nii in os.listdir(os.path.join(data_path, subj, age)):
                    if('fail1' in nii): 
                        T2_path = os.path.join(data_path, subj, age, "T2_masked.nii.gz")
                        T1_path = os.path.join(data_path, subj, age, nii)
                        self.data_list.append((T1_path, T2_path, age))

    def __getitem__(self, index):
        sub, sub2, month = self.data_list[index]
        # ref_month = random.choice(self.sub2month[main_sub])

        data_path = sub
        data_path2 = sub2
        data = nib.load(data_path).get_fdata()
        data = proc_nib_data(data)
        data2 = nib.load(data_path2).get_fdata()
        data2 = proc_nib_data(data2)
        
        return data.unsqueeze(0), data2.unsqueeze(0), proc_month(month), data_path

    def __len__(self):
        return len(self.data_list)


def get_dataset(args, config):

    dataset, test_dataset = MRI_Data_inference(config.data.data_path,config.data.data_file_name), None


    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
