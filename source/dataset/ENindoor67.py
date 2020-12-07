#################################################################################
#                                                                               #
# MIT License                                                                   #
# Copyright (c) Wilson Lam 2020                                                 #
#                                                                               #
# Permission is hereby granted, free of charge, to any person obtaining a copy  #
# of this software and associated documentation files (the "Software"), to deal #
# in the Software without restriction, including without limitation the rights  #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     #
# copies of the Software, and to permit persons to whom the Software is         #
# furnished to do so, subject to the following conditions:                      #
#                                                                               #
# The above copyright notice and this permission notice shall be included in all#
# copies or substantial portions of the Software.                               #
#                                                                               #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE #
# SOFTWARE.                                                                     #
#-------------------------------------------------------------------------------#
# ENindoor67.py contains the main dataset classes to work with MIT Indoor 67    #
# To use the classes: `from ENindoor67 import [class name]                      #
#################################################################################

import os
import pandas as pd
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import PIL.ImageEnhance as ie
import PIL.Image as im
import glob
import random
import cv2
import warnings
import time
import pickle
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler, ConcatDataset
from PIL import Image
from math import floor
from sklearn.model_selection import train_test_split
from dataset.ENindoor67_preprocessor import ENindoor67Preprocessor
from botocore.response import StreamingBody

# Random seeds
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)

"""
TRANSFORMATION CLASSES
source: https://github.com/mratsim/Amazon-Forest-Computer-Vision/blob/master/src/p_data_augmentation.py
"""
class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

class RandomFlip(object):
    """Randomly flips the given PIL.Image with a probability of 0.25 horizontal,
                                                                0.25 vertical,
                                                                0.5 as is
    """
    
    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img.transpose(im.FLIP_LEFT_RIGHT),
            3: img.transpose(im.FLIP_TOP_BOTTOM)
        }
    
        return dispatcher[random.randint(0,3)] #randint is inclusive

class RandomRotate(object):
    """Randomly rotate the given PIL.Image with a probability of 1/6 90°,
                                                                 1/6 180°,
                                                                 1/6 270°,
                                                                 1/2 as is
    """
    
    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img,            
            3: img.transpose(im.ROTATE_90),
            4: img.transpose(im.ROTATE_180),
            5: img.transpose(im.ROTATE_270)
        }
    
        return dispatcher[random.randint(0,5)] #randint is inclusive
    
class PILColorBalance(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Color(img).enhance(alpha)

class PILContrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Contrast(img).enhance(alpha)


class PILBrightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Brightness(img).enhance(alpha)

class PILSharpness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Sharpness(img).enhance(alpha)
    

# Check ImageEnhancer effect: https://www.youtube.com/watch?v=_7iDTpTop04
# Not documented but all enhancements can go beyond 1.0 to 2
# Image must be RGB
# Use Pillow-SIMD because Pillow is too slow
class PowerPIL(RandomOrder):
    def __init__(self, rotate=True,
                       flip=True,
                       colorbalance=0.4,
                       contrast=0.4,
                       brightness=0.4,
                       sharpness=0.4):
        self.transforms = []
        if rotate:
            self.transforms.append(RandomRotate())
        if flip:
            self.transforms.append(RandomFlip())
        if brightness != 0:
            self.transforms.append(PILBrightness(brightness))
        if contrast != 0:
            self.transforms.append(PILContrast(contrast))
        if colorbalance != 0:
            self.transforms.append(PILColorBalance(colorbalance))
        if sharpness != 0:
            self.transforms.append(PILSharpness(sharpness))
            
"""
DATA EXPLORATION AND PREPARATION CLASSES
"""

class Composer(torchvision.transforms.Compose):
    """
    Define some constants for EfficientNet Resolution:
    https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#keras-implementation-of-efficientnet
    """
    RESOLUTIONS = {
        'efficientnet-b0' : 224,
        'efficientnet-b1' : 240,
        'efficientnet-b2' : 260,
        'efficientnet-b3' : 300,
        'efficientnet-b4' : 380,
        'efficientnet-b5' : 456,
        'efficientnet-b6' : 528,
        'efficientnet-b7' : 600
    }
    
    def __init__(self, mode='raw', efficientnet='efficientnet-b0', augment=False, transformer=None):
        """
        instantiate a Compose object customized to EfficientNet
        @param efficientnet (string) : the class name of EfficientNet
        @param augment (bool) : if True, Composer will compose a series of data augmentation techniques;
                                  otherwise, only resizing and normalization will be performed
        """
        
        self.mode = mode
        self.efficientnet = efficientnet
        self.augment = augment
        self.transformer = PowerPIL() if not transformer else transformer
        
        assert mode in ['raw', 'tensor'], "Composer mode can only be either `raw` or `tensor`."
            
        resolution = self.RESOLUTIONS[efficientnet.lower()]
        
        if augment:
            super().__init__([
                transforms.RandomResizedCrop(resolution),
                self.transformer,
            ])
        
        else:
            super().__init__([
                transforms.Resize((resolution, resolution)),
            ])
        
        if mode == 'tensor':
            self.transforms += [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean & SD
            ]
            
    @classmethod
    def resolutions(cls):
        """
        return class RESOLUTIONS
        """
        return cls.RESOLUTIONS
    
    def __str__(self):
        return "Composer(mode={}, efficientnet={}, augment={}, transformer={})".format(self.mode,
                                                                                       self.efficientnet,
                                                                                       self.augment,
                                                                                       self.transformer)

# Helper class for ENindoor67 Dataset            
class ImageDataFrame(pd.DataFrame):
    def __init__(self, root_dir):
        """
        Read in root directory and label categories and their corresponding numerical labels
        @param data_dir (string) : directory path to the data / .csv file containing S3 paths
        """
        if root_dir.endswith('.csv'):
            super().__init__(pd.read_csv(root_dir, delimiter=',', index_col=False))
            self.categories = {v: k 
                   for k, v in dict(pd.Series(self['Category'].unique())).items()}
        else:
            image_files = np.array(glob.glob(os.path.join(root_dir, '**', '*.*'), recursive=True))
            super().__init__(image_files, columns = ['Path'])
            self['Category'] = self.Path.apply(lambda df : df.split('/')[-2])
            self.categories = {v: k 
                               for k, v in dict(pd.Series(self['Category'].unique())).items()}
            self['Class'] = self.Category.apply(lambda df: self.categories.get(df))
        
        
    def describe(self):  # override parent method
        """
        return the count of each category
        """
        return pd.concat([self['Category'].describe()[:2],
                          self.groupby('Category').size()])
    
    def describe_categories(self):
        return self.groupby('Category').size().describe().astype('int32')
    

# Naive implementation of ENindoor67 dataset; facilitate exploration and preprocessing    
class ENindoor67(Dataset):
    def __init__(self,
                 root_dir='data/mit_indoor_67/raw/Images',
                 data_file=None,
                 s3_bucket=None,
                 mode='raw',
                 transform=True,
                 efficientnet='efficientnet-b0',
                 augment=False,
                 transformer=None):
        """
        Naive implementation -
        Custom Dataset for MIT 67 Indoor dataset and EfficientNet.
        Users can perform basic data augmenetation, visualization,
        train-val-test-split, upload.
        
        """
        
        assert mode in ['raw', 'tensor'], "Dataset can only be in `tensor` or `raw` mode."
        
        self.s3_bucket = s3_bucket  # for AWS
         
        if data_file: # instantiate dataset with S3 locations (path)
            if s3_bucket:
                self.dataframe = ImageDataFrame(data_file)
                self.root_dir = None
            else:
                raise Exception('S3 Bucket information required for initializing dataset with .csv file.')
        else:  # instantiate dataset with local directory
            self.dataframe = ImageDataFrame(root_dir)
            self.root_dir = root_dir
            
        self.mode = mode
        self.transform = transform
        self.efficientnet = efficientnet
        self.augment = augment
        self.transformer = transformer
        self.composer = Composer(mode=mode, efficientnet=efficientnet, augment=augment, transformer=transformer)
        self.train = []
        self.val = []
        self.test = []
        self.samplers = None

    
    def __len__(self):
        """
        return the length of the dataframe, i.e. dataset
        """
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Item getter by index.
        @param idx (int) : item index
        
        Return:
        Image Mode : dictionary containing the corresponding image, path, size, category, label +/- set
        Tensor Mode : tuple of image in tensor and class label
        """
        # unpack with respect to current dataframe columns
        try:
            path, category, label = self.dataframe.iloc[idx]
        except ValueError:
            path, category, label, subset = self.dataframe.iloc[idx]
        
        # retrieve from S3 bucket if available
        if self.s3_bucket:
            s3_location = path
            response = self.s3_bucket.Object(path).get()
            path = response['Body']
        
        image = Image.open(path).convert('RGB')
        size = image.size
        augment = self.augment  # store current augment state
        
        if self.transform:
            if idx not in self.train:  # if image is not in train set, switch off augmentation
                self.augment = False
                self._update_composer()
                
            image = self.composer(image)

            if self.mode == 'tensor':
                size = tuple(image.size())
            else:
                size = image.size
            self.augment = augment  # revert back to initial augment state
        
        if not isinstance(path, str):
            path = s3_location
        
        if self.mode == 'tensor':
            return (image, label)
        try:
            return {'Image': image, 'Path': path, 'Size': size, 'Category': category, 'Label': label, 'Set': subset}
        except UnboundLocalError:
            return {'Image': image, 'Path': path, 'Size': size, 'Category': category, 'Label': label}
    
    
    """
    Add switches to make dataset more flexible for visualization and dataloading & training
    """
    
    def _update_composer(self):
        """
        Reinstantiate a new composer class when switches parameters change
        """
        self.composer = Composer(mode=self.mode,
                                 efficientnet=self.efficientnet,
                                 augment=self.augment,
                                 transformer=self.transformer)
        
    def toraw(self):
        """
        Dataset change to raw mode
        """
        self.mode = 'raw'
        self._update_composer()
    
    def totensor(self):
        """
        Dataset change to tensor, normalized mode; returned as tensor array
        """
        self.mode = 'tensor'
        self._update_composer()
    
    def transformed(self):
        """
        Dataset is transformed when called
        """
        self.transform = True
    
    def augmented(self):
        """
        Dataset is augmented when called
        """
        self.augment = True
        self._update_composer()
    
    def original(self):
        """
        Dataset return as original when called
        @param augment (bool) : if True, data will be augmented
        """
        self.transform = False
        self.mode = 'raw'
        self._update_composer()
    
    def switchNet(self, efficientnet):
        """
        Dataset will be resized according to the required resolution of EfficientNet
        """
        self.efficientnet = efficientnet
        self._update_composer()
    
    def switchTransformer(self, transformer):
        """
        Dataset will be transformed by a new transformer
        """
        self.transformer = transformer
        self._update_composer()
    
    def distributions(self, groupby=["Set", "Category"]):
        """
        Show the count of categories.
        If already split, show count of `groupby`.
        Default: the count of each category in `train`, `val` & `test`
        Alternatively, can define groupby = `Set` to show count of each set
        @param groupby (str / list) : the column(s) to count; same as pd.DataFrame.groupby()
        
        return :
        Count of image in each category if data not yet split /
        Count of image as defined in `groupby`
        """
        if "Set" in self.dataframe:
            return self.dataframe.groupby(groupby).size()
        return self.dataframe.describe()[2:]
    
    def categories(self, tolist=False):
        """
        Return a dictionary of categories as key and their corresponding class label (int) as value
        @param tolist (bool) : if `True` return in list type; default `False`.
        """
        if tolist:
            return list(self.dataframe.categories.keys())
        return self.dataframe.categories
    
    def set_summary(self):
        """
        Return the counts of train, val, and test set
        """
        return self.dataframe.groupby('Set').size()
    
    def describe_categories(self):
        """
        Return the descriptive statistics of image samples of all categories
        """
        return self.dataframe.describe_categories()
    
    def show_sample(self, subset='default', random_state=1):
        """
        Show 9 random images without normalization
        """
#         if self.tensor:  # sanity check
#             raise Exception('Cannot show samples when dataset is in tensor mode.')
        
        assert subset in ['default', 'train', 'val', 'test'], "Allowed `subset` values: `default`, `train`, `val`, `test`"
        
        if (subset in ['train', 'val', 'test'] and 
            not any([len(dataset) > 0 for dataset in [self.train, self.val, self.test]])):
            raise Exception("You must split the data before you call samples from `train`, `val` or `test`.")
            
        mode = self.mode  # store current mode
        
        if mode == "tensor":  # warn user for switching to raw mode temporarily
            warnings.warn("Dataset current mode: `tensor`\nSwitching to `raw` mode temporarily.")
        
        if subset == 'train':
            sample_idx = self.train
        
        elif subset == 'val':
            sample_idx = self.val
        
        elif subset == 'test':
            sample_idx = self.test
            
        else:
            sample_idx = np.arange(len(self.dataframe))
            
        np.random.seed(random_state)
        np.random.shuffle(sample_idx)
        sample_idx = list(sample_idx)[:9]  # get random samples
        self.toraw()  # make sure data is not in tensor mode
        
        # plotting
        fig = plt.figure(figsize=(20,20))
        size = self.composer.RESOLUTIONS[self.efficientnet]
        transform_subtitle = f"{self.efficientnet} ({size} x {size})" if self.transform else 'ORIGINAL'
        augment_subtitle = 'AUGMENTED' if (subset == 'train' and self.augment and self.transform) else ''
        fig.suptitle(f"INDOOR 67 SAMPLE DATASET\n{transform_subtitle}\n{augment_subtitle}\nSUBSET:{subset}", fontsize=18)
        
        for i in range(9):
            sample_img = self.__getitem__(sample_idx[i])
            ax = plt.subplot(3, 3, i+1)
            plt.title(f"Sample #{i+1} - Category: {sample_img['Category']}\nfilename:{sample_img['Path'].split('/')[-1]}")
            plt.imshow(np.array(sample_img['Image']))
        plt.show()
        
        # revert to initial mode
        self.mode = mode
        self._update_composer()
                      
    def train_val_test_split(self, train_size=0.7, val_size=0.15, shuffle=True, random_state=1, stratify=False):
        """
        split with class proportions
        @param train_size (float) : proportion of train set
        @param val_size (float) : proportion of val set
        @param shuffle (bool) : if True, the indices will be shuffled before splitting
        @random_state (int) : random seed int
        
        Return:
        train_idx (list) : indices of train set
        val_idx (list) : indices of val set
        test_idx (list) : indices of test set
        """
        # Sanity Check
        if sum([train_size, val_size]) > 1.0:
            raise ValueError("Sum of split size cannot exceed 1.0")
        if not all(isinstance(val, float) for val in [train_size, val_size]):
            raise TypeError("Split size must be of float type.")
        
        if stratify:
            train_idx, test_idx, train_y, test_y = train_test_split(self.dataframe.index,
                                                                    self.dataframe.Class,
                                                                    train_size=train_size,
                                                                    shuffle=shuffle,
                                                                    random_state=random_state,
                                                                    stratify=self.dataframe.Class)
            train_idx, val_idx, train_y, val_y = train_test_split(train_idx,
                                                                  train_y,
                                                                  train_size= (1 - val_size),
                                                                  shuffle=shuffle,
                                                                  random_state=random_state,
                                                                  stratify=train_y)
                      
        else:
            train_idx, test_idx = train_test_split(self.dataframe.index,
                                                   train_size=train_size,
                                                   shuffle=shuffle,
                                                   random_state=random_state)
            train_idx, val_idx = train_test_split(train_idx,
                                                  train_size= (1 - val_size),
                                                  shuffle=shuffle,
                                                  random_state=random_state)
        
        train_idx = list(train_idx)
        val_idx = list(val_idx)
        test_idx = list(test_idx)
                      
        # Store indices in the dataset object
        self.train = train_idx
        self.val = val_idx
        self.test = test_idx
        self._assign_split_index()
                      
        return train_idx, val_idx, test_idx
    
    def _assign_split_index(self):
        """
        assign split sets index to self.dataframe
        """
        subsets = np.array(["" for _ in range(self.__len__())], dtype="<U8")
        subsets[self.train] = 'train'
        subsets[self.val] = 'val'
        subsets[self.test] = 'test'
        self.dataframe['Set'] = subsets
    
    def plot_distributions(self):
        """
        plot the class distribution in the subsets. Helps indicate class imbalance.
        """
        try:
            subsets = ['train', 'val', 'test']
            
            for subset in subsets: 
                self.distributions()[subset].sort_values(ascending=False).plot.bar(figsize=(20,5), rot=85, title="INDOOR 67 DATASET\nClass distribution in {} set".format(subset))
                plt.show()

        except KeyError:
            self.distributions().sort_values(ascending=False).plot.bar(figsize=(20,5), rot=85, title="INDOOR 67 DATASET\nClass distribution")

    def _assign_weights(self):
        """
        assign weights to train set samples
        """
        class_weights = 1./torch.tensor(self.distributions().tolist(), dtype=torch.float)
        sample_weights = class_weights[self.dataframe['Class'].iloc[self.train].tolist()]
        self.weights = torch.tensor(sample_weights)
    
    def sample_weights(self):
        """
        return the weights for train set samples
        """
        try:
            return self.weights
        except:
            self._assign_weights()
            return self.weights
    
    def sample(self, method='weighted', train_size=0.7, val_size=0.15, shuffle=True, random_state=1, stratify=False):
        """
        sampling from dataset.
        return a sampler object according to `method`.
        @param method (string) : Sampling method - `weighted` or `subsetrandom`
                                 if method falls out of these 2 methods, `weighted` will be used by default
        @param train_size (float) : proportion of train set
        @param val_size (float) : proportion of val set
        @param shuffle (bool) : if True, the indices will be shuffled before splitting
        @random_state (int) : random seed int 
        
        return:
        `weighted` : WeightedRandomSampler(train), SubsetRandomSampler(val), SubsetRandomSampler(test)
        `subsetrandom` : SubsetRandomSampler(train), SubsetRandomSampler(val), SubsetRandomSampler(test)
        """
        method = method.lower()
        
        # check if the set is split or not, else split by the input parameters
        if all([self.train, self.val, self.test]):
            train_idx, val_idx, test_idx = self.train, self.val, self.test
        
        else:
            train_idx, val_idx, test_idx = self.train_val_test_split(train_size=train_size,
                                                         val_size=val_size,
                                                         shuffle=shuffle,
                                                         random_state=random_state,
                                                         stratify=stratify)
        # check for valid methods              
        if method not in ['subsetrandom', 'weighted']:
            warnings.warn(f"'{method}' sampling method not defined in dataset. Return weighted random sampler.")
            method = 'weighted'
        
        # if `weighted`, train set will be sampled by assigned weights
        if method == 'weighted':
            weights = self.sample_weights()
            self.samplers = WeightedRandomSampler(weights=weights,
                                                  num_samples=len(weights),
                                                  replacement=True), SubsetRandomSampler(val_idx), SubsetRandomSampler(test_idx)
        
        # if `subsetrandom`, train set will be sampled by random
        if method == 'subsetrandom':
            self.samplers = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx), SubsetRandomSampler(test_idx)
        
        return self.samplers
    
    def preprocess(self, target_dir, augment=False, subset='train', parts=4, **kwargs):
        """
        Preprocess the data according to `subset` and pickled into N `parts`.
        @param target_dir (str) : target location to save the pickles
        @param augment (bool) : whether or not to augement the data (for train set)
        @param subset (str) : either `train`, `val`, or `test`
        @param parts (int) : number of pickles to save the dataset
        
        Save N `parts` of pickled dataset to `target_dir`
        """
        # make sure the dataset has been split
        assert all([self.train, self.val, self.test]), "You must split the dataset before preprocessing it"
        
        # create paths if not exists
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        sub_dir = os.path.join(target_dir, subset)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        target_dir = os.path.join(sub_dir, self.efficientnet.replace('-', '_'))
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
                      
        # call preprocessor to pickle; preprocess extra augemented set if it's `train` and `augment`
        if subset == 'train' and augment:
            ENindoor67Preprocessor().dump(self, target_dir=target_dir, augment=augment, subset=subset, parts=parts, **kwargs)
        ENindoor67Preprocessor().dump(self, target_dir=target_dir, augment=False, subset=subset, parts=parts, **kwargs)

                      
# Optimized Dataset for loading from s3 and training
class ENindoor67Preprocessed(Dataset):
    """
    Preprocessed dataset optimizied for GPU performance.
    Modified from: 
    https://github.com/aws-samples/sagemaker-gpu-performance-io-deeplearning/tree/master/src
    """
    def __init__(self, preprocessed_pickle, mode='tensor'):
        """
        instantiates an ENindoor67Preprocessed dataset.
        @params preprocessed_pickle (str) : preprocessed pickle file name; ends with '.pkl'
        @param mode (str) : mode to instanitate, can be switched later. accepts: `raw` or `tensor`
        """
        
        # Value checking
        assert mode in ['raw', 'tensor'], "dataset can only be in `tensor` or `raw` mode."
        assert preprocessed_pickle.endswith('.pkl'), "Preprocessed dataset only accepts pickle format."
                      
        self.mode = mode
        self.images, self.labels, self.categories = ENindoor67Preprocessor().load(preprocessed_pickle)        
        self.transformer = torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean & SD
            ])
                      
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
                      
        image, label, category = self.images[idx], self.labels[idx], self.categories[idx]
        
        if self.mode == 'tensor':       
            image = self.transformer(image)
            return image, label
        
        return image, category
                      
    def toraw(self):
        self.mode = 'raw'
    
    def totensor(self):
        self.mode = 'tensor'
    
    def show_sample(self, random_state=1):
        """
        Show 9 randome samples. Same as naive implementation.
        """
        np.random.seed(random_state)                   
        mode = self.mode  # store current mode
        sample_idx = np.arange(self.__len__())
        np.random.shuffle(sample_idx)
        sample_idx = list(sample_idx)[:9]  # get random samples
        self.toraw()  # make sure data is not in tensor mode
                      
        # plotting
        fig = plt.figure(figsize=(20,20))
        fig.suptitle("INDOOR 67 SAMPLE DATASET", fontsize=16)
        for i in range(9):
            image, category = self.__getitem__(sample_idx[i])
            ax = plt.subplot(3, 3, i+1)
            plt.title(f"Sample #{i+1} - Category: {category}")
            plt.imshow(np.array(image))
        plt.show()
        
        # revert to initial mode
        self.mode = mode
   

# ConcatDataset for concatenating ENindoor67 `train` original and augmented sets
class ENindoor67Datasets(ConcatDataset):
      """
      Used together with ENindoor67Preprocessed to prepare data for training.
      """
      def __init__(self, datasets):
          super().__init__(datasets)
                      
      def _get_weights(self):
                      
          labels = []
          for dataset in self.datasets:
              for i in range(len(dataset)):
                  tensor, label = dataset[i]
                  labels.append(label)   
          labels = torch.tensor(labels)
          
          label_counts = torch.tensor([
              (labels == label).sum() 
              for label in torch.unique(labels, sorted=True)])
          
          weights = 1./ label_counts.float()
           
          return torch.tensor([weights[label] for label in labels])
       
      def get_sampler(self, method='weighted', random_state=1):
           
          assert method in ['weighted', 'subsetrandom'], "Sampling methods can only be: `weighted` or `subsetrandom`"
          
          torch.manual_seed(random_state)
          torch.cuda.manual_seed(random_state)
                      
          weights = self._get_weights()
          
          if method == 'weighted':
              return WeightedRandomSampler(weights=weights,
                                           num_samples=len(weights),
                                           replacement=False)
          return SubsetRandomSampler(np.arange(self.__len__()))

                      
class ENindoor67StreamingBody(ENindoor67Preprocessed):
    
    def __init__(self, streamingbody, mode='tensor'):
        assert mode in ['raw', 'tensor'], "dataset can only be in `tensor` or `raw` mode."
        assert isinstance(streamingbody, StreamingBody), "Only accepts StreamingBoy; got {}.".format(type(streamingbody))
                      
        self.mode = mode
        data = pickle.loads(streamingbody.read())     
        self.images, self.labels, self.categories = data["images"], data["labels"], data["categories"]
        self.transformer = torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean & SD
            ])