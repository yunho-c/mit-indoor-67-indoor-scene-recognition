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
# utils.helpers contains helper functions working with data in training.        #
# To use the module: `import utils.helpers` or `from utils.helpers import [fn]` #
#################################################################################

import os
import torch
from glob import glob
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from adabound import AdaBound
from dataset.ENindoor67 import ENindoor67Preprocessed, ENindoor67Datasets


def get_dataloader(data_dir,
                   config,
                   batch_size,
                   method='weighted',
                   random_state=1,
                   subset='train',
                   workers=2):
    
    """
    Get data loader for train set
    @param data_dir (str) : s3 path containing pickled preprocessed data
    @param config (str) : either ``, `augmented` or `original`
                          if ``(empty), all preprocessed `train` data will
                          be used for training; otherwise, only the specified
                          set will be used
    @batch_size (int) : size of each batch
    @method (str) : sampling method, either `weighted` or ` subsetrandom`
    @random_state (int) : seed for random generation
    @subset (str) : either `train` or `val`
    @workers (int) : number of workers to train the model
    
    return: 
    DataLoader object of concatenated train set
    """
    
    print("Getting {} set data loader...".format(subset))
    
    # If `config` is specific, select the specific subset accordingly
    if subset == 'train' and config:
        datasets = [ENindoor67Preprocessed(pickle)
                    for pickle in glob(os.path.join(data_dir, '*'))
                    if config in pickle]
        print("Loading {} train dataset.".format(config))
    
    # Otherwise, the entrie subset is selected for training
    else:
        datasets = [ENindoor67Preprocessed(pickle)
                    for pickle in glob(os.path.join(data_dir, '*'))]
        print("Loading {} dataset.".format(subset))
        
    
    # Concat datasets
    datasets = ENindoor67Datasets(datasets)
    print("Dataset size: {}".format(len(datasets)))
    
    # Get Sampler of the data  
    if subset == 'train':
        sampler = datasets.get_sampler(method=method,
                                       random_state=random_state)

        # $eturn data loader
        return DataLoader(datasets, batch_size=batch_size,
                          sampler=sampler, num_workers=workers)

    
    else:  # Non-train set will always use SubsetRandomSampler
 
        return DataLoader(datasets, batch_size=batch_size,
                          shuffle=True, num_workers=workers)


def initialize_summary():
    
    """
    Initialize a dictionary object for logging
    This function takes no arguments.
    """
    
    summary = {
                "loss": {
                            "train" : [],
                            "val" : []
                },
                "accuracy" : {
                            "train" : [],
                            "val" : []
                },
                "training_time" : [],
                "best_acc" : 0.0,
                "min_val_loss" : 0.0,
                "time_elapsed" : '',
                "hyperparameters" : {},
                "best_acc_model_wts" : None,
                "min_val_loss_model_wts" : None
              }
    
    return summary


def get_hyperparameters(args):
    
    """
    Store all arguments in `main.py`, except `SM_CHANNEL`
    and `model`, in a dictionary
    
    return:
    Dictionary of selected arguments passed to `main.py`
    """
    
    return {param : val for param, val in args.__dict__.items()
            if (not param.endswith('_dir')) and param != 'model'}


