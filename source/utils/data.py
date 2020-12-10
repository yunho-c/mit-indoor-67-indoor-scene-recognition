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
# utils.data contains helper functions workings with data in preprocessing      #
# To use the module: `import utils.data` or `from utils.data import [fn name]   #
#################################################################################


import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from dataset.ENindoor67 import Composer,\
ENindoor67, ENindoor67Preprocessed,\
ENindoor67StreamingBody, ENindoor67Datasets


def pickle_sets(target_dir,
                data_file,
                bucket,
                sets=['train', 'val', 'test'],
                efficientnet='efficientnet-b0',
                augment=False,
                parts=4,
                split_parameters=dict(
                    train_size=0.8,
                    val_size=0.2,
                    shuffle=True,
                    random_state=1
                ),
                **kwargs):
    
    """
    Preprocess and pickle designated `sets` for a particular
    `efficientnet` resolution, and save to `target_dir`.
    DataFrame of the dataset is pickled and saved to `target_dir`.
    
    @params target_dir (str) : local directory where the pickles
                               will be saved to
    @params data_file (str) : .csv file storing s3 locations of images
    @params bucket (S3 Bucket) : s3 Bucket object to retrieve files
                                 from S3
    @params sets (list) : list of subsets to preprocess and pickle;
                          valid values: `train`, `val`, `test`
    @params efficientnet (str) : version of EfficientNet
    @params augment (bool) : applies only to `train` set;
                             if True, additional augmented `train` data
                             will be created. Default: False.
    @params parts (int) : partition size
    @params split_parameters (dict) : dictionary of params to split the data:
                                     'train_size' (float) : size of train set
                                     'val_size" (float) : size of val set
                                     'shuffle' (bool) : shuffle if `True`
                                     'random_state' (int) : random seed
    
    return:
    pd.DataFrame with split set index in the `Set` column
    """
    
    # Value checking
    valid = {'train', 'val', 'test'}
    assert not set(sets).difference(valid), \
    f"sets can only be `train`, `val`, `test`; got {invalid_set}"
    
    available_nets = list(Composer.resolutions().keys())
    assert efficientnet in available_nets, \
    f"invalid efficientnet value {efficientnet}"
    
    # Instantiates dataset
    dataset = ENindoor67(data_file=data_file,
                         s3_bucket=bucket)    
    
    print("Preprocessing {} set(s) for {}..."
          .format(', '.join(sets), efficientnet))
    
    # Make sure the dataset is in the target EfficicentNet version
    dataset.switchNet(efficientnet)  
    # Split the data
    train, val, test = dataset.train_val_test_split(**split_parameters)  
    
    # Preprocess the sets
    for subset in sets:
        print("subset {} length: {}"
              .format(subset, len(dataset.__dict__[subset])))
        # Preprocess the data; Note: resizing to speed up training later
        dataset.preprocess(target_dir=target_dir,
                           augment=augment,
                           subset=subset,
                           parts=parts, **kwargs)
    
    # Pickle file
    dataframe_pickle = os.path.join(target_dir,
                                    "mitindoor67_split_data.pkl"
                                    .format(efficientnet))
    
    dataset.dataframe.to_pickle(dataframe_pickle)
    
    return dataset.dataframe
 
    
def compile_metadata(metadata_dir,
                     subset,
                     s3_dir,
                     efficientnet='efficientnet-b0'):
    
    """
    compile caches for datasets.
    
    @params metadata_dir (str) : local path of where the metadata
                                 / cache will be stored
    @params subset subset (str) : `train`, ` val` or `test` set of
                                  dataset
    @params s3_dir (str) : s3 paths where the pickled dataset is stored
    @params efficientnet (str) : version of efficienet net
    
    """
    
    # Create directory if not exist
    if not os.path.exists(metadata_dir):
        os.mkdir(metadata_dir)
    
    # File path
    efficientnet = efficientnet.replace("-", "_")
    pickle_file = os.path.join(metadata_dir, "{}.pkl".format(efficientnet))
    
    if os.path.exists(pickle_file):  # open cache file if already exists
        metadata = pickle.load(open(pickle_file, "rb"))
    else:
        metadata = {}
        
    # Write / update metadata
    metadata[subset] = s3_dir
    
    print(f"""Pickle metadata:
    {metadata}""")
    
    # Write to file
    with open(pickle_file, "wb") as f:
        pickle.dump(metadata, f)


def show_train_images(filesystem,
                      data_dir,
                      bucket,
                      rounds=2,
                      n_rows=3,
                      category=None,
                      original_tag='original',
                      augmented_tag='augmented'):
    
    # List files in S3 directory
    files = filesystem.ls(data_dir)
    
    # Locate original & augment files
    
    org_start = None
    aug_start = None
    
    # Locate index
    for i, file in enumerate(files):
        if (original_tag in file and
            not org_start):
            org_start = i
        
        if (augmented_tag in file and
            not aug_start):
            aug_start = i
        
        if org_start and aug_start:
            break
    
    # Zip original files and augment files
    # Put original data first; skip last file
    if org_start > aug_start:
        trim = org_start
        data = list(zip(files[trim:-1], files[:(trim-1)]))
    else:
        trim = aug_start
        data = list(zip(files[:(trim-1)], files[trim:-1]))
    
    # Shuffle the list and get the pair of data files
    random.shuffle(data)
    org_data, aug_data = data.pop()
    
    
    # Get StreamingBody from S3
    org_data = org_data.split(bucket.name+'/')[-1]
    aug_data = aug_data.split(bucket.name+'/')[-1]
    org_data = bucket.Object(org_data).get()['Body']
    aug_data = bucket.Object(aug_data).get()['Body']
    
    # Visualize the data
    _show_train_images(org_data,
                       aug_data,
                       rounds,
                       n_rows,
                       category)
    

def _show_train_images(original_data,
                   augment_data,
                   rounds=2,
                   n_rows=3,
                   category=None):
    
    """
    Helper function to visualize original and augmented train images.
    Note: original and augment data must be paired in advance.
    @params  original_data (StreamingBody)    : boto3 score streamingbody of
                                                original train image
    @params  augment_data (StreamingBody)     : boto 3 score streamingbody of
                                                augmented train image
    @params  rounds       (int)               : number of extra augmented set
                                                default: 2
    @params  n_rows       (int)               : number of rows to show
    @params  category     (str)               : MIT Indoor 67 category name
    
    Show random `n_rows` row of original + `rounds` extra set of augmented image
    if `category` is specified, show specific images of the `category`
    """
    
    # Prepare dataset and set to raw mode
    augment_data = ENindoor67StreamingBody(augment_data)
    augment_data.toraw()
    original_data = ENindoor67StreamingBody(original_data)
    original_data.toraw()
    
    # Concat dataset
    dset = [original_data, augment_data]
    dset = ENindoor67Datasets(dset)
    
    # Initiailize variables
    start_aug = len(original_data)  # where augmented image starts
    element = 1  # index of plotting element in subplots
    
    # Set plotting layout
    size = n_rows * rounds * 2
    fig = plt.figure(figsize=(size, size))
    
    if category:
        # Type checking
        assert isinstance(category, str), "Category must be a str."
        found = 0  # tracking no. of match category 
        
        fig.suptitle('MIT INDOOR 67 TRAIN SET SAMPLES: {}'
                          .format(category.upper()), fontsize=16)
        
        # Iterate through the dataset
        for i in range(start_aug):
            
            # Plot and stop if no. of images `found` == `n_row`
            if found >= n_rows:
                plt.show()
                break
            
            aug_idx = start_aug  # Initialize index to augmented img
            org_img, org_category = dset[i]  # retrieve image and category label
            
            # If label matches, plot the image on axis
            if org_category == category:
                
                # Ploting
                ax = plt.subplot(n_rows, rounds+1, element) 
                ax.set_title('original: {}'.format(org_category))
                plt.imshow(org_img)
                
                # Increment / update variables 
                element += 1
                aug_idx += (i * rounds)
                
                # Enumerate augmented images
                for j in range(rounds):
                    aug_idx += j  # Increment index
                    
                    # Plotting
                    aug_img, aug_category = dset[aug_idx]
                    ax = plt.subplot(n_rows, rounds+1, element)
                    ax.set_title('augmented #{}: {}'.format(j+1, aug_category))
                    plt.imshow(aug_img)
                    
                    # Increment axis index
                    element += 1
                
                # Increment no. of images `found`
                found += 1
        
    else:
        
        # Initidialize variables
        idx = random.randint(0, start_aug - (rounds + 1))
        aug_idx = start_aug + (idx * rounds)
        
        # Iterate through specific range
        for i in range(idx, idx+n_rows):

            # Plotting original image
            org_img = dset[i]
            ax = plt.subplot(n_rows, rounds+1, element)
            ax.set_title('original: {}'.format(org_img[1]))
            plt.imshow(org_img[0])
            element += 1  # Increment axis index
            
            # Plotting augmented image
            for j in range(rounds):
                aug_idx += j
                aug_img = dset[aug_idx]
                ax = plt.subplot(n_rows, rounds+1, element)
                ax.set_title('augmented #{}: {}'.format(j+1, aug_img[1]))
                plt.imshow(aug_img[0])
                element += 1  # Increment axis index

            aug_idx += 1  # Increement augmented image index
        
        fig.suptitle('MIT INDOOR 67 TRAIN SET SAMPLES', fontsize=16)
        plt.show()