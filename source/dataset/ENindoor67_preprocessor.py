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
# ENindoor67_preprocessor.py contains helper class for preprocessing images     #
# To use the classes: `from ENindoor67_preprocessor import [class name]         #
#################################################################################

"""
THIS CODE IS MODIFIED FROM THE FOLLOWING AWS TUTORIAL:
https://github.com/aws-samples/sagemaker-gpu-performance-io-deeplearning/blob/master/src/datasets/caltech_image_preprocessor.py

Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import os
import pickle
from glob import glob
from PIL import Image

class ENindoor67Preprocessor:
    
    
    def dump(self, dataset, target_dir, augment, subset='train', parts=4, **kwargs): 

#         if not dataset.root_dir:
#             raise Exception("PathPointerError: Image Path should point to local directory, not S3 locations.")
        
        if not any([dataset.train, dataset.val, dataset.test]):
            raise Exception("Splitting required before preprocessing.")
            
        dataset.transformed()
        dataset.toraw()
        augmented = dataset.augment
        
        if subset == 'train' and augment:
            dataset.augmented()


        print(f"Preprocessing {subset} set data...")
        print(f"Composer configuration: {dataset.composer}")
        self._preprocess(dataset, subset, target_dir, augment, parts, **kwargs)
        dataset.augment = augmented
        dataset._update_composer()
        
    def _preprocess(self, dataset, target_set, target_dir, augment, parts, **kwargs):
        
        img_indices = dataset.__dict__[target_set]
        result = {}
        images = []
        labels = []
        categories = []
        result_len = 0
        
        files_per_pickle = len(img_indices) // parts
        pickle_part = 1
        
        if ((target_set == 'train') and augment and 'rounds' in kwargs):
            print("train will be augmented for {} rounds.".format(kwargs['rounds']))

        for i, idx in enumerate(img_indices):
            path, category, label, subset = dataset.dataframe.iloc[idx]
            response = dataset.s3_bucket.Object(path).get()
            path = response['Body']
            image = Image.open(path).convert('RGB')
            
            if ((target_set == 'train') and augment and ('rounds' in kwargs)):
                for j in range(kwargs['rounds']):
                    images.append(dataset.composer(image))
                    labels.append(label)
                    categories.append(category)
            else:
                images.append(dataset.composer(image))
                labels.append(label)
                categories.append(category)
                
            result = {
                      "images": images,
                      "labels" : labels,
                      "categories" : categories,
                     }

            if (i+1) % 200 == 0:
                print("Processed {} / {} image.".format((i+1), len(img_indices)))

            if (not target_set == 'test') and (i+1) % files_per_pickle == 0:
                self._save_set(target_dir, target_set, result, pickle_part, augment)
                images = []
                labels = []
                categories = []
                result = {}
                pickle_part += 1
                    
                    
        if result:
            self._save_set(target_dir, target_set, result, pickle_part, augment)
        print(f"Pickled {pickle_part} parts of {target_set} data to {target_dir}.")

            
    
    def _save_set(self, target_dir, target_set, result_obj, pickle_part, augment):
        
        if target_set == 'train':
            augmented = 'augmented' if augment else 'original'
            
            pickle_file = os.path.join(target_dir, "indoor67_{}_{}_{:02d}.pkl".format(augmented, target_set, pickle_part))
        
        
        elif target_set == 'test':
            
            pickle_file = os.path.join(target_dir, "indoor67_{}.pkl".format(target_set))
        
        else:
            pickle_file = os.path.join(target_dir, "indoor67_{}_{:02d}.pkl".format(target_set, pickle_part))
        
        with open(pickle_file, "wb") as f:
            pickle.dump(result_obj, f)
        
        print(f"Pickled part {str(pickle_part)} of {target_set} set data to {pickle_file}.")
        
    def load(self, pickle_obj):
        
        if isinstance(pickle_obj, str):
            
            with open(pickle_obj, "rb") as f:
                obj = pickle.load(f)
                
        elif isinstance(pickle_obj, bytes):
            obj = pickle.loads(pickle_obj)
        
        else:
            
            raise TypeError("Pickle can only load file in `str` or object in `byte`")

        return obj["images"], obj["labels"], obj["categories"]

            
        
        