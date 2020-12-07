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
# utils.s3 contains helper functions that work with files on S3 / SM instances. #
# To use the module: `import utils.s3` or `from utils.s3 import [function name]`#
#################################################################################

import os
import glob
import sys
import subprocess

"""
HELPER FUNCTIONS TO WORK WITH FILES AND LIBRARIES ON S3 / INSTANCES
"""

def s3_upload(root_dir,
              session,
              bucket,
              prefix,
              preprocessed_key=None,
              verbose=True):
    """
    Upload split image dataset to AWS S3.
    @param root_dir (str) : local root directory to image files
    @param session (str) : AWS Session
    @param bucket (str) : AWS S3 bucket name
    @param prefix (str) : AWS S3 folder name
    @param preprocessed_key (str) : Subdirectory of preprocessed
                                    files; if `None`, image files
                                    will be uploaded
    @param verbose (bool) : if `True`, print uploading messages
    
    return: uploaded path
    """
    
    # Upload image files in `root_dir` to S3
    if not preprocessed_key:
        folders = glob.glob(os.path.join(root_dir, '*'))
        for folder in folders:
            uploaded = session.upload_data(folder,
                                           bucket=bucket,
                                           key_prefix = os.path.join(prefix,
                                                                     folder.split('/')[-1]))
            if verbose:
                print(f"Uploaded: {uploaded}")
                
    # Upload Preprocessed files to S3
    else:
        uploaded = session.upload_data(root_dir, bucket=bucket,
                                       key_prefix = os.path.join(prefix,
                                                                 preprocessed_key))
        if verbose:
            print(f"Uploaded {root_dir} to {uploaded}")
    
    return uploaded


# def install(package, upgrade=False):
#     """
#     Install of libraries - patching `requirements.txt`.
#     @param package (str) : name of package
#     @upgrade (bool) : whether to `--upgrade` to latest version
#     """
    
#     # OS command
#     command_line = [sys.executable, '-m', 'pip', 'install']
    
#     # Upgrade to latest version
#     if upgrade:
#         command_line += ['--upgrade']
    
#     # Execute command
#     command_line += [package]
#     status = subprocess.check_call(command_line)
    
#     # Notify if sucess
#     if status == 0:
#         print("Upgrade/Install package: {}".format(package))

        
