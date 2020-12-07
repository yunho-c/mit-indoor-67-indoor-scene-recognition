import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import copy
import subprocess
import random
from glob import glob
from PIL import Image
from six import BytesIO
from torch.utils.data import DataLoader
from dataset.ENindoor67 import Composer, ENindoor67Preprocessed
from utils.helpers import get_dataloader, get_hyperparameters
from utils.trainer import Trainer
from model import ModelMaker
from adabound import AdaBound

# Random seeds
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model 
    model, _ = ModelMaker().make_model(**model_info)
    
    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model


def load_weights():
    
    print(os.listdir('/opt/ml/input/data/base/'))
    print("Unzipping base model...")
    subprocess.check_call(['tar','-xvf', '/opt/ml/input/data/base/model.tar.gz'])
    subprocess.check_call(['rm', '-rf', '/opt/ml/input/data/base/model.tar.gz'])
    
    model_path = os.path.join('.', 'model.pth')
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f)
    
    model_info = {}
    model_info_path = os.path.join('.', 'model_info.pth')
    with open(model_info_path, 'rb') as summaryf:
        model_info = torch.load(summaryf)
    
    print("Loaded base model: {}".format(model_info))
    
    return state_dict
    
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-metadata-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val-metadata-dir', type=str, default=os.environ['SM_CHANNEL_VAL'])
    try:
        parser.add_argument('--base-model-dir', type=str, default=os.environ['SM_CHANNEL_BASE'])
    except KeyError:
        print("No base model weights found.")
    
    # Training Parameters
    parser.add_argument('--config', type=str, default='', choices=['', 'augmented', 'original'],
                        metavar='CONF', help='dataset configuration (default: None)')
    parser.add_argument('--sampling', type=str, default='weighted', choices=['weighted', 'subsetrandom'],
                        metavar='SM', help='sampling method for train set (default : `weighted`)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--final-lr', type=float, default=0.1, metavar='FLR',
                        help='final learning rate for AdaBound (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--optimizer', type=str, default='Adam', metavar='OPTIM',
                        help='optimizer used in training (default: `Adam`)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='MTM',
                        help='momentum factor used in SGD (default: 0.9)')
    parser.add_argument('--model', type=str, default='ResNeXt101', metavar='M',
                        help='pretrained model for training (default: `ResNeXt101`)')
    parser.add_argument('--workers', type=int, default=1, metavar='W',
                        help='number of workers (default: 1)')
    parser.add_argument('--patience', type=int, default=3, metavar='P',
                        help='number of epochs for no improvments in val loss for early stopping. (default: 3)')
    parser.add_argument('--num-classes', type=int, default=67, metavar='NC',
                        help='number of predicted classes (default: 67)')
    parser.add_argument('--blocks-unfrozen', type=int, default=0, metavar='LU',
                        help='number of layers to unfreeze in EfficientNet (default:2)')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='DO',
                        help='dropout rate for fine-tuning (default: 0.2)')
    parser.add_argument('--depth', type=int, default=1, metavar='DP',
                        help='depth of tuned model (default: 1)')
    parser.add_argument('--weight-decay', type=float, default=0.01, metavar='WD',
                        help='weight decay rate / L2 regularization to Adam / AdamW (default: 0.01)')
    
    
    
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load the training and validation data.
    train_loader = get_dataloader(args.train_metadata_dir,
                                  args.config,
                                  batch_size=args.batch_size,
                                  method=args.sampling,
                                  random_state=args.seed,
                                  subset='train',
                                  workers=args.workers)
    val_loader = get_dataloader(args.val_metadata_dir,
                                args.config,
                                batch_size=args.batch_size,
                                subset='val',
                                random_state=args.seed,
                                workers=args.workers)
    
    #instantiate the model
    model, model_info = ModelMaker().make_model(model=args.model,
                                                num_classes=args.num_classes,
                                                blocks_unfrozen=args.blocks_unfrozen,
                                                dropout=args.dropout,
                                                depth=args.depth)
    print("Model - {} loaded.".format(model))
    
    if 'SM_CHANNEL_BASE' in os.environ:
        model.load_state_dict(load_weights())
        print("Loaded base model weights.")
    
    
    # optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               amsgrad=False,
                               weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'amsgrad':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               amsgrad=True,
                               weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=args.lr,
                             final_lr=args.final_lr)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr,
                                amsgrad=False,
                                weight_decay=args.weight_decay)
        
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer()
    model, summary = trainer.run(model=model,
                                 train_loader=train_loader,
                                 val_loader=val_loader,
                                 epochs=args.epochs,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 device=device,
                                 patience=args.patience)
    
    summary['hyperparameters'] = get_hyperparameters(args)
    
    
    # Save the training summary 
    summary_path = os.path.join(args.model_dir, 'training_summary.pth')
    with open(summary_path, 'wb') as summary_file:
        torch.save(summary, summary_file)
    
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as model_info_file:
        torch.save(model_info, model_info_file)
    
    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

        
        