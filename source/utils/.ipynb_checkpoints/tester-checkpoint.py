#############################################################
# Copyright (c) Wilson Lam 2020                             #
# utils.tester contains classes that work with data & model #
# for testing. To use the module: `import utils.tester` or  #
# `from utils.tester import [class name]                    #
#############################################################

import os
import torch
import subprocess
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
from model import ModelMaker
from dataset.ENindoor67 import ENindoor67Preprocessed, ENindoor67StreamingBody, ENindoor67Datasets
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from matplotlib.ticker import MaxNLocator
from collections import Counter
from heapq import nlargest, nsmallest

BENCHMARK_MODEL = 'resnext101'


class ModelLoader:
    
    """
    Base class for Tester. Loads model artefacts.
    """
    
    def __init__(self,
                 model_loading_params,
                 model_weights='best_acc',
                 **kwargs):
        
        """
        Instantiates a ModelLoader.
        @params model_loading_params (dict) : keyword arguments
                                              to load a model
        @params model_weights (str)         : either `best_acc` or
                                              `min_val_loss`
                                              
        return:
        ModelLoader object.
        """
        
        # Value checking
        assert model_weights in ['best_acc', 'min_val_loss'],\
        "Invalid model weight option: {}.\
         Either `best_acc` or `min_val_loss`".format(model_weights)
        
        self._model_weights = model_weights  # model weights to store
        self._model_path = self._download_model(**model_loading_params)
        self.load_model()  # load model
        
        
    def _download_model(self,
                downloader,
                model_uri,
                local_path,
                session,
                training_job,
                model_file='model.tar.gz',
                **kwargs):
        
        """
        Helper function to download model from S3.
        @params downloader (S3Downloader)     : S3downloader object
        @params model_uri  (str)              : model's S3 location
        @params local_path (str)              : target path to store `model_file`
        @params session    (sagemaker.session): Sagemaker session
        @params training_job (str)            : name of training job
        @params model_file (str)              : model file name;
                                                default: `model.tar.gz`
        
        return:
        Path to model arefacts (str)
        """
        
        # Set paths & store training job name
        model_path = os.path.join(local_path, model_file)
        model_local_dir = os.path.join(local_path, training_job)
        self._training_job_name = training_job
        
        # Create directory, download model & unzip if file not exist yet
        if not os.path.exists(model_local_dir):
            # Create folder
            os.mkdir(model_local_dir)
            
            # Download model artefacts
            print("Downloading model - {} from {}.".format(training_job,
                                                          model_uri))
            downloader.download(s3_uri=model_uri,
                                local_path=local_path,
                                sagemaker_session=session)
            
            # Unzip artefacts
            print("Unzipping {}...".format(model_file))

            subprocess.check_call(['tar','-zxvf', model_path,
                                   '-C', model_local_dir+'/'])
            subprocess.check_call(['rm', '-rf', model_path])
        
        return model_local_dir
    
    
    def load_model(self,
                   files=dict(
                       info='model_info.pth',
                       model='model.pth',
                       summary='training_summary.pth'),
                   benchmark=BENCHMARK_MODEL):
        
        """
        Helper function to load model into ModeLoader object.
        @params files     (dict)    : collection of filenames
        @params benchmark (str)     : name of benchmark model
        """
        print("Loading model.")

        # Load the parameters used to create the model.
        model_info = {}
        model_info_path = os.path.join(self._model_path,
                                       files['info'])
        with open(model_info_path, 'rb') as f:
            model_info = torch.load(f)
            self.model_info = model_info
        
        print("model_info: {}".format(model_info))

        # Determine the device and construct the model
        device = torch.device("cuda"
                              if torch.cuda.is_available()
                              else "cpu")

        # Instantiate the model 
        model, _ = ModelMaker().make_model(**model_info)
        
        # Load the training summary
        summary_path = os.path.join(self._model_path,
                                    files['summary'])
        
        with open(summary_path, 'rb') as f:
            # Skip alternative weights for benchmark model
            if benchmark in self._training_job_name:
                skips = ['best_acc_model_wts',
                         'min_val_loss_model_wts']
                summary = torch.load(f, map_location='cpu')
                self._training_summary = {k:v
                                          for k, v in summary.items()
                                          if k not in skips}
            else:
                self._training_summary = torch.load(f,map_location='cpu')
        
        # Model weights to load
        load_wts = self._model_weights + '_model_wts'
        
        if benchmark in self._training_job_name:
            model_path = os.path.join(self._model_path,
                                      files['model'])
            with open(model_path, 'rb') as model_f:
                model_weights = torch.load(model_f,
                                           map_location='cpu')
                model.load_state_dict(model_weights)
        else:
            model.load_state_dict(self._training_summary[load_wts])
        
        # Set to eval mode
        model.to(device).eval()

        print("Done loading model.")
        self.model = model
        
        
    def to_model(self, model_weights, benchmark=BENCHMARK_MODEL):
        
        """
        Switch model weights to a model with best val acc
        or min val loss.
        
        @params model_weights (str)    : either `best_acc` or `min_val_loss`
        @params benchmark     (str)    : name of benchmark model
        """
        
        # Value checking
        assert model_weights in ['best_acc', 'min_val_loss'],\
        "Invalid model weight option: {}.\
        Either `best_acc` or `min_val_loss`".format(model_weights)
        
        # Limit switching for benchmark to avoid KeyError
        assert benchmark not in self._training_job_name, \
        "Cannot switch a benchmark model."
        
        # Switch and Load model weigths
        self._model_weights = model_weights
        load_wts = self._model_weights + '_model_wts'
        self.model.load_state_dict(self._training_summary[load_wts])
        
        
class TestLoader:
    
    """
    Helper & base class for loading test data.
    """
    
    def __init__(self, data, seed=1, **kwargs):
        
        """
        Instantiates a TestLoader object.
        @params data (str)    : local or S3 location of
                                preprocessed data
        @params seed (int)    : random seed
        """
        
        # Read data into dataloading objects
        if isinstance(data, BytesIO):
            self.dataset = ENindoor67Preprocessed(data)
            
        else:
            self.dataset = ENindoor67StreamingBody(data)
                
        print("Compiled test data.")
        
        # Set random seed
        self.seed = seed
    
    
    def _load_data(self, batch_size=32, shuffle=True):
        
        """
        Fetch Dataloader object.
        @params batch_size (int)   : size of each batch
        @params shuffle    (bool)  : if `True` the data
                                     will be shuffled
        
        Return torch.utils.data.DataLoader object
        """
        
        # Set random seed
        torch.random.manual_seed(self.seed)
        
        return DataLoader(self.dataset,
                          batch_size=batch_size,
                          shuffle=shuffle)
    
    def _get_class_dict(self):

        """
        Helper function to retrieve numerical label
        & actual class label mapping dictionary.
        """
        
        # Make sure there is a dataset 
        assert self.dataset is not None, \
        "Empty Dataset: please read in data first."

        return dict(zip(self.dataset.labels,
                        self.dataset.categories))

class Summaries(ModelLoader):
    
    """
    Extends ModelLoader; carries training
    +/- evaluation summaries.
    Parent class of TesterBase object.
    """
    
    def __init__(self, **kwargs):
        
        """
        Instantiates a Summaries object.
        @params keyword arugments from TesterBase
        
        Following attributes will be loaded with
        _read_summaries():
        @attr _conf_mat (np.array)    : confusion matrix
        @attr _predictions (np.array) : model predictions
        @attr _ground_truths(np.array): ground truth label
        @attr _acc_dict    (dict)     : label-accuracy mappings
        @attr _accuracy    (float)    : overall accuracy from test
        """
        
        ModelLoader.__init__(self, **kwargs)
        self._conf_mat = None
        self._predictions = None
        self._ground_truths = None
        self._accuracy = 0.0
        self._acc_dict = {}  # Old version attribute
        self._read_summaries(**kwargs)
    
    
    def __repr__(self):
        
        """
        Return all attributes except eval_summary
        """

        info = 'MIT INDOOR 67 SUMMARIES('

        for k, v in self.__dict__.items():

            if k == '_eval_summary':
                continue

            info += "        {}={}\n".format(k, v)

        info += ")"

        return info
    
    
    def _read_summaries(self, **kwargs):
        
        """
        Read in .pth files to Summaries object.
        Keyword arugments:
        @params summary_path (str)  : path to model artefacts
                                      & summary data
        """
        
        # Set paths
        if 'summary_path' in kwargs:
            eval_sum_pth = os.path.join(kwargs['summary_path'],
                                         'evaluation_summary.pth')
            
            # Old file format from LocalTester
            eval_sum_pkl = os.path.join(kwargs['summary_path'],
                                         'evaluation_summary.pkl')
     
            train_sum_path = os.path.join(kwargs['summary_path'],
                                          'training_summary.pth')
            
            model_info_path = os.path.join(kwargs['summary_path'],
                                           'model_info.pth')
            
            # Evaluation Summary
            
            if (os.path.exists(eval_sum_pth)
            or os.path.exists(eval_sum_pkl)):
            
                if os.path.exists(eval_sum_pth):
                    summary = torch.load(open(eval_sum_pth, 'rb'),
                                         map_location='cpu')
                else:
                    summary = pickle.load(open(eval_sum_pkl, 'rb'))

                items = self.__dict__.keys()
                for item in items:
                    try:
                        self.__dict__[item] = summary[self._model_weights][item]
                    except KeyError:
                        continue

            self._eval_summary = self.__dict__
            
            # Training Summary
            if os.path.exists(train_sum_path):
                self._training_summary = torch.load(open(train_sum_path, 'rb'),
                                                   map_location='cpu')
            
            # Model Info
            if os.path.exists(model_info_path):
                self.model_info = torch.load(open(model_info_path, 'rb'),
                                             map_location='cpu')

    
    def confusion_matrix(self, normalized=True, show=True):
        
        """
        Return confusion matrix of the evaluated model.
        @params normalized  (bool) : if `True` the matrix
                                     will be normalized;
        @params show (bool)        : if `True`, plot the matrix
        """
        
        assert self._conf_mat is not None, \
        "Empty Confusion Matrix. Please evaluate the model first."
        
        # Copy the confusion matrix
        conf_mat = self._conf_mat
        
        # Set plot title
        title = "Confusion Matrix"
        
        # Normalize the matrix & update title if `normalized` = True
        if normalized:
            conf_mat = normalize(conf_mat, axis=1, norm='l1')
            title = "Normalized " + title
        
        # Return matrix in np.array if `show` is False
        if not show:
            return conf_mat
        
        fig, ax = plt.subplots(figsize=(15,15))
        sns.heatmap(conf_mat, cmap='BuPu', linewidths=.5, ax=ax)
        ax.set_title(title)
        plt.show()
   

    def report(self, averaged=False, style=False):
        
        """
        Return classification report from evaluation as pd.Dataframe.
        """
        
        # Data Check
        assert (self._ground_truths is not None
                and self._predictions is not None),\
        "Empty data. Please evaluate the model first."
        
        # Get report as dictionary
        report_dict = classification_report(self._ground_truths,
                                            self._predictions,
                                            zero_division=0,
                                            output_dict=True)
        
        output = pd.DataFrame.from_dict(report_dict).T
        
        if averaged:
            output = pd.DataFrame.from_dict(report_dict).T[-3:]
        
        if style:
            return output.style
        
        return output
        
    
    def _sort_k(self, k, metric, ascending):
        
          
        """
        Base function of bottom_k() and top_k()
        @params k         (int): number of classes to show
        @params metric    (str): metric to determine sorting:
                                 `precision`, `recall`, or `f1`
        @params ascending(bool): `True` for bottom_k(),
                                 `False` for top_k()
        """
        
        # Data Check
        assert (self._ground_truths is not None
                and self._predictions is not None),\
        "Empty data. Please evaluate the model first."
        
        assert k <= len(self._predictions), \
        "Invalid number for MIT 67 Indoort Dataset. \
        Must be smaller or equal to {}." \
        .format(len(self._predictions))
        
        assert metric in ['precision', 'recall', 'f1-score'], \
        "Metric can only either be `precision`, `recall`,"\
        " or `f1`, not {}".format(metric)
        
        try:
            class_dict = self._get_class_dict()
            
        except:
            if '_acc_dict' in self.__dict__:  # Old version attribute
                class_dict = dict([label.split(' - ')
                                   for label in self._acc_dict])
            
            else:
                class_dict = self.class_dict
                
        
        return (self.report()
                   .reset_index()
                   .replace(class_dict)
                   .rename(columns={'index':'category'})
                   .set_index('category')[:-3][metric]
                   .sort_values(ascending=ascending)[:k]
                   .apply(pd.Series)
                   .rename(columns={0:metric})
                   .style)
        
    
    
    def bottom_k(self, k, metric='precision'):
        
        """
        Return the lowest k classes w.r.t `metric`.
        @params k      (int): number of classes to show
        @params metric (str): metric to determine sorting:
                              `precision`, `recall`, or `f1`
        """
        
        return self._sort_k(k=k, metric=metric, ascending=True)
        
    
    def top_k(self, k, metric='precision'):
        
        """
        Return the highest k classes w.r.t `metric`.
        @params k      (int): number of classes to show
        @params metric (str): metric to determine sorting:
                              `precision`, `recall`, or `f1`
        """
        
        return self._sort_k(k=k, metric=metric, ascending=False)
            
        
    def training_history(self):
        
        """
        Plot training and validation accuracy and loss.
        """
        
        # Check if value exists
        assert self.model_info is not None, \
        "Model info not found."
        assert self._training_summary is not None, \
        "Training summary not found."
        
        # Load information and write to title
        
        # Model information 
        model_info = "model: {}".format(self.model_info['model'])
        
        # Hyperparameters, model architecture, and optimizer settings
        hp = ', '.join(["{}={}".format(k,v)
                        for k, v in self._training_summary['hyperparameters'].items()
                        if k in ['batch_size',
                                 'optimizer',
                                 'lr',
                                 'dropout',
                                 'sampling',
                                 'blocks_unfrozen']])
        
        # Best validation accuracy 
        best_val_acc = "best val acc: \
        {}".format(self._training_summary['best_acc'])
        
        # Minimum validation loss
        min_val_loss = "min val loss: \
        {}".format(self._training_summary['min_val_loss'])
        
        # Setting layouts 
        fig, axes = plt.subplots(1, 2, figsize=(16,8))
        fig.subplots_adjust(top=0.8)
        
        # Enumerate `training_summary` dict and plot data
        aspects = ['loss', 'accuracy']
        for i, aspect in enumerate(aspects):
            hist = self._training_summary[aspect]
            for phase in ['train', 'val']:
                axes[i].plot(np.arange(1, len(hist[phase])+1), hist[phase], label=phase)
                axes[i].set_title(aspect.upper())
                axes[i].set_xlabel('epochs')
                axes[i].set_ylabel(aspect)
                axes[i].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                axes[i].legend()

        
        # Write title
        fig.suptitle("Training History\
        \n{}\n{}\n{}\n{}".format(model_info,
                                  hp,
                                  best_val_acc,
                                  min_val_loss), fontsize=16, y = 0.98)
        plt.show()

    
class TesterBase(TestLoader, Summaries):
    
    """
    Base class for Testers:
    Parent classes: TestLoader & Summaries
    """
    
    def __init__(self, **kwargs):
                               
        """
        Construct by passing kwargs to parent classes
        """
                               
        Summaries.__init__(self, **kwargs)
        TestLoader.__init__(self, **kwargs)
        
                               
class LocalTester(TesterBase):
                               
    """
    Object to administer testing at local computer.
    Parent class: TesterBase
    """                       
    
    def __init__(self,
                 show_history=False,
                 **kwargs):
        
        """
        Construct LocalTester.
        @params show_history  (bool) : if `True`, plot training
                                       history; default: `False`
        @params keyword arguments to be passed to parent class
        """
        # Pass kwargs to parent class                       
        TesterBase.__init__(self, **kwargs)
        self.class_dict = self._get_class_dict()
                               
        # Plot training history if show_history is `True`                     
        if show_history:
            self.training_history()
          
                               
    def __repr__(self):
                               
        """
        Show Training summary / test summary.
        """
        
        skip_items = ['training_time',
                      'best_acc_model_wts',
                      'min_val_loss_model_wts']
        
        details = self._training_summary
        info = 'MIT_INDOOR_67_TRAINING_SUMMARY('
        if self._eval_summary:
            info = 'MIT_INDOOR_67_TESTING SUMMARY('
            details = self._eval_summary
            skip_items += ['_training_summary']
        
        for k, v in details.items():
            
            if k in skip_items:
                continue
            
            info += "        {}={}\n".format(k, v)
        
        info += ")"
        
        return info
          
                               
    def run(self, batch_size=32):
        
        """
        Evaluate model on test set;
        load test set from grandparent class TestLoader.
        
        @param batch_size  (int)    : size of each batch
        
        Write evaluation summary to self._eval_summary
        and file `evaluation summary.pth`
        """
                                  
        device = torch.device("cuda"
                              if torch.cuda.is_available()
                              else "cpu")
        
        # Initialize variables
        num_classes = self._training_summary['hyperparameters']['num_classes']
        self._conf_mat = np.zeros((num_classes, num_classes),int)  
        ground_truth = np.array([])
        predicted = np.array([])
        dataloader = self._load_data(batch_size=batch_size)
        
        print("Start evaluating model on test data...")
        
        self.model.to(device)
        
        # Iterate through each batch of test data
        for i, batch in enumerate(dataloader):
            
            batch_x, batch_y = batch
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            with torch.set_grad_enabled(False):
                y_preds = self.model(batch_x)
                _, preds = torch.max(y_preds.data, 1)
            
            ground_truth = np.append(ground_truth,
                                     batch_y.numpy())
            predicted = np.append(predicted,
                                  preds.numpy())
            
            if (i+1) % 10 == 0:
                print("{} / {} batch of test data evaluated"
                      .format(i+1, len(dataloader)))
        
        # Store y and y_hat in LocalTester instance
        self._ground_truths = ground_truth.astype('int32')
        self._predictions = predicted.astype('int32')
        
        # Count y and y_hat pairs
        temp_dict = Counter(zip(self._ground_truths,
                                self._predictions))
        
        # Enumerate y and y_hat and store values in confusion matrix 
        for k, v in temp_dict.items():
            self._conf_mat[k] = v
        
        # Calcuate the overall Top-1 accuracy of each class
        true_positives = np.diag(self._conf_mat)
        self._accuracy = (true_positives.sum() / 
                               (len(true_positives) * 
                                1.0))
        
                               
        print("Evaluation ends.")
        print("TEST ACC: {}".format(self._accuracy))
        
        # Write evaluation summary to instance and file
        self.eval_summary(verbose=False, to_pickle=True)
      
                               
    def eval_summary(self, verbose=True, to_pickle=False):
                               
        """
        Store evaluation summary to instance.
        @params verbose  (bool)    : if `True`, print out summary
        @params to_pickle(bool)    : if `True`, write summary to file
        """
        
        # Items to skip 
        skip_items = ['model',
                      'dataset',
                      '_model_path',
                      'seed',
                      '_model_weights',
                      '_training_summary',
                      'model_info',
                      '_eval_summary']
        
        # Store values in a dictionary
        self._eval_summary = {
                              self._model_weights :
                                  {
                                      k : v
                                      for k, v in self.__dict__.items()
                                      if k not in skip_items
                                  }
                             }
        
        # Print evaluation summary
        if verbose:
            print(self)
        
        # Write to file
        if to_pickle:
            eval_file = os.path.join(self._model_path, 'evaluation_summary.pth')
            
            if os.path.exists(eval_file):
                existing_summary = torch.load(open(eval_file, 'rb'),
                                              map_location='cpu')
                self._eval_summary.update(existing_summary)

            with open(eval_file, 'wb') as f:
                torch.save(self._eval_summary, f)
                

# Under construction
# class S3Tester(TesterBase):
    
#     def __init__(self, model, **kwargs):
#         TesterBase.__init__(self, **kwargs)
#         self.dataloader = self._load_data(batch_size=200)
#         self.model = model
    
#     def run(self):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(device)
#         self._predictions = np.array([])
#         self._ground_truths = np.array([])
        
#         for batch in self.dataloader:
            
#             batch_x, batch_y = batch
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#             with torch.set_grad_enabled(False):
#                 y_preds = self.model(batch_x)
#                 _, preds = torch.max(y_preds.data, 1)
            
#             self._ground_truth = np.append(self._ground_truth,
#                                            batch_y.cpu().detach().numpy())
#             self._predictions = np.append(self._predictions,
#                                           preds.cpu().detach().numpy())
        
#         assert len(self._ground_truth) == len(self._predictions)
#         length = len(self._ground_truth)
#         output = np.concatenate([self._predictions,
#                                  self._ground_truths]).reshape(2, length)
            
#         return output
        
    
    
    