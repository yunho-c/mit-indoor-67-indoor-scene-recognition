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
# utils.trainer contains `Trainer` class working with data & model in training. #
# To use: `import utils.trainer` or `from utils.trainer import Trainer`         #
#################################################################################


import time
import copy
import torch
import numpy as np
from utils.helpers import initialize_summary

class Trainer:
    """
    Helper class to implement training of models.
    """
    
    def __init__(self):
        """
        Constructor takes no arguments
        """
        pass
    
    @staticmethod
    def run(model,
            train_loader,
            val_loader,
            epochs,
            criterion,
            optimizer,
            device,
            patience):
        
        """
        This is the training method that is called by the PyTorch training script.
        The parameters passed are as follows:
        @params model        - The PyTorch model that we wish to train.
        @params train_loader - The PyTorch DataLoader that should be used during training.
        @params val_loader   - The PyTorch DataLoader that should be used during validation.
        @params epochs       - The total number of epochs to train for.
        @params criterion    - The loss function used for training. 
        @params optimizer    - The optimizer to use during training.
        @params device       - Where the model and data should be loaded (gpu or cpu).
        @params patience     - No. of non-loss-improving trials before early stopping.
        
        Return:
        Model with the least validation loss,
        Summary of the training - include both best acc and min val loss weights.
        """

        model.to(device)

        print("""<========== TRAINING JOB BEGINS ==========>""")
        
        # Initialize variables
        since = time.time()
        best_acc_model_wts = copy.deepcopy(model.state_dict())
        min_val_loss_model_wts = copy.deepcopy(model.state_dict())
        min_val_loss = np.inf
        best_acc = 0.0
        epoch_no_improve = 0
        summary = initialize_summary()
        
        # Begin training
        for epoch in range(1, epochs + 1):
            print("<============= Epoch {} / {} =============>"
                  .format(epoch, epochs))

            
            # Switch phase
            for phase in ['train', 'val']:

                if phase == 'train':
                    model.train()
                    loader = train_loader

                else:
                    model.eval()
                    loader = val_loader
                
                epoch_start = time.time()
                total_loss = 0
                size = 0
                correct = 0
            
                for i, batch in enumerate(loader):
                    # get data
                    batch_x, batch_y = batch

                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        y_preds = model(batch_x)
                        _, preds = torch.max(y_preds.data, 1)
                        loss = criterion(y_preds, batch_y)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    total_loss += loss.item()

                    correct += torch.sum(preds == batch_y.data)
                    size += batch_y.size(0) 
                    
                    # Print every 100 batch
                    if phase == 'train':
                        if i > 0 and (i+1) % 100 == 0:
                            print("  {} / {} batch loaded for training."
                                  .format(i+1, len(loader)))
                
                # Record training time for each epoch
                if phase == 'train':        
                    summary['training_time'].append((time.time() - epoch_start))
                
                # Record loss and accuracy
                epoch_loss = (total_loss / len(loader))
                epoch_acc = (correct / (size * 1.0)).item()
                
                # Report Loss and Acc
                print("Epoch: {} - {}_LOSS: {:.4f}, {}_ACC: {:.4f}"
                      .format(epoch, phase.upper(),
                              epoch_loss, phase.upper(), epoch_acc))            
                summary['loss'][phase].append(epoch_loss)
                summary['accuracy'][phase].append(epoch_acc)
                
                # Keep track of no improvement epochs
                if phase == 'val':

                    if epoch_loss < min_val_loss:
                        epoch_no_improve = 0
                        min_val_loss_model_wts = copy.deepcopy(model.state_dict())
                        min_val_loss = epoch_loss
                    else: 
                        epoch_no_improve += 1
                    
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_acc_model_wts = copy.deepcopy(model.state_dict())
            
            # Early Stopping
            if epoch_no_improve == patience:
                print("Early Stopping at epoch {}.".format(epoch))
                break

        print("""<============ TRAINING JOB FINISHES ============>""")
        
        # Documenting Training History
        time_elapsed = time.time() - since
        summary['min_val_loss'] = min_val_loss
        summary['best_acc'] = best_acc
        summary['time_elapsed'] = ('{:.0f}m {:.0f}s'
                                  .format(time_elapsed // 60, time_elapsed % 60))
        
        # This is a bit of cheating
        # but we save best val acc and loss for comparison in case they are different
        summary['best_acc_model_wts'] = best_acc_model_wts
        summary['min_val_loss_model_wts'] = min_val_loss_model_wts
        print('Training complete in {}'.format(summary['time_elapsed']))
        print('MIN VAL LOSS: {:4f}'.format(summary['min_val_loss']))
        print('BEST VAL ACC: {:4f}'.format(summary['best_acc']))

        # Load best model weights: Validation Accuracy
        model.load_state_dict(best_acc_model_wts)

        return model, summary