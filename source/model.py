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
# model.py contains model classes that is used during training                  #
# To use the models: `import model` or `from model import [class name]          #
#################################################################################


import torch
import torch.nn as nn
import importlib
import pkgutil
from pathlib import Path
from efficientnet_pytorch import EfficientNet
from efficientnet_lite_pytorch import EfficientNet as EfficientNetLite
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
from efficientnet_lite1_pytorch_model import EfficientnetLite1ModelFile
from efficientnet_lite2_pytorch_model import EfficientnetLite2ModelFile


# Configuration of fc layer: `shrink` - every layer shrinks by half
# `expand` - every layer expands by double
TUNE_CONFIG = 'shrink'

LITE_WEIGHTS = [EfficientnetLite0ModelFile.get_model_file_path(),
                EfficientnetLite1ModelFile.get_model_file_path(),
                EfficientnetLite2ModelFile.get_model_file_path()]

class EfficientNetBase(nn.Module):
    
    """
    Base class for EfficientNets.
    Parent Class: torch.nn.Module
    """
    
    def __init__(self, version, num_classes, **kwargs):
        
        """
        Construct EfficientNetBase.
        @params  version     (str)    : version of EfficientNet
        @params  num_classes (int)    : number of output classes
        @params  keyword arugments from child class
        """
        
        super().__init__()
        # Instantiate an EfficientNet model
        
        if 'lite' in version:
            
            lite_version = version[-1]
            
            assert lite_version.isnumeric(),\
            "EfficientNet Lite version must be an integer, got {}"\
            .format(lite_version)
            
            lite_version = int(lite_version)
            
            assert lite_version < len(LITE_WEIGHTS),\
            "EfficientNet Lite version only offer up to {}; got Lite{}"\
            .format(len(LITE_WEIGHTS)-1, lite_version)
            
            self.model = (EfficientNetLite
                          .from_pretrained(version,
                                           weights_path=LITE_WEIGHTS[lite_version],
                                           num_classes=num_classes))
        
        else:
            self.model = (EfficientNet
                          .from_pretrained(version,
                                           num_classes=num_classes))
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set dropout
        if 'dropout' in kwargs:
            dropout = kwargs.get('dropout')
            self.model._dropout = nn.Dropout(p=dropout)
        
        # Unfreeze blocks
        if 'blocks_unfrozen' in kwargs:
            blocks_unfrozen = kwargs.get('blocks_unfrozen')
            
            # Value checking
            assert blocks_unfrozen <= len(self.model._blocks),\
            "Too many blocks to unfreeze"
            
            if blocks_unfrozen > 0:
                self._unfreeze_blocks(blocks_unfrozen)
        
        # Unfreeze final fc layer
        self._unfreeze_fc()
                                
            
    def forward(self, x):
        
        """
        Forward propagation.
        @param x  (Tensor array):  input data
        """

        return self.model(x)
    
    
    def _unfreeze_blocks(self, n):
        
        """
        Unfreeze `n` number of blocks in the model.
        """
        
        # Turn on the blocks
        for block in self.model._blocks[-n:]:
            for _, layer in list(block.named_modules()):
                for param in layer.parameters():
                    param.requires_grad = True
                    
        # Turn off the BatchNorm2d Layer                
        for _, layer in list(self.model.named_modules()):
            if isinstance(layer, torch.nn.BatchNorm2d):
                for param in layer.parameters():
                    param.requires_grad = False 
        
                                                                
    def _unfreeze_fc(self):
        
        """
        Unfreeze the final fc layer.
        """
        
        for param in self.model._fc.parameters():
            param.requires_grad = True

            
class EfficientNetTuned(EfficientNetBase):
    
    """
    EfficientNet with expanded final fc layer.
    Parent class: EfficientNetBase
    """
    
    def __init__(self,
                 version,
                 num_classes,
                 depth=1,
                 config=TUNE_CONFIG,
                 **kwargs):
        
        """
        Construct EfficientNetTuned model.
        @params  version    (str)  : version of EfficientNet
        @params  num_classes(int)  : number of output nodes
        @params  depth      (int)  : number of layers in fc
        @params  config     (str)  : `shrink` or `expand`
                                     if `shrink` current layer
                                     is half the size of the previous;
                                     if  `expand` doubles that
                                     of the previous layer
        @params keyword arguments pass to parent class
        """
        # Instantiate parent class
        super().__init__(version, num_classes, **kwargs)
        
        # Value checking
        assert config in ['shrink', 'expand'], "FC layer can only be `shrink` or `expand`"
        
        # Default dropout
        dropout = 0.2
        
        # Update if `dropout` in kwargs
        if 'dropout' in kwargs:
            dropout = kwargs['dropout']
            self.model._dropout = nn.Dropout(p=dropout)
        
        # Input dimension from previous layer
        in_dim = self.model._fc.in_features
        
        # Generate layers
        modules = []
        modules = self._generate_fc_layers(modules,
                                           in_dim,
                                           num_classes,
                                           dropout,
                                           config,
                                           depth)
        
        # Set hidden layers to the final fc layer
        self.model._fc = nn.Sequential(*modules)
        
        # Unfreeze the fc layer
        self._unfreeze_fc()
        
        
    def _generate_fc_layers(self,
                            modules,
                            in_dim,
                            num_classes,
                            dropout,
                            config,
                            depth=1):
        
        """
        Recursive function to generate layers for fc.
        @params modules    (list)  : list of torch.nn layers
        @params in_dim     (int)   : dimension of input
        @params num_classes(int)   : size of output
        @params dropout    (float) : probability of dropout
        @params config     (str)   : configuration of extra layers
        @params depth      (int)   : number of extra hidden layers in fc
        """
        
        # Base case
        if ((in_dim // 2) <= (num_classes * 2)) or (depth == 1):
            modules.append(nn.Linear(in_dim, num_classes))
            return modules
        
        # Configuration
        if config == 'shrink':
            out_dim = in_dim // 2
        else:
            out_dim = in_dim * 2
        
        # Expand the layers
        modules.append(nn.Linear(in_dim, out_dim))
        modules.append(nn.LeakyReLU(inplace=True))
        
        # Recursive call
        return self._generate_fc_layers(modules, out_dim, num_classes, dropout, config, depth - 1)
        

# EfficientNet and EfficientNet-tuned
        
class efficientnet_b0(EfficientNetBase):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)

class efficientnet_lite0(EfficientNetBase):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)
        
class efficientnet_b0_tuned(EfficientNetTuned):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)

class efficientnet_b1(EfficientNetBase):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)
        
class efficientnet_lite1(EfficientNetBase):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)
        
class efficientnet_b1_tuned(EfficientNetTuned):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)

class efficientnet_b2(EfficientNetBase):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)

class efficientnet_lite2(EfficientNetBase):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)

class efficientnet_b2_tuned(EfficientNetTuned):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)
        
class efficientnet_b3(EfficientNetBase):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)
        
class efficientnet_b3_tuned(EfficientNetTuned):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)
        
class efficientnet_b4(EfficientNetBase):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)
        
class efficientnet_b4_tuned(EfficientNetTuned):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)

class efficientnet_b5(EfficientNetBase):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)

class efficientnet_b5_tuned(EfficientNetTuned):
    def __init__(self, version, num_classes, **kwargs):
        super().__init__(version, num_classes, **kwargs)
        

class resnext101(nn.Module):
    """
    Benchmark model - `ResneXt101_32x16d`:
    https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth
    Replicate from ashrutkumar (2019) github repo:
    https://github.com/ashrutkumar/Indoor-scene-recognition/blob/master/indoor_scene_recognition.ipynb
    
    MIT License

    Copyright (c) 2019 Ashrut Kumar

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__()
        
        # load the model
        self.model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        
        # freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # customize the final layer
        self.model.fc = nn.Sequential(nn.Dropout(p=0.5),
                                     nn.Linear(2048, 1024),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(1024,num_classes))
#                                      nn.Softmax(dim=1))
        
    def forward(self, x):
        """
        return predicted output Softmax(dim=1)
        """
        
        return self.model(x)

    
class ModelMaker:
    
    def __init__(self):
        """
        Helper class to retrieve target model from this module.
        """       
        BASECLASS = nn.Module
        # retrive models
        self.models = dict(self._locate_models(baseclass=BASECLASS))
    
    def _locate_models(self, baseclass):
        """
        recursive function to locate all subclass models of baseclass
        within this module.
        @param baseclass (class) : baseclass of target models
                                   default(nn.Module)
        return:
        models (list of tuples) : [(class_name_1, class obj_1)...]
        """
        
        models = []
        
        # Base case
        if not baseclass.__subclasses__():
            return [(baseclass.__name__, baseclass)]
        
        # Loop through all subclasses to locate subclasses of subclasses
        for cls in baseclass.__subclasses__():
            if cls.__module__ == Path(__file__).stem:
                models += self._locate_models(cls)

        return models

    def make_model(self, model, **kwargs):
        
        """
        main function to instantiate a model object.
        @param model (str) : target model to instantiate
        @param **kwargs : keyword arguments required by
                          the constructor of models
        
        return:
        target model instance, dict of model information
        """


        model = model.lower()

        assert model.replace("-", "_") in self.models,\
        "{} is not a valid / an available model.".format(model)

        for _, name, _ in pkgutil.iter_modules([__file__]):
            importlib.import_module(Path(__file__).stem + '.' + model)

        model_info = dict(model=model, **kwargs)

        if 'efficientnet' in model:

            if "tuned" in model:

                return ((self.models[model.replace("-", "_")]
                        (version=model.split('-tuned')[0],
                         **kwargs)),
                        model_info)

            return ((self.models[model.replace("-", "_")]
                     (version=model, **kwargs)),
                    model_info)

        return self.models[model](**kwargs), model_info
        