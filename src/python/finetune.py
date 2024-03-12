from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import numpy as np
import torch.nn.functional as F
import PIL
import torchvision
import torchvision.transforms as transforms

from time import time
import logging
from compensation.utils import computeLoss
import copy
import torch.optim as optim
from torchvision.transforms import Resize



class Finetune(nn.Module):

    def __init__(self, model, optimizer, loss_type='l1+l2+ssim'):
        super().__init__()
        

        self.model = model
        self.model_anchor = model
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer = optimizer   
        self.loss_type = loss_type

        self.count = 0
        self.title = 'finetune'

        self.shot_id_last = -1
        self.shot_id_curr = -1



    def forward(self, x,y,yt):

        loss, outputs = self.forward_and_finetune(x, y, yt)

  
        return loss,outputs

    def reset(self, model, optimizer, shot_id, loss_type = 'l1+l2+ssim'):

        self.model_state = copy.deepcopy(model.state_dict())
     

        self.shot_id_last = self.shot_id_curr
        self.shot_id_curr = shot_id

        self.model.load_state_dict(self.model_state, strict=True)

        self.model.train()
       

    def getTeacherModel(self):
        return self.model
    
    def setAnchorModel(self,short_model):
        model_state = deepcopy(short_model.state_dict())
        """Restore the model and from short model."""
        self.model_anchor.load_state_dict(model_state, strict=True)
        
    def getModel(self):
        return self.model

    def compensation(self,x):

        return self.model(x,upsampling=True).detach()
    
    '''
    x : 1024*1024
    s : 1024*1024
    y : 1024*1024
    yt: 1024*1024

    
    '''
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    
    def forward_and_finetune(self, x, y,yt):


        outputs = self.model(x,upsampling=True) 

        
        loss = computeLoss(outputs,y,loss_option=self.loss_type)  + 2*computeLoss(x,yt,loss_option=self.loss_type) 
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        

        return loss, outputs


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
                    
    return params, names


