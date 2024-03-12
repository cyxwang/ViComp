from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import copy
import PIL
import torchvision.transforms as transforms

import time 
import logging
from compensation.utils import computeLoss
from torchvision.transforms import Resize


def update_ema_variables(ema_model, anchor_model,model, alpha_teacher):
    for ema_param, anchor_param, param in zip(ema_model.parameters(),anchor_model.parameters(), model.parameters()):
        ema_param.data[:] = 0.6*alpha_teacher * ema_param[:].data[:] + 0.4*alpha_teacher * param[:].data[:]+\
         (1 - alpha_teacher) * anchor_param[:].data[:]
    return ema_model



class CASST(nn.Module):

    def __init__(self,  model, optimizer,  mt_alpha=0.8, loss_type='l1+l2+ssim'):
        super().__init__()
        self.model =  model
 
        self.optimizer = optimizer

        self.loss_type = loss_type

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.mt = mt_alpha

  

    def forward(self, x,y,yt):
        loss, outputs = self.forward_and_adapt(x, y, yt, self.optimizer)
        return loss,outputs
    def setAnchorModel(self,short_model):
        model_state = deepcopy(short_model.state_dict())
        self.model_anchor.load_state_dict(model_state, strict=True)

    
    def getTeacherModel(self):
        return self.model_ema

    def reset(self,model,optim):
        
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
     
    def compensation(self,x):

        return self.model_ema(x,upsampling=True).detach()
    
    def compensation_wo_updae(self,x):
   
        return self.model_anchor(x,upsampling=True).detach()
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, y,yt, optimizer):
      
        self.model.train()

        outputs_long = self.model(x,upsampling=True) # student

        standard_ema = self.model_ema(x,upsampling=True) # teacher
        
        loss = computeLoss(outputs_long,y,loss_option='l1+l2+ssim') + \
            computeLoss(standard_ema,y,loss_option='l1+l2+ssim') + \
            +2*computeLoss(x,yt,loss_option='l1+l2+ssim')
        
      
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
  
        self.model_ema = update_ema_variables(ema_model = self.model_ema, anchor_model = self.model_anchor,  model = self.model, alpha_teacher=self.mt)
   
        return loss, standard_ema



def collect_params(model):

    params = []
    names = []

    for nm, m in model.named_modules():
        if True:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names



def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    for param in model_anchor.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    
    model.train()
    
    for m in model.modules():
        m.requires_grad_(True)

    return model


