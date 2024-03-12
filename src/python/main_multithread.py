'''
Script for ViComp

This script runs ViComp on different environments.
All training options can be specified in 'online.yaml'.


Example:
    python main_multithread.py --cfg online.yaml


Citation:

    @inproceedings{wang2024vicomp,
        author = {Yuxi Wang and Haibin Ling and Bingyao Huang},
        title = {ViComp: Video Compensation for Projector-Camera Systems},
        booktitle = {IEEE Virtual Reality and 3D User Interfaces (VR)},
        month = {April},
        year = {2024} }

'''

import threading
import queue

from time import sleep 
import logging

import cv2
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
from compensation import Models
from compensation.utils import saveImgs,convertImgs
import copy
import os
import finetune,casst
import torch.nn.functional as F
import torchvision
from conf import cfg, load_cfg_fom_args
import time
from os.path import join as fullfile
import shutil
from project2capture import Projecting,Capturing
from CMPScheduler import CMPScheduler
import schedule
from compensation.warping import *



logger = logging.getLogger(__name__)

is_saving_images = False

cam_quene = queue.Queue(maxsize=1)
cmp_quene = queue.Queue(maxsize=1)

load_cfg_fom_args('compensation')


# %% load training and validation data for CompenNe(S)t++

def load_model(model_name,pretrained_model,mask_corners = None):
 
    if model_name == 'ff2compennest':
        compen_nest = Models.CompenNeSt()
        warping_net = Models.FlowWarpingNet() 
        if pretrained_model !='':
            warping_net = nn.DataParallel(warping_net)
            compen_nest = nn.DataParallel(compen_nest)
            compen_nest.load_state_dict(torch.load(pretrained_model)) 
          
        compen_hd = Models.CompenNeStFF(warping_net, compen_nest)
                
    return compen_hd

def capture2detect(sch):

    schedule.every(0.01).seconds.do(sch.capture2detect)

    while True:
        schedule.run_pending()

def save_images(sch):
    while True:
        sch.save_images()

def save_training_images(sch):
    while True:
        sch.save_traindata()


def short_memory_update(sch):
    while True:
        sch.short_memory_update()


def long_memory_update(sch):
    while True:
        sch.long_memory_update()


def compensation_online(sch,im_st,im_len,shot_file):
    print('start compensation online!')
    sleep(1)

    shot_id = shot_file.readline()
    shot_id = 0 #  int(shot_index)
    sid = shot_id
    shot_id_last = im_st
    shot_count = 1
    shot_id =   im_st # int(shot_id)
    num_max = 4  # UPDATE 3 TIMES
    num_min = 1   # UPDATE 0 TIMES
    

    for i in range(im_st,im_len+im_st):
        if i==shot_id:
                sch.short_memory_reset(shot_id) # short_memory_reset
                shot_id_last = shot_id
                shot_id = shot_file.readline()
                shot_id = int(shot_id)
                shot_count = shot_count+1
                shot_freq = max(min(int((shot_id - shot_id_last)/(shot_count*10)),num_max),num_min)
                shot_step = int((shot_id-shot_id_last)/shot_freq)
                start_id = int(shot_id_last+shot_step)
                end_id = int(shot_id-shot_step)
               
                sch.long_memory_set_status(shot_id)
               
        sch.compensation(i)
    return


def compensation_pretrain(sch,im_st,im_len,shot_file):
    print('start compensation pretrain!')
    sleep(1)

    for i in range(im_st,im_len+im_st):
        sch.compensation(i)
    
    return

def compensation_finetune(sch,im_st,im_len,shot_file):
    print('start compensation finetune!')
    sleep(1)


    for i in range(im_st,im_len+im_st):
        sch.compensation(i)
    
    return
        

def capturing(filename):
    
    sleep(1)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
   
    cap.release()
    cv2.imwrite(filename, frame)
    
    return frame


def test(description):
    
    
    device = cfg.OPTIM_FINETUNE.DEVICE

    # evaluate on each severity and type of corruption in turn
    
    checkpoint_dir = '../../checkpoint'
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    
    for data_name in cfg.CORRUPTION.DATA_NAME:
        shot_file =  open(fullfile( cfg.CORRUPTION.DATA_DIR,data_name+'_shot_index.txt'),'r') 



        if is_saving_images:
            make_save_dirs(data_name)
        
        '''
        # multi-thread
        '''

        '''
        # start compensation and projecting
        '''
        print('start compensation and projecting!')
        if cfg.MODEL.PCMODELNAME == 'CompenNeSt':
            cam_size =  (1920,1080) 
            scn_size =  (1920,1080) 
            prj_size = (1024,1024)
        else:
    
            scn_size =  (1920,1080) 
            cam_size = (1920,1080) 
            prj_size = (1024,1024) 

        '''
        # project patterns and captured them for geometric correction
        '''
        project = Projecting()
        marker = np.zeros((240,240,3),dtype = np.uint8)

        im_ref = cv2.imread(fullfile(cfg.CORRUPTION.DATA_DIR, 'ref', 'img_gray.png'))

        # model_name = cfg.MODEL.ARCH+'_'+cfg.MODEL.SHORT+'_'+ cfg.MODEL.LONG+'_'+ cfg.MODEL.SHORT+'_'+ cfg.OPTIM_CASST.LOSS+'_' +str(cfg.OPTIM_CASST.MT)+'_'+str(cfg.OPTIM_CASST.RST)
        if cfg.MODEL.UNCMP:
            model_name = 'UNCMP'
        else:
            if cfg.MODEL.SUPDATE:
                model_name = cfg.MODEL.ARCH+'_'+cfg.MODEL.SHORT+'_'+ cfg.MODEL.LONG+'_'+ cfg.OPTIM_CASST.LOSS+'_' +str(cfg.OPTIM_CASST.MT)+'_'+str(cfg.OPTIM_CASST.RST)
            else:
                model_name = cfg.MODEL.ARCH+'_N_'+ cfg.MODEL.LONG+'_'+ cfg.OPTIM_CASST.LOSS+'_' +str(cfg.OPTIM_CASST.MT)+'_'+str(cfg.OPTIM_CASST.RST)
   

        sf_name = fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME, model_name, 'cam', 'ref','img_gray94.png')
 
        project.projecting(im_ref,marker)
        
        cv2.waitKey(5)
        
        cam_ref = capturing(sf_name)

        
        cb_name = fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME, model_name,'cam', 'cam_cbPrj.png')
        
        if cfg.MODEL.GCMODELNAME =='hmat':
            im_checkboard = cv2.imread(fullfile(cfg.CORRUPTION.DATA_DIR, 'cbPrj.png'))
        else:
            im_checkboard = cv2.imread(fullfile(cfg.CORRUPTION.DATA_DIR, 'texImg.png'))

        im_checkboard = cv2.resize( im_checkboard, prj_size)
        
        project.projecting(im_checkboard,marker)

        cv2.waitKey(5)

        cam_checkboard = capturing(cb_name)  

        cam_checkboard = cv2.resize(cam_checkboard,cam_size)
    
        bbox,prj_fov_mask,bbox_max_in,warping_field = calWarpingField(im_checkboard,cam_checkboard, cam_ref,model = cfg.MODEL.GCMODELNAME, cfg=cfg) #  model_pretrain= cfg.MODEL.GCMODEL_DIR)
        

        cam_ref_des_des_crop = cropCapturedImage(cam_ref,bbox,prj_fov_mask)


        cam_checkboard_torch = cropCapturedImage(cam_checkboard,bbox,prj_fov_mask)


        checkboard_warped ,checkboard_warped_torch =  warpImageUsingWF(cam_checkboard_torch, warping_field, prj_fov_mask = prj_fov_mask, bbox=bbox, prj_size=prj_size,model = cfg.MODEL.GCMODELNAME,isCMP=True)
        cam_ref_warped ,cam_ref_warped_torch =  warpImageUsingWF(cam_ref_des_des_crop, warping_field,prj_fov_mask = prj_fov_mask,bbox=bbox,prj_size=(1024,1024),model = cfg.MODEL.GCMODELNAME,isCMP=True)

        '''
        set model
        '''
        # configure model
        base_model = load_model(cfg.MODEL.PCMODELNAME, cfg.MODEL.PCMODEL_DIR)
        base_model = base_model.to(device)

        if cfg.MODEL.PCMODELNAME == 'ff2compennest':
            base_model.setFlow(cam_ref_des_des_crop,warping_field)

        # set models

        if cfg.MODEL.LONG == "casst":
            logger.info("test-time adaptation: CASST")
            long_model,long_opt = setup_casst(base_model)
        else:
            long_model,long_opt = setup_finetune(base_model)


        if cfg.MODEL.SHORT == "casst":
            logger.info("test-time adaptation: CASST")
            short_model,short_opt = setup_casst(base_model)
        else:
            short_model,short_opt = setup_finetune(base_model)
            


        savepath = fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name)
        imagepath = fullfile(cfg.CORRUPTION.DATA_DIR)
    
        archname = cfg.MODEL.ARCH
        sch = CMPScheduler(imagepath,savepath,archname,data_name, prj_fov_mask, bbox, bbox_max_in,short_model,short_opt,long_model,long_opt,cam_ref_warped_torch,cam_size =cam_size,prj_size=prj_size, uncmp=cfg.MODEL.UNCMP,supdate=cfg.MODEL.SUPDATE)
        
        td_cam = threading.Thread(name = 'cap2det',target = capture2detect,args=(sch,),daemon = True)
        td_cam.start()
        if  'Online' == cfg.MODEL.ARCH :
  
            td_cmp = threading.Thread(name = 'compensation',target = compensation_online,args=(sch,cfg.CORRUPTION.START,cfg.CORRUPTION.NUM,shot_file),daemon = True)
            td_cmp.start()
            td_short = threading.Thread(name = 'short_update',target = short_memory_update,args=(sch,),daemon = True)
            td_short.start()
            td_long = threading.Thread(name = 'long_update',target = long_memory_update,args=(sch,),daemon = True)
            td_long.start()
        
        if cfg.MODEL.ARCH == 'Pretrain':
            td_cmp = threading.Thread(name = 'compensation',target = compensation_pretrain,args=(sch,cfg.CORRUPTION.START,cfg.CORRUPTION.NUM,shot_file),daemon = True)
            td_cmp.start()
            
        if cfg.MODEL.ARCH == 'Finetune':
            td_cmp = threading.Thread(name = 'compensation',target = compensation_finetune,args=(sch,cfg.CORRUPTION.START,cfg.CORRUPTION.NUM,shot_file),daemon = True)
            td_cmp.start()
            td_short = threading.Thread(name = 'short_update',target = short_memory_update,args=(sch,),daemon = True)
            td_short.start()
      


        schedule.every(0.01).seconds.do(sch.projecting2)
        st = time.time()
        while sch.get_image_count()< cfg.CORRUPTION.NUM: 
            schedule.run_pending()
            # print(sch.get_image_count())
        running_time = time.time()-st
        print(running_time)
        sleep(1)
        return


            

def setup_finetune(model):
    """Set up finetune adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    params, param_names = finetune.collect_params(model)
    optimizer = setup_finetune_optimizer(params)
    finetune_model = finetune.Finetune( model, optimizer,
                           loss_type = cfg.OPTIM_FINETUNE.LOSS)

    return finetune_model,optimizer


def setup_casst(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model  = casst.configure_model(model)
    params, param_names = casst.collect_params(model)
    optimizer = setup_casst_optimizer(params)

    casst_model = casst.CASST(model, optimizer,
                           mt_alpha=cfg.OPTIM_CASST.MT,
                           loss_type = cfg.OPTIM_CASST.LOSS)

    return casst_model,optimizer


def setup_optimizer(params):

    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    

    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def setup_finetune_optimizer(params):

 
    if cfg.OPTIM_FINETUNE.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM_FINETUNE.LR,
                    betas=(cfg.OPTIM_FINETUNE.BETA, 0.999),
                    weight_decay=cfg.OPTIM_FINETUNE.WD)
    

    elif cfg.OPTIM_FINETUNE.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM_FINETUNE.LR,
                   momentum=cfg.OPTIM_FINETUNE.MOMENTUM,
                   dampening=cfg.OPTIM_FINETUNE.DAMPENING,
                   weight_decay=cfg.OPTIM_FINETUNE.WD,
                   nesterov=cfg.OPTIM_FINETUNE.NESTEROV)
    else:
        raise NotImplementedError

def setup_casst_optimizer(params):

    if cfg.OPTIM_CASST.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM_CASST.LR,
                    betas=(cfg.OPTIM_CASST.BETA, 0.999),
                    weight_decay=cfg.OPTIM_CASST.WD)
    

    elif cfg.OPTIM_CASST.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM_CASST.LR,
                   momentum=cfg.OPTIM_CASST.MOMENTUM,
                   dampening=cfg.OPTIM_CASST.DAMPENING,
                   weight_decay=cfg.OPTIM_CASST.WD,
                   nesterov=cfg.OPTIM_CASST.NESTEROV)
    else:
        raise NotImplementedError

    
def make_save_dirs(data_name):
    if cfg.MODEL.UNCMP:
        model_name = 'UNCMP'
    else:
        if cfg.MODEL.SUPDATE:
            model_name = cfg.MODEL.ARCH+'_'+ cfg.MODEL.SHORT+'_'+ cfg.MODEL.LONG+'_'+ cfg.OPTIM_CASST.LOSS+'_' +str(cfg.OPTIM_CASST.MT)+'_'+str(cfg.OPTIM_CASST.RST)
        else:
            model_name = cfg.MODEL.ARCH+'_N_'+ cfg.MODEL.LONG+'_'+ cfg.OPTIM_CASST.LOSS+'_' +str(cfg.OPTIM_CASST.MT)+'_'+str(cfg.OPTIM_CASST.RST)
   

    if not os.path.exists(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'cam')): os.makedirs(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'cam'))
    
    if not os.path.exists(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'cam','ref')): os.makedirs(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'cam','ref'))
    if not os.path.exists(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'cam','raw')): os.makedirs(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'cam','raw'))
    if not os.path.exists(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'cam','crop')): os.makedirs(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'cam','crop'))
    if not os.path.exists(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'cam','warp')): os.makedirs(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'cam','warp'))
    if not os.path.exists(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'pair_short')): os.makedirs(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'pair_short'))
    if not os.path.exists(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'pair_long')): os.makedirs(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'pair_long'))
    if not os.path.exists(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'prj')): os.makedirs(fullfile(cfg.CORRUPTION.DATA_DIR,cfg.CORRUPTION.SURFACE,data_name,cfg.MODEL.GCMODELNAME,model_name,'prj'))

    return


if __name__=='__main__':

    test('"compensation evaluation.')



    


