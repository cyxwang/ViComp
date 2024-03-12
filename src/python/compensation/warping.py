


import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

import numpy as np
import torch.nn as nn
from torchvision.transforms import Resize
from PIL import Image

from compensation import ImgProc
from compensation.utils import repeat_np

from compensation.gma.network import RAFTGMA

from compensation.flowformer.FlowFormer.LatentCostFormer.transformer import FlowFormer


import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() >= 1:
    print('Train with GPU!')
else:
    print('Train with CPU!')


def saveTrainingImgs(inputData, savename):

    if type(inputData) is torch.Tensor:
        if inputData.requires_grad:
            inputData = inputData.detach()
        if inputData.device.type == 'cuda':
            imgs = inputData.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        else:
            imgs = inputData.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)

    else:
        imgs = inputData

    # imgs must have a shape of (N, row, col, C)
    imgs = np.uint8(imgs[:, :, :, ::-1] * 255)  # convert to BGR and uint8 for opencv

    cv2.imwrite(savename, imgs[0, :, :, :])  # faster than PIL or scipy

def calCamRegionBoundingBox(im_surf):
        im_surf_torch = cv2.cvtColor(im_surf,cv2.COLOR_BGR2RGB)
        im_surf_torch = torch.from_numpy(im_surf_torch).unsqueeze(0).permute((0, 3, 1, 2)).float().div(255)
 
        # threshold im_diff with Otsu's method

        im_mask, mask_corners,bbox = ImgProc.thresh(im_surf)

        bbox_max_in = ImgProc.getLargestRect(im_mask)

        prj_fov_mask = repeat_np(torch.Tensor(np.uint8(im_mask)).unsqueeze(0), 3, 0)

        prj_fov_mask = prj_fov_mask.bool()


        torch_resize = Resize([1024,1024])
        fov_out = torch_resize(prj_fov_mask[:,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]])
        return bbox, fov_out.unsqueeze(0),bbox_max_in

def calWarpingFieldFromTextureImage(flow_name,cfg,im_cam,im_prj,im_surf):
        bbox,prj_fov_mask, bbox_max_in = calCamRegionBoundingBox(im_surf)
        h,w,c =  im_prj.shape

        im_cam_torch = cropCapturedImage(im_cam,bbox,prj_fov_mask)
        im_prj_torch = cv2.cvtColor(im_prj,cv2.COLOR_BGR2RGB)
        im_prj_torch = torch.from_numpy(im_prj_torch).unsqueeze(0).permute((0, 3, 1, 2)).float().div(255)

        im_cam_torch = im_cam_torch.to(device)
        im_prj_torch = im_prj_torch.to(device)

    
        lambda0 = 1.0
        if flow_name == 'flowformer':
            lambda0 = 1.0
            flow_model = FlowFormer(cfg.latentcostformer)# .to(device)
            flow_model = nn.DataParallel(flow_model).to(device)
            flow_model.load_state_dict(torch.load(cfg.latentcostformer.pretrained_model))
            print(f"====Loaded FLOWFORMER checkpoint at {cfg.latentcostformer.pretrained_model}")

        flow_model.eval()
        outflow = torch.zeros((1,2,h,w))
        with torch.no_grad():

                tmp1 = im_prj_torch.to(device)
                tmp2 = im_cam_torch.to(device)

          
                if flow_name == 'flowformer':
                        flow_train = flow_model(tmp1,tmp2)
                     
                             
                flow = flow_train[0][0] *lambda0 # * 20.0
                flow = flow.cpu().data.numpy()
                flow = np.swapaxes(np.swapaxes(flow, 0, 1), 1, 2) # 
                u_ = cv2.resize(flow[:,:,0],(h,w))
                v_ = cv2.resize(flow[:,:,1],(h,w))
                flow = np.dstack((u_,v_))
                flow = torch.from_numpy(flow).to(device)
                flow = flow.permute((2,0,1)) 
                outflow[0,:,:,:] = flow 

        return  bbox,prj_fov_mask,bbox_max_in,outflow.cuda()

# %% local functions
def warpImagesUsingFlow(input, flow,prj_fov_mask, bbox ,out_size = (256,256),isCMP = False):
        
        if len(input.shape)<4:
             input.unsqueeze(0)

        if isCMP:
             imgs = input
        else:
             
             imgs = input[:,:,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]

        torch_resize = Resize([out_size[1],out_size[0]])
        
        imgs = torch_resize(imgs)
        imgs[~prj_fov_mask] = 0

        imgs = torch_resize(imgs)

        imgs_size = imgs.shape
        scale_x = out_size[0]/imgs_size[2]
        scale_y = out_size[1]/imgs_size[3]


        xx = torch.arange(0,out_size[0]).view(1,-1).repeat(out_size[1],1)
        yy = torch.arange(0,out_size[1]).view(-1,1).repeat(1,out_size[0])
        xx = xx.view(1,out_size[0],out_size[1])
        yy = yy.view(1,out_size[0],out_size[1])
        grid = torch.cat((xx,yy),0).float().cuda()  # [1,2,h,w]
         
        up_flow = flow.cpu().data.numpy()
        u_ = cv2.resize(up_flow[0,0,:,:],(out_size[0],out_size[1]))*scale_x
        v_ = cv2.resize(up_flow[0,1,:,:],(out_size[0],out_size[1]))*scale_y
        up_flow = np.dstack((u_,v_))
        up_flow = torch.from_numpy(up_flow).cuda()
        
        up_flow = up_flow.permute(2,0,1)

        coarse_grid = grid+up_flow
        coarse_grid = 2.0*coarse_grid/(out_size[0]-1)-1.0  # [-1,1]
        
        coarse_grid = coarse_grid.repeat(imgs_size[0], 1, 1, 1) #[bs,2,h,w]
        
        coarse_grid = coarse_grid.permute((0, 2, 3, 1)) 
        out = F.grid_sample(imgs, coarse_grid, mode='bilinear', align_corners=False)

        return out


def findHMatFromCheckerboard(img_proj,img_cam):
    # img_proj = cv2.imread(imname1)
    # cv2.imshow('win',img_proj)
    gray_proj = cv2.cvtColor(img_proj,cv2.COLOR_BGR2GRAY)
    # img_cam = cv2.imread(imname2)

    gray_cam = cv2.cvtColor(img_cam,cv2.COLOR_BGR2GRAY)
    _,thresh_proj = cv2.threshold(gray_proj,127,255,cv2.THRESH_BINARY)
    cv2.imwrite('tp1.png',thresh_proj)

    _,thresh_cam = cv2.threshold(gray_cam,150,255,cv2.THRESH_BINARY)
    cv2.imwrite('tp2.png',thresh_cam)

    ret1, corners_proj = cv2.findChessboardCorners(thresh_proj,(8,7),None)
    ret2, corners_cam = cv2.findChessboardCorners(thresh_cam,(8,7),None)
    cv2.drawChessboardCorners(img_proj,(8,7),corners_proj,ret1)
    cv2.imwrite('tp3.png',img_proj)
    cv2.drawChessboardCorners(img_cam,(8,7),corners_cam,ret2)
    cv2.imwrite('tp4.png',img_cam)
    
    corners_proj = cv2.cornerSubPix(gray_proj,corners_proj, (5,5),(-1,-1),criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30,0.001))
    corners_cam = cv2.cornerSubPix(gray_cam,corners_cam, (5,5),(-1,-1),criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30,0.001))

    Hmatrix, _ = cv2.findHomography(corners_cam,corners_proj,cv2.RANSAC,1.0)
    

    return Hmatrix



def calWarpingField(prj_checkboard,cam_checkboard,im_surf=None, model = 'hmat',cfg = None):
    # bbox = None
    if model == 'hmat':
        warping_field = findHMatFromCheckerboard( prj_checkboard,cam_checkboard)

    else:
        bbox,prj_fov_mask,bbox_max_in,warping_field = calWarpingFieldFromTextureImage(model,cfg,cam_checkboard,prj_checkboard,im_surf)

         
    return bbox, prj_fov_mask,bbox_max_in,warping_field 

def warpImageUsingWF(cam_img, warping_field, prj_fov_mask, bbox=None, prj_size=(256,256),model = 'hmat',isCMP=False):

    if model == 'hmat':
        warped_image =cv2.warpPerspective(cam_img, warping_field,prj_size)
        warped_image_torch = cv2.cvtColor(warped_image,cv2.COLOR_BGR2RGB)
        warped_image_torch = torch.from_numpy(warped_image_torch).to(device).unsqueeze(0).permute((0, 3, 1, 2)).float().div(255)

    else:
        

        warped_image_torch = warpImagesUsingFlow(cam_img, warping_field,prj_fov_mask, bbox,out_size = prj_size,isCMP=isCMP)
        if warped_image_torch.device.type == 'cuda':
            warped_image = warped_image_torch.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        else:
            warped_image = warped_image_torch.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        warped_image = np.uint8(warped_image[0, :, :, ::-1] * 255)  # convert to BGR and uint8 for opencv

    return warped_image , warped_image_torch


def cropCapturedImage(im_cam,bbox,fov_mask):
    
    cam_des_crop = im_cam[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
    torch_resize = Resize([1024,1024])
    cam_des_crop_torch = cv2.cvtColor(cam_des_crop,cv2.COLOR_BGR2RGB)
    cam_des_crop_torch = torch.from_numpy(cam_des_crop_torch).cuda().unsqueeze(0).permute((0, 3, 1, 2)).float().div(255)
    cam_des_crop_torch = torch_resize(cam_des_crop_torch)
    cam_des_crop_torch[~fov_mask] = 0
    return cam_des_crop_torch


def cropCapturedImage2(im_cam,bbox,bbox_max_in):
    cam_des =  np.zeros((1080,1920,3),dtype = np.uint8) 
    cam_des[bbox_max_in[1]:bbox_max_in[1]+bbox_max_in[3],bbox_max_in[0]:bbox_max_in[0]+bbox_max_in[2],:] = \
    im_cam[bbox_max_in[1]:bbox_max_in[1]+bbox_max_in[3],bbox_max_in[0]:bbox_max_in[0]+bbox_max_in[2],:]
    
    cam_des_crop = cam_des[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
    torch_resize = Resize([1024,1024])
    cam_des_crop_torch = cv2.cvtColor(cam_des_crop,cv2.COLOR_BGR2RGB)
    cam_des_crop_torch = torch.from_numpy(cam_des_crop_torch).cuda().unsqueeze(0).permute((0, 3, 1, 2)).float().div(255)
    cam_des_crop_torch = torch_resize(cam_des_crop_torch)
    return cam_des_crop_torch



def cropDesiredImageTorch(im_des,bbox,bbox_max_in):
    torch_resize = Resize([int(bbox_max_in[3]),int(bbox_max_in[2])])
    cam_des_torch = torch.zeros((1,3,1080,1920))
    cam_des_torch[:,:,bbox_max_in[1]:bbox_max_in[1]+bbox_max_in[3],bbox_max_in[0]:bbox_max_in[0]+bbox_max_in[2]] = \
    torch_resize(im_des)
    cam_des_crop = cam_des_torch[:,:,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    torch_resize = Resize([1024,1024])
    cam_des_crop = torch_resize(cam_des_crop)
    cam_des_crop = cam_des_crop.cuda().float()
    return cam_des_crop


def cropDesiredImage(im_cam,bbox,bbox_max_in,fov_mask=None):
    cam_des =  np.zeros((1080,1920,3),dtype = np.uint8) 
    cam_des[bbox_max_in[1]:bbox_max_in[1]+bbox_max_in[3],bbox_max_in[0]:bbox_max_in[0]+bbox_max_in[2],:] = \
    cv2.resize(im_cam,(bbox_max_in[2],bbox_max_in[3]))
    cam_des_crop = cam_des[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
    cam_des_crop = cv2.resize(cam_des_crop,(1024,1024))
    cam_des_crop_torch = cv2.cvtColor(cam_des_crop,cv2.COLOR_BGR2RGB)
    cam_des_crop_torch = torch.from_numpy(cam_des_crop_torch).cuda().unsqueeze(0).permute((0, 3, 1, 2)).float().div(255)
    return cam_des_crop_torch


def UncmpDesiredImage(im_cam):
    cam_des_crop_torch = cv2.cvtColor(im_cam,cv2.COLOR_BGR2RGB)
    cam_des_crop_torch = torch.from_numpy(cam_des_crop_torch).cuda().unsqueeze(0).permute((0, 3, 1, 2)).float().div(255)
    return cam_des_crop_torch


