
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cv2
import numpy as np


from torchvision.transforms import Resize


# CompenNeSt (journal version) 

'''collected from https://github.com/BingyaoHuang/CompenNeSt-plusplus'''
class CompenNeSt(nn.Module):
    def __init__(self):
        super(CompenNeSt, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()

        self.simplified = False
        self.pool = nn.AdaptiveAvgPool2d(256)

        # siamese encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)

        # output layer
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)

        # skip layers (see s3 in forward)
        # s1
        self.skipConv11 = nn.Conv2d(3, 3, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(3, 3, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(3, 3, 3, 1, 1)

        # s2
        self.skipConv21 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s_pre', None)
        self.register_buffer('res2_s_pre', None)
        self.register_buffer('res3_s_pre', None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    def simplify(self, s):
        s = self.pool(s)
        res1_s = self.relu(self.skipConv11(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        self.res1_s_pre = res1_s

        s = self.relu(self.conv1(s))

        res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        self.res2_s_pre = res2_s

        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        self.res3_s_pre = s

        self.res1_s_pre = self.res1_s_pre.squeeze()
        self.res2_s_pre = self.res2_s_pre.squeeze()
        self.res3_s_pre = self.res3_s_pre.squeeze()

        self.simplified = True

    # x is the input uncompensated image, s is a 1x2sx2s56x2s56 surface image
    def forward(self, x, s):
        # surface feature extraction
        
        s = self.pool(s)
        x = self.pool(x)

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            res1_s = self.relu(self.skipConv11(s))
            res1_s = self.relu(self.skipConv12(res1_s))
            res1_s = self.skipConv13(res1_s)

            s = self.relu(self.conv1(s))

            res2_s = self.skipConv21(s)
            res2_s = self.relu(res2_s)
            res2_s = self.skipConv22(res2_s)

            s = self.relu(self.conv2(s))
            res3_s = self.relu(self.conv3(s))

        res1 = self.relu(self.skipConv11(x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        res1 = res1-res1_s

        x = self.relu(self.conv1(x))

        res2 = self.skipConv21(x)
        res2 = self.relu(res2)
        res2 = self.skipConv22(res2)
        res2 = res2-res2_s

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        x = x -res3_s # s3

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = self.relu(self.transConv1(x) + res2)
        x = self.relu(self.transConv2(x))
        x = self.relu(self.conv6(x) + res1)

        x = torch.clamp(x, min=0)

        x = torch.clamp(x, max=1)


        return x

        # return x, prj_valid_cmp_large



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



'''geometric correction using the estimated displacement field'''

class FlowWarpingNet(nn.Module):
    def __init__(self,Output_Size=(256,256), With_Refine=False):
        super(FlowWarpingNet, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.flow = Flow
        self.img_size = Output_Size
        
        self.fine_grid = None
        xx = torch.arange(0,self.img_size[0]).view(1,-1).repeat(self.img_size[1],1)
        yy = torch.arange(0,self.img_size[1]).view(-1,1).repeat(1,self.img_size[0])
        xx = xx.view(1,self.img_size[0],self.img_size[1])
        yy = yy.view(1,self.img_size[0],self.img_size[1])
        self.grid = torch.cat((xx,yy),0).float().to(device) # [1,2,h,w]

        self.scale_x = 0.25
        self.scale_y = 0.25
        self.fine_grid = None
        self.flow = None


    def setFlow(self,flow):
        self.flow = flow
        up_flow = flow.cpu().numpy()
        u_ = cv2.resize(up_flow[0,0,:,:],(self.img_size[0],self.img_size[1]))*self.scale_x
        v_ = cv2.resize(up_flow[0,1,:,:],(self.img_size[0],self.img_size[1]))*self.scale_y
        up_flow = np.dstack((u_,v_))
        up_flow = torch.from_numpy(up_flow).unsqueeze(0).cuda()

        coarse_grid = self.grid.unsqueeze(0).permute((0, 2, 3, 1)) + up_flow
        
        self.fine_grid = 2.0*coarse_grid/(self.img_size[1]-1)-1.0  # [-1,1]

        
    def forward(self, x):

        x = F.grid_sample(x, self.fine_grid, mode='bilinear', align_corners=False)

        return x



'''our base compensation model'''

class CompenNeStFF(nn.Module):
    def __init__(self,warping_net=None, compen_net=None):
        super(CompenNeStFF, self).__init__()
        self.name = self.__class__.__name__

        # initialize from existing models or create new models
        self.warping_net = copy.deepcopy(warping_net.module) if warping_net is not None else FlowWarpingNet()
        self.compen_nest = copy.deepcopy(compen_net.module) if compen_net is not None else CompenNeSt()
        
        self.s = None
        self.flow = None
        self.sw = None

    def setFlow(self, s,flow):

        self.s = s
        self.flow = flow
        self.warping_net.setFlow(flow)
        self.sw = self.warping_net(s)
    def getSurf(self):
        return self.sw
    # s is Bx3x256x256 surface image
    def forward(self, x, upsampling=False ):
        # geometric correction using WarpingNet (both x and s)
        xw = self.warping_net(x)
        # photometric compensation using CompenNet
        out = self.compen_nest(xw, self.sw)

        if upsampling:
            prj_valid_cmp_large = F.interpolate(out,size=(1024,1024),mode='bicubic')
            prj_valid_cmp_large = torch.clamp(prj_valid_cmp_large,min=0)
            prj_valid_cmp_large = torch.clamp(prj_valid_cmp_large,max=1)
            return prj_valid_cmp_large
        else:
            return out



               

