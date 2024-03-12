
import numpy as np
import cv2
from compensation.utils import saveImgs,computeMetrics,convertImgs
import threading
import queue
import torch.nn.functional as F
from os.path import join as fullfile
from time import sleep 
import torch
from torchvision.transforms import Resize

from compensation.warping import cropCapturedImage, cropDesiredImage,UncmpDesiredImage
class CMPScheduler():
    def __init__(self,img_path,save_path,arch_name,data_name, prj_fov_mask, bbox, bbox_max_in, short_model,short_opt,long_model,long_opt,cam_ref_warped,cam_size =(480,270),prj_size=(256,256), device='cuda',uncmp = False,supdate = True):
        self._lock = threading.RLock()

        self.cam_size = cam_size
        self.prj_size = prj_size

        self.bbox = bbox
        self.bbox_max_in = bbox_max_in
        self.prj_fov_mask = prj_fov_mask

        self.cap = cv2.VideoCapture(0)

        self.isprojected = False

        self.cid = 0

        self.img_path = img_path
        self.data_name = data_name
        self.arch_name = arch_name

        self.long_model = long_model
        self.short_model = short_model
        self.short_opt = short_opt
        self.long_opt = long_opt
        self.device = device

        self.uncmp = uncmp
        self.supdate = supdate
    
        self.isreset = True
        self.img_name = None

        self._long_model_bq= queue.Queue(maxsize=1)
        self._long_model_bq.put(True)
        self._eval_model_bq= queue.Queue(maxsize=1)
        self._cmp_bq = queue.Queue(maxsize=1)
        self._cam_bq = queue.Queue(maxsize=1)
        self._prj_bq = queue.Queue(maxsize=1)
        self._mid_bq = queue.Queue(maxsize=1)
        self._prj_img_bq = queue.Queue(maxsize=100)
        self._short_data_bq =queue.Queue(maxsize=1)
        self._long_data_bq =queue.Queue(maxsize=1)

        self._shot_id_bq= queue.Queue(maxsize=100)
        self._shot_id_bq.put(0)
        
        self.prj_win = 'full_screen'
        self.sw,self.sh =  1920, 1080 # screen.width,screen.height
        cv2.namedWindow(self.prj_win,cv2.WINDOW_NORMAL)  # 

        cv2.moveWindow(self.prj_win,1921,0)
        cv2.setWindowProperty(self.prj_win,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    
        self.torch_resize = Resize([256,256])
        self.torch_resize_hr = Resize([1024,1024])

        self.s = self.torch_resize(cam_ref_warped) # None   # surface
        self.prj_img = np.zeros((self.sh,self.sw,3),dtype = np.uint8)

        self.marker_id_last = -1
        
        self.params = cv2.aruco.DetectorParameters_create()
        self.arucoDict = cv2.aruco.Dictionary_get( cv2.aruco.DICT_4X4_100)

        self.des_imgs = torch.Tensor(100,3,prj_size[0],prj_size[1]).cuda()
        self.prj_imgs = torch.Tensor(100,3,prj_size[0],prj_size[1]).cuda()
        self.epoch_ids = -1*np.ones(100,dtype = np.int64)
 
        self.cmp_img = None

        self.shot_id = 0
        self.img_id = -1

        self.shot_model_update_flag = False
        self.long_update_flag = False
 

    def projecting2(self):
        
        cmp_img = self._prj_img_bq.get()
        cv2.imshow(self.prj_win,cmp_img)

        cv2.waitKey(1)

    def short_memory_update(self):
        train_data = self._short_data_bq.get()
        self._lock.acquire()
        shot_id = self.shot_id
        self._lock.release()
        if (train_data['pid']==shot_id-1 or train_data['pid']==shot_id) :  
             self.shot_model_update_flag =False 
             return 
        else:
            self._lock.acquire()

            if self.supdate:
                self.short_model(train_data['cam_train'],train_data['prj_train'],train_data['des_train'])

            self._lock.release()

        # self.shot_model_update_flag =False
        return 

    def short_memory_reset(self,shot_id):
        
        self._lock.acquire()

        self.short_model.reset(self.long_model.getTeacherModel(),self.short_opt,shot_id)
        self.shot_model_update_flag = True

        self._lock.release()

        return 
    
    def get_capturing_all_count(self):
        return self.cap_all_count+1
    
    def get_image_count(self):
        return self.img_id+1
    def get_capturing_count(self):
        return self.cid
    
    def long_memory_set_status(self,shot_id):
        self._lock.acquire()

        self._shot_id_bq.put(shot_id)

        self._lock.release()

        return 
    def long_memory_update(self):
        train_data = self._long_data_bq.get()
        self.shot_id = self._shot_id_bq.get() 
        
        self._lock.acquire()

        self.long_model(train_data['cam_train'],train_data['prj_train'],train_data['des_train'])
        self.long_update_flag = False

        self._lock.release()
        return 

    def get_short_model(self):
        return self.short_model,self.short_opt
    def get_long_model(self):
        return self.long_model,self.long_opt

    def get_cmp_status(self):
        self._lock.acquire()
        cmp_img = self._cmp_bq.get()
        self._lock.release()
        return cmp_img

    def set_cam_status(self, s):
        self._lock.acquire()
        self._cam_bq.put(s)
        self._lock.release()
        return

    def get_mid_status(self):
        self._lock.acquire()
        mid = self._mid_bq.get()
        self._lock.release()
        return mid


    def get_prj_status(self):
        self._lock.acquire()
        isprojected = self._prj_bq.get()
        self._lock.release()
        return isprojected

    def set_prj_status(self, s):
        self._lock.acquire()
        self._prj_bq.put(s)
        self._lock.release()
        return


    def compensation(self,img_id):
        img_name = '{:0>8d}.png'.format(img_id)
        cam_valid_des = cv2.imread(fullfile(self.img_path,self.data_name, img_name))
        cam_valid_des_crop = cropDesiredImage(cam_valid_des,self.bbox,self.bbox_max_in)
        self._lock.acquire()

        prj_valid_cmp_large =  self.short_model.compensation(cam_valid_des_crop)   # cam_valid_des_crop #

        self._lock.release()
        
        self.img_id = img_id
        
        prj_valid_cmp_n = convertImgs(prj_valid_cmp_large)
        prj_data =  np.zeros((self.sh,self.sw,3),dtype = np.uint8)
        # prj_data[:1024,:1024,:3] = prj_valid_cmp_n[:1024,:1024,:]
        prj_data[:1024,1600-1024:1600,:3] = prj_valid_cmp_n[:1024,:1024,:]

        if img_id%2==0:
            marker_id = img_id%100
            
            self._lock.acquire()

            self.prj_imgs[marker_id,:,:,:] = prj_valid_cmp_large
            self.des_imgs[marker_id,:,:,:] = cam_valid_des_crop

            self.epoch_ids[marker_id] = img_id 
            

            marker_name = fullfile(self.img_path, 'markers4_4_100', '{}.png'.format(marker_id))

            marker = cv2.imread(marker_name)
            marker = cv2.resize(marker,[300,300])
            prj_data[200:500,5:305,2] = marker[:,:,2]

    
            self._lock.release()
            
        self._lock.acquire()
        self._prj_img_bq.put(prj_data)
        self._lock.release()
        return

    def capture2detect(self):
        self._lock.acquire()

        cid = self.cid
        ret, frame = self.cap.read()
        corners,marker_id,reject = cv2.aruco.detectMarkers(frame[:,:,2],self.arucoDict,parameters = self.params)
        self.cid = self.cid+1
        cam_valid_cap_crop_torch = cropCapturedImage(frame,self.bbox,self.prj_fov_mask)

        self._lock.release()
        if marker_id is not None :
            if self.epoch_ids[marker_id[0][0]]<=self.marker_id_last:
                return
            else:
                self.marker_id_last = self.epoch_ids[marker_id[0][0]]
            self._lock.acquire()

            train_data = dict(cid = cid, pid =self.epoch_ids[marker_id[0][0]], cam_train=cam_valid_cap_crop_torch,
                          prj_train=self.prj_imgs[marker_id[0][0],:,:,:].unsqueeze(0),des_train=self.des_imgs[marker_id[0][0],:,:,:].unsqueeze(0))
            self._short_data_bq.put(train_data)

            if (self.arch_name == 'Online') and (self.shot_id <self.epoch_ids[marker_id[0][0]]) and (self.long_update_flag is False) : 
                
                self.long_update_flag = True
                self._long_data_bq.put(train_data)

            self._lock.release()
  
        return 
    
    
    def release(self):
        self.cap.release()
        return




