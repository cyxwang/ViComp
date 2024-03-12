
import numpy as np
import cv2


class Projecting():
    def __init__(self):
        super().__init__()
        self.prj_win = 'full_screen'
        self.sw,self.sh =  1920,1080 #  screen.width,screen.height
        cv2.namedWindow(self.prj_win,cv2.WINDOW_NORMAL)  # 
        cv2.moveWindow(self.prj_win,1921,0)
        cv2.setWindowProperty(self.prj_win,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        self.img = np.zeros((self.sh,self.sw,3),dtype = np.uint8)
    
    def forward(self, img):
        self.projecting2(img)



    def projecting(self,cmp_img,marker):

        h,w,c = cmp_img.shape
        if h<1024:
            cmp_img =  cv2.resize(cmp_img,(1024,1024))
        h,w,c = cmp_img.shape
        marker = cv2.resize(marker,(300,300))
        self.img[:1024,1600-1024:1600,:3] = cmp_img[:,:,:]
        self.img[5:305,5:305,2] = marker[:,:,2]

        cv2.imshow(self.prj_win,self.img)
        cv2.waitKey(10)
        

    def projecting2(self,cmp_img):
        cv2.imshow(self.prj_win,cmp_img)
        cv2.waitKey(10)

    def capturing(self,filename):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        cv2.imwrite(filename, frame)
    
        return frame

class Capturing():
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        
    def forward(self, filename):
        
        self.capturing(filename)

    def capturing(self,filename):
        
        ret, frame = self.cap.read()
        cv2.imwrite(filename, frame)
        return frame
    
    
    def release(self):
        self.cap.release()
        return
    
    
        
    