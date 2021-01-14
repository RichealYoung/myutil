import numpy as np
import cv2
def readimg(imgpath):
    img=cv2.imdecode(np.fromfile(imgpath,dtype=np.uint8),-1)
    return img