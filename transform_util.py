import cv2
import numpy as np
import random
import torch
from einops import rearrange
class Transform:
    def __init__(self,operator_list,args_list):
        self.operator_instance_list=[]
        assert len(operator_list)==len(args_list)
        for operator,args in zip(operator_list,args_list):
            operator_instance=operator(**args)
            if operator_instance.check():
                self.operator_instance_list.append(operator_instance)
    def __call__(self,img_list):
        for operator_instance in self.operator_instance_list:
            img_list = operator_instance(img_list)
        return img_list
class Crop:
    def __init__(self,crop_h,crop_w,crop_h_start,crop_w_start,mod_by=0):
        self.crop_h=crop_h
        self.crop_w=crop_w
        self.crop_h_start=crop_h_start
        self.crop_w_start=crop_w_start
        self.mod_by=mod_by
    def check(self):
        if (self.crop_h and self.crop_w) or self.mod_by:
            return True
        else:
            print('not crop')
            return False
    def __call__(self,img_list,):
        """ crop every imgs in img_list

        Args:
            img_list (list): every imgs in img_list must have shape=[H,W,C]

        Returns:
            list: a list containing all imgs which are cropped
        """
        img_list_cropped=[]
        H, W, C = img_list[0].shape
        if self.crop_h and self.crop_w:
            crop_h = min(self.crop_h, H - self.crop_h_start)
            crop_w = min(self.crop_w, W - self.crop_w_start)  
            for img in img_list:
                if self.mod_by:
                    crop_h = (crop_h // self.mod_by) * self.mod_by
                    crop_w = (crop_w // self.mod_by) * self.mod_by
                img_list_cropped.append(img[self.crop_h_start:self.crop_h_start + crop_h,
                                    self.crop_w_start:self.crop_w_start + crop_w, :])
        elif self.mod_by:
            for img in img_list:
                if self.mod_by:
                    crop_h = (H // self.mod_by) * self.mod_by
                    crop_w = (W // self.mod_by) * self.mod_by
                img_list_cropped.append(img[0:0 + crop_h,
                                    0:0 + crop_w, :])
        else:
            img_list_cropped = img_list
        return img_list_cropped
class RandomCrop:
    def __init__(self,randomcrop):
        self.randomcrop=randomcrop
    def check(self):
        if self.randomcrop:
            return True
        else:
            print('not randomcrop')
            return False
    def __call__(self,img_list,):
        """ randomcrop every imgs in img_list

        Args:
            img_list (list): every imgs in img_list must have shape=[H,W,C]

        Returns:
            list: a list containing all imgs which are randomcropped
        """
        H, W, C = img_list[0].shape
        randomcrop_h_start = random.randint(0, max(0, H - self.randomcrop))
        randomcrop_w_start = random.randint(0, max(0, W - self.randomcrop))
        randomcrop=Crop(self.randomcrop,self.randomcrop,randomcrop_h_start,randomcrop_w_start)
        return randomcrop(img_list)
class Resize:
    def __init__(self,resize_h,resize_w):
        self.resize_h=resize_h
        self.resize_w=resize_w
    def check(self):
        if self.resize_h and self.resize_w:
            return True
        else:
            print('not resize')
            return False
    def __call__(self,img_list,):
        """ resize every imgs in img_list

        Args:
            img_list (list):

        Returns:
            list: a list containing all imgs which are resized
        """
        
        img_list_resized = [cv2.resize(img, dsize=(self.resize_w, self.resize_h)) for img in img_list]
        # if an img's shape=[H,W,1], then cropped img's shape will be [H,W]
        img_list_resized = [img[...,np.newaxis] for img in img_list_resized if len(img.shape)==2]
        return img_list_resized

class AtleastResize:
    def __init__(self,atleast_h,atleast_w):
        self.atleast_h=atleast_h
        self.atleast_w=atleast_w
    def check(self):
        if self.atleast_h and self.atleast_w:
            return True
        else:
            print('not atleastresize')
            return False
    def __call__(self,img_list,):
        """ resize every imgs in img_list if img_h < atleast_h or img_w < atleast_w

        Args:
            img_list (list):

        Returns:
            list: a list containing all imgs which are resized or not
        """
        def atleastresize(img):
            H,W,_=img.shape
            if H>=self.atleast_h and W>=self.atleast_w:
                return img
            elif H>=self.atleast_h and W<self.atleast_w:
                return cv2.resize(img, dsize=(self.atleast_w,H))
            elif H<self.atleast_h and W>=self.atleast_w:
                return cv2.resize(img, dsize=(W,self.atleast_h))
        img_list_resized = list(map(atleastresize,img_list))
        # if an img's shape=[H,W,1], then cropped img's shape will be [H,W]
        def check(img):
            if len(img.shape)==2:
                return img[...,np.newaxis]
            else:
                return img
        img_list_resized = list(map(check,img_list_resized))
        return img_list_resized
class FlipRoat:
    def __init__(self,fliprot):
        self.fliprot=fliprot
    def check(self):
        if self.fliprot:
            return True
        else:
            print('not fliproat')
            return False
    def __call__(self,img_list,):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5
        return [fliproat_img(img, hflip,vflip,rot90) for img in img_list]
def fliproat_img(img, hflip,vflip, rot90):
    if hflip:
        img = img[:, ::-1, :]
    if vflip:
        img = img[::-1, :, :]
    if rot90:
        img = img.transpose(1, 0, 2)
    return img
class Normalize:
    def __init__(self,mean,std,range):
        self.mean=mean
        self.std=std
        self.range=range
    def check(self):
        if self.mean==0 and self.std==1 and self.range==1:
            print('not normalize')
            return False
        else:
            return True
    def __call__(self,img_list,):
        return [(img.astype(np.float64)/self.range-self.mean)/self.std for img in img_list]
class ToTensor:
    def __init__(self,layout=None):
        self.layout=layout
    def check(self):
        return True
    def __call__(self,img_list,):
        if self.layout == None:
            return [torch.from_numpy(img).type(torch.FloatTensor) for img in img_list]
        else:
            return [torch.from_numpy(rearrange(img,self.layout)).type(torch.FloatTensor) for img in img_list]
class ListToTensor:
    def __init__(self,layout=None):
        self.layout=layout
    def check(self):
        return True
    def __call__(self,img_list,):
        if self.layout == None:
            return torch.cat([torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0) for img in img_list])
        else:
            return rearrange(torch.cat([torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0) for img in img_list]),self.layout)

def crop2batch(img_list,cropsize=3):
    imgbatch_list=[]
    pad_size=cropsize//2
    for img in img_list:
        H,W,_=img.shape
        img_pad=np.pad(img,[[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        for h in range(pad_size,H+pad_size):
            for w in range(pad_size,W+pad_size):
                imgcrop=img_pad[h-pad_size:h-pad_size+cropsize,w-pad_size:w-pad_size+cropsize,:]
                imgcrop=imgcrop.flatten()
                imgbatch_list.append(imgcrop)
    return imgbatch_list
