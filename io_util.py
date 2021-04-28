import numpy as np
import cv2
from multiprocessing import Pool
import os
from os.path import join as opj
def gen_pathlist_fromimgdir(imgdir)->list:
    imgnamelist=os.listdir(imgdir)
    imgpathlist=[opj(imgdir,imgname) for imgname in imgnamelist]
    imgpathlist.sort()
    return imgpathlist
def gen_pathlist_fromimgdirdir(imgdirdir)->list:
    imgpathlist=[]
    imgdirnamelist=os.listdir(imgdirdir)
    imgdirlist=[opj(imgdirdir,imgdirname) for imgdirname in imgdirnamelist]
    imgdirlist.sort()
    for imgdir in imgdirlist:
        imgnamelist=os.listdir(imgdir)
        imgpathlist_=[opj(imgdir,imgname) for imgname in imgnamelist]
        imgpathlist_.sort()
        imgpathlist.extend(imgpathlist_)
    return imgpathlist
def read_any8img(imgpath)->np.ndarray:
    img=cv2.imdecode(np.fromfile(imgpath,dtype=np.uint8),-1)
    return img
def read_raw16img(imgpath)->np.ndarray:
    # return np.uint16
    with open(imgpath, 'rb') as fid:
        data_chunk = fid.read()
    img = np.frombuffer(data_chunk, dtype=np.uint16)
    return img
def read_raw12img(imgpath)->np.ndarray:
    # return np.uint16
    with open(imgpath, 'rb') as fid:
        data_chunk = fid.read()
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    img = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    return img
def read_multiproc(imgdir,proc_count,read_func,save_func,img_count=None):
    '''
    You have to write your own read_func and save_func in the "main" function like below:
    def read_func(i,imgpath):
        img=io_util.read_raw16img(imgpath)
        return [i,img]
    def save_func(i_img):
        global img_data,img_shape
        i=i_img[0]
        img=i_img[1]
        img_data[:,:,i]=img.reshape((*img_shape))
    '''
    pool = Pool(proc_count)
    imglist=os.listdir(imgdir)
    if img_count!=None:
        imgpath_list=[opj(imgdir,img) for img in imglist][0:img_count]
    else:
        imgpath_list=[opj(imgdir,img) for img in imglist]
    for i,imgpath in enumerate(imgpath_list):
        pool.apply_async(read_func, args=(i,imgpath),callback=save_func)
    pool.close()
    pool.join()
def read_multiproc_from_imgpath_list(imgpath_list,proc_count,read_func,save_func):
    '''
    You have to write your own read_func and save_func in the "main" function like below:
    def read_func(i,imgpath):
        img=io_util.read_raw16img(imgpath)
        return [i,img]
    def save_func(i_img):
        global img_data,img_shape
        i=i_img[0]
        img=i_img[1]
        img_data[:,:,i]=img.reshape((*img_shape))
    '''
    pool = Pool(proc_count)
    for i,imgpath in enumerate(imgpath_list):
        pool.apply_async(read_func, args=(i,imgpath),callback=save_func)
    pool.close()
    pool.join()
def save_any8img(img,imgpath):
    """save 8bit img

    Args:
        img : 0-255
        imgpath
    """
    cv2.imencode('.png', img.astype(np.uint8))[1].tofile(imgpath)