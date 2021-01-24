import numpy as np
import cv2
from multiprocessing import Pool
import os
from os.path import join as opj
def read_any8img(imgpath)->np.ndarray:
    img=cv2.imdecode(np.fromfile(imgpath,dtype=np.uint8),-1)
    return img
def read_raw16img(i,imgpath)->np.ndarray:
    # return np.uint16
    with open(imgpath, 'rb') as fid:
        data_chunk = fid.read()
    data = np.frombuffer(data_chunk, dtype=np.uint16)
    return [i,data]
def read_raw12img(i,imgpath)->np.ndarray:
    # return np.uint16
    with open(imgpath, 'rb') as fid:
        data_chunk = fid.read()
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    return [i,np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])]
def read_multiproc(imgdir,thread_count,img_count,read_func,save_func):
    '''
    You have to write your own save_func in the "main" function like below:
    def save_img(i_img):
        global img_data,img_shape
        i=i_img[0]
        img=i_img[1]
        img_data[:,:,i]=img.reshape((*img_shape))
    '''
    pool = Pool(thread_count)
    imglist=os.listdir(imgdir)
    imgpath_list=[opj(imgdir,img) for img in imglist][0:img_count]
    for i,imgpath in enumerate(imgpath_list):
        pool.apply_async(read_func, args=(i,imgpath),callback=save_func)
    pool.close()
    pool.join()
