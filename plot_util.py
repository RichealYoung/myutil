import numpy as np
import matplotlib.pylab as plt
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from os.path import join as opj
from os.path import dirname as opd
from tqdm import tqdm
def plot(img,title="",savename="",savedir=None):
    plt.figure()
    plt.title(title)
    plt.imshow(img,vmax=img.max(),vmin=0)
    if savedir!=None:
        plt.savefig(opj(savedir,savename+'.png'),dpi=200)
    else:
        plt.show()
    plt.close()
def plot12(img1,img2,title1="",title2="",title="",savename="",savedir=None):
    plt.figure()
    plt.title(title)
    plt.subplot(121)
    plt.title(title1)
    plt.imshow(img1,vmax=img1.max(),vmin=0)
    plt.subplot(122)
    plt.title(title2)
    plt.imshow(img2,vmax=img2.max(),vmin=0)
    if savedir!=None:
        plt.savefig(opj(savedir,savename+'.png'),dpi=200)
    else:
        plt.show()
    plt.close()

