import numpy as np
import matplotlib.pylab as plt
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from os.path import join as opj
from os.path import dirname as opd
from tqdm import tqdm
def set_ap(scene,ap):
    scene=scene.astype(np.float64)
    H,W=scene.shape
    photon_count=ap*H*W
    grey_count=scene.sum()
    photon_per_grey=photon_count/grey_count
    scene_ptn=scene*photon_per_grey
    return scene_ptn
def ptn2u_mag(ptn,alpha):#TODO 未从理论上验证准确性
    # ptn \propto u2, assume ptn=alpha*u2
    # u2 =u_mag**2
    u2=ptn/alpha
    u_mag=np.sqrt(u2)
    return u_mag
def u2ptn(u,alpha):#TODO 未从理论上验证准确性
    # ptn \propto u2, assume ptn=alpha*u2
    # u2 =u_mag**2
    u_mag=np.abs(u)
    u2=u_mag**2
    ptn=u2*alpha
    return ptn
def op2fp(op_u):
    # simulate the fourier plane's complex amplitude from the object plane's complex amplitude
    fp_u=np.fft.fftshift(np.fft.fft2(op_u))
    cor_factor=1/np.sqrt(fp_u.size)
    # energy consistent
    fp_u=fp_u*cor_factor
    return fp_u

