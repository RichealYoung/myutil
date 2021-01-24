import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
def cal_quality(img_gt,img_hat):
    img_hat=img_hat*img_gt.mean()/img_hat.mean()
    img_hat_psnr=psnr(img_gt,img_hat,data_range=img_gt.max())
    img_hat_ssim = ssim(img_gt,img_hat,data_range=img_gt.max())
    return img_hat_psnr,img_hat_ssim
def cal_R2(y,yhat):
    #https://en.wikipedia.org/wiki/Coefficient_of_determination
    SStot=np.sum((y-y.mean(0))**2,0)
    SSres=np.sum((yhat-y)**2,0)
    R2=1-SSres/SStot
    return R2