import numpy as np
from multiprocessing import Pool
import dcor
def cal_R2(y,yhat):
    #https://en.wikipedia.org/wiki/Coefficient_of_determination
    SStot=np.sum((y-y.mean(0))**2,0)
    SSres=np.sum((yhat-y)**2,0)
    R2=1-SSres/SStot
    return R2
def cal_dCor(x,y):
    #https://github.com/vnmabus/dcor
    return dcor.distance_correlation(x,y, method='AVL')
def cal_list_multiproc(x,y_list,proc_count,cal_func,save_func):
    pool = Pool(proc_count)
    for i,y in enumerate(y_list):
        pool.apply_async(cal_func, args=(i,x,y),callback=save_func)
    # H,W,_=img_data.shape
    # x0 = img_data[loc_h,loc_w,:]
    # for i in range(H):
    #     for j in range(W):
    #         x1 = img_data[i,j,:]
    #         pool.apply_async(cal_dCor_, args=(x0,x1,i,j),callback=save_func)
    pool.close()
    pool.join()