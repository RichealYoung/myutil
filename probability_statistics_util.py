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
def cal_KL(sample1,sample2):
    return 0 #TODO
def cal_list_multiproc(x,y_list,proc_count,cal_func,save_func):
    '''
    You have to write your own cal_func and save_func in the "main" function like below:
    def cal_func(i,x,y):
        value=probability_statistics_util.cal_dCor(x,y)
        return [i,value]
    def save_func(i_value):
        global value_list
        value_list[i_value[0]]]=i_value[1]
    '''
    pool = Pool(proc_count)
    for i,y in enumerate(y_list):
        pool.apply_async(cal_func, args=(i,x,y),callback=save_func)
    pool.close()
    pool.join()
def cal_matrix_multiproc(x,y_matrix,proc_count,cal_func,save_func):
    '''
    You have to write your own cal_func and save_func in the "main" function like below:
    def cal_func(i,j,x,y):
        value=probability_statistics_util.cal_dCor(x,y)
        return [i,j,value]
    def save_func(i_j_value):
        global value_matrix
        value_matrix[i_j_value[0],i_j_value[1]]]=i_j_value[2]
    '''
    pool = Pool(proc_count)
    _,H,W=y_matrix.shape
    for i in range(H):
        for j in range(W):
            y=y_matrix[:,i,j]
            pool.apply_async(cal_func, args=(i,j,x,y),callback=save_func)
    pool.close()
    pool.join()