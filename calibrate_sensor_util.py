import numpy as np
from quality_util import cal_R2
def calibrate_alpha_beta(t,X):
    '''
    Use Least Square method to applying "point estimate" on alpha and beta,
    where X[k]=alpha*L*t[k]+beta
    '''
    cal_t_count=t.shape
    _,H,W=X.shape
    t=t.reshape(-1,1)
    A=np.concatenate([t,np.ones((cal_t_count,1))],1)
    ls_estimation=np.linalg.inv(A.T@A)@A.T@X.reshape(cal_t_count,-1)
    ls_estimation=ls_estimation.reshape(2,H,W)
    alpha_L=ls_estimation[0]
    beta=ls_estimation[1]
    alpha=alpha_L.max()
    # calculate the Coefficient_of_determination
    X_hat=alpha_L.reshape(1,H,W)*t.reshape(cal_t_count,1,1)+beta.reshape(1,H,W)
    R2=cal_R2(X,X_hat)
    return alpha,beta,R2

