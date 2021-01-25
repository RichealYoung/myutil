import numpy as np
from myutil.probability_statistics_util import cal_R2

def calibrate_alpha_beta(t,X):
    '''
    Use Least Square method to applying "point estimate" on alpha and beta,
    where X[k]=alpha*L*t[k]+beta
    '''
    cal_t_count=t.shape[0]
    _,H,W=X.shape
    t=t.reshape(-1,1)
    A=np.concatenate([t,np.ones((cal_t_count,1))],1)
    ls_estimation=np.linalg.inv(A.T@A)@A.T@X.reshape(cal_t_count,-1)
    ls_estimation=ls_estimation.reshape(2,H,W)
    alpha_L=ls_estimation[0]
    beta=ls_estimation[1]
    # calculate the Coefficient_of_determination
    X_hat=alpha_L.reshape(1,H,W)*t.reshape(cal_t_count,1,1)+beta.reshape(1,H,W)
    R2=cal_R2(X,X_hat)
    # use the one's alpha_L whose R2 is highest to assign alpha, 
    alpha=alpha_L[R2==R2.max()]
    # replace the one's beta whose R2 is too low(such as lower than 0.8) by the neighboor beta
    pass
    return alpha,beta,R2

