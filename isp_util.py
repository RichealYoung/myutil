import numpy as np

def emccd_isp(img_ptn,EMgain=300,addnoise=True,damp=False,clip=True):
    #### 1. config
    QE = 0.9
    readout = 139
    c = 0.00145 # spurious charge
    e_per_edu = 0.1866
    baseline = 500
    EMgain = 300
    ptn_max=((65535-baseline)/(e_per_edu*EMgain)-c)/QE
    damp_ratio=np.sort(img_ptn.flatten())[-2]/ptn_max*1.1
    #TODO 确保只有零频分量可以饱和，其余值必须被准确测量，即不能饱和，但是由于受噪声影响，也不能完全保证
    if damp:
        img_ptn=img_ptn/damp_ratio
    #### 2. add noise (related to ptn)
    if addnoise:
        img_ptn_electric = QE * img_ptn + c
        img_ptn_electric[img_ptn_electric<1e-6] = 1e-6
        n_ie = np.random.poisson(lam=img_ptn_electric, size=img_ptn.shape)
        n_oe = np.random.gamma(shape=n_ie, scale=EMgain, size=img_ptn.shape)
        #### 3. add noise (readout)
        n_oe = n_oe + np.random.normal(loc=0, scale=readout, size=img_ptn.shape)
    else:
        n_oe=(QE * img_ptn + c)*EMgain
    #### 4. ptn --> value
    ADU_out =  np.floor(np.dot(n_oe, e_per_edu) + baseline)
    ADU_out =  np.dot(n_oe, e_per_edu) + baseline
    #### 5. clip
    # print(ADU_out[ADU_out>65535])
    if clip:
        ADU_out=ADU_out.clip(0,65535)
    return ADU_out

def inv_emccd_isp(ADU_out,error_baseline=0):#TODO 与photon保持相同的比值即可，因此偏置一定要准确知道
    baseline = 500
    baseline=(1-error_baseline)*baseline
    img_ptn=ADU_out-baseline
    img_ptn=img_ptn.clip(0,np.inf)
    return img_ptn