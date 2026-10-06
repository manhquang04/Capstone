"""Independent array metric reload, no benchmark runner or private decoder state."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity


def images(truth,reconstructed):
    truth,reconstructed=np.asarray(truth),np.asarray(reconstructed)
    if truth.shape!=reconstructed.shape or truth.ndim!=4 or not np.isfinite(truth).all() or not np.isfinite(reconstructed).all():
        raise ValueError('invalid independent image arrays')
    costs=np.mean((truth[:,None]-reconstructed[None,:])**2,axis=(2,3,4))
    _,columns=linear_sum_assignment(costs)
    records=[]
    for source,index in enumerate(columns):
        a=np.clip(truth[source],0,1).transpose(1,2,0)
        b=np.clip(reconstructed[index],0,1).transpose(1,2,0)
        mse=float(np.mean((a-b)**2))
        if mse<=0:
            raise ValueError('nonfinite PSNR: zero metric MSE')
        records.append(dict(mse=mse,psnr_db=float(-10*np.log10(mse)),
                            ssim=float(structural_similarity(a,b,data_range=1.,channel_axis=2))))
    means={key:float(np.mean([r[key] for r in records])) for key in records[0]}
    if not all(math.isfinite(value) for value in means.values()):
        raise ValueError('nonfinite independent image metrics')
    return means,records,columns.tolist()


def standardized_mse(guess,truth,std):
    guess,truth,std=[np.asarray(value,dtype=np.float64) for value in (guess,truth,std)]
    if not all(np.isfinite(value).all() for value in (guess,truth,std)) or np.any(std<=0):
        raise ValueError('invalid BN metric inputs')
    return float(np.mean(((guess-truth)/std)**2))


def gaussian_epsilon(sigma,ratio=1.,delta=1e-5):
    if not math.isfinite(sigma) or sigma<=0 or ratio<=0 or not 0<delta<1:
        raise ValueError('invalid Gaussian accounting input')
    rho=ratio**2/(2*sigma**2)
    return rho+2*math.sqrt(rho*math.log(1/delta))
