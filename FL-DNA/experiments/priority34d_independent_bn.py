"""Independent public-checkpoint BN decoder; no truth/raw/noise arguments.

Dense SciPy Hadamard replaces the experiment's FWHT implementation. This module
does not load benchmark targets or write earlier evidence. Full result audit
integration is a separate required step, not established by synthetic tests.
"""
import numpy as np
from scipy.linalg import hadamard


def seed_mix(*values):
    state=0x9E3779B9
    for value in values:
        state^=int(value)+0x9E3779B9+((state<<6)&0xFFFFFFFF)+(state>>2)
        state&=0xFFFFFFFF
    return state


def projection(metadata):
    n,k,size=metadata.padded_size,metadata.sketch_size,metadata.original_size
    sampled=np.asarray(metadata.sampled_indices,dtype=np.int64)
    if n<1 or n&(n-1) or not 0<size<=n or not 0<k<=n:
        raise ValueError('invalid projection dimensions')
    if sampled.shape!=(k,) or len(set(sampled.tolist()))!=k or np.any((sampled<0)|(sampled>=n)):
        raise ValueError('invalid public sampling metadata')
    signs=np.random.default_rng(seed_mix(metadata.seed,17)).choice([-1.,1.],size=n,replace=True)
    normalized=hadamard(n).astype(np.float64)/np.sqrt(n)
    return np.sqrt(n/k)*normalized[sampled,:size]*signs[:size]


def decode(weight,bias,running_mean,momentum,received):
    if set(received)-{'kind','q','metadata'}:
        raise ValueError('protected decoder accepts transmitted fields only')
    w,b,mean=[np.asarray(value,dtype=np.float64) for value in (weight,bias,running_mean)]
    q=np.asarray(received['q'],dtype=np.float64).reshape(-1)
    if not all(np.isfinite(value).all() for value in (w,b,mean,q)) or not np.isfinite(momentum) or momentum<=0:
        raise ValueError('nonfinite/invalid public decoder input')
    if w.ndim!=2 or b.shape!=(w.shape[0],) or mean.shape!=b.shape:
        raise ValueError('incompatible public first-linear/BN dimensions')
    offset=momentum*(b-mean)
    kind=received['kind']
    if kind in ('raw','v1'):
        if q.shape!=b.shape:
            raise ValueError('invalid received BN delta dimension')
        # Equivalent linear system to original torch least-squares, independent
        # NumPy solver and equation arrangement.
        result=np.linalg.lstsq(w,mean+q/momentum-b,rcond=None)[0]
    elif kind=='v2':
        matrix=projection(received['metadata'])
        if matrix.shape[1]!=w.shape[0] or q.shape!=(matrix.shape[0],):
            raise ValueError('incompatible received sketch')
        result=np.linalg.lstsq(matrix@(momentum*w),q-matrix@offset,rcond=None)[0]
    else:
        raise ValueError('unknown received mechanism; no fallback')
    if not np.isfinite(result).all():
        raise ValueError('nonfinite independent recovery')
    return result


def defender_decode_for_distortion(received):
    """Only audit original defender distortion definition, NOT attack recovery."""
    q=np.asarray(received['q'],dtype=np.float64)
    if received['kind']=='v1':
        return q.astype(np.float32)
    if received['kind']!='v2':
        raise ValueError('unknown distortion transform')
    metadata=received['metadata']
    return (projection(metadata).T@q).reshape(metadata.original_shape).astype(np.float32)
