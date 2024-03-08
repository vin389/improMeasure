import numpy as np

def xcorr(x1, x2):
    """
    Cross-correlation function estimates.
    """
    corr = np.correlate(x1, x2, mode='full')
    lags = np.arange(1 - x1.size, x2.size)
    return corr, lags

