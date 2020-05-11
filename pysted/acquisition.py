
import numpy


def rescue(signal, period, photon_thresholds, time_thresholds):
    '''Return the signal obtained using RESCue with the given parameters and
    the associated map of pixel dwell times.
    '''
    # counts per second
    cps = signal / period
    # empty matrices to store results
    rescued_signal = numpy.empty(signal.shape, dtype=numpy.int64)
    pdt = numpy.empty(signal.shape)
    
    is_valid = numpy.full(signal.shape, True, dtype=numpy.bool)
    # apply lower thresholds
    for pth, tth in zip(photon_thresholds, time_thresholds):
        in_th = numpy.logical_and(cps * tth < pth, is_valid)
        pdt[in_th] = tth
        rescued_signal[in_th] = cps[in_th] * tth
        is_valid[in_th] = False
    # apply upper threshold
    pth = photon_thresholds[-1]
    pdt[is_valid] = pth / cps[is_valid]
    rescued_signal[is_valid] = pth
    return rescued_signal, pdt
