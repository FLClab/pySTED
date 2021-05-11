
import time
import numpy
from matplotlib import pyplot as plt
import bleach_funcs
cimport numpy
cimport cython

from libc.math cimport exp
from libc.stdlib cimport rand, srand, RAND_MAX

INTDTYPE = numpy.int32
FLOATDTYPE = numpy.float64

ctypedef numpy.int32_t INTDTYPE_t
ctypedef numpy.float64_t FLOATDTYPE_t


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def raster_func_c_self_bleach_split_g(
        object self,
        object datamap,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] acquired_intensity,
        numpy.ndarray[INTDTYPE_t, ndim=2] pixel_list,
        int ratio,
        int rows_pad,
        int cols_pad,
        int laser_pad,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] pdt_roi,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] p_ex_roi,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] p_sted_roi,
        bint bleach,   # bint is a bool
        dict bleached_sub_datamaps_dict,
        int seed,
        object bleach_func,   # uncertain of the type for a cfunc, but this seems to be working so ???
        object sample_func,
        list steps
):
    cdef int row, col
    cdef int sprime, tprime
    cdef int h, w
    cdef int current
    cdef int max_len = len(pixel_list)
    cdef FLOATDTYPE_t value
    cdef int sampled_value
    cdef int prob
    cdef int rsamp
    cdef FLOATDTYPE_t pdt, p_ex, p_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] pre_effective, effective
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex, k_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex, i_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
    cdef numpy.ndarray[int, ndim=2] bleached_datamap
    cdef FLOATDTYPE_t duty_cycle
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] step

    """
    raster_func_c_self_bleach executes the simultaneous acquisition and bleaching routine for the case where the 
    excitation power (p_ex) AND/OR sted power (p_sted) vary through the sample. This function thus requires these
    parameters to be passed as arrays of floats the same size as the ROI being imaged.

    Additionally, this function seperately bleaches the different parts composing the datamap (i.e. the base and flash
    components of the datamap are bleached separately).
    """
    if seed == 0:
        # if no seed is passed, calculates a 'pseudo-random' seed form the time in ns
        srand(int(str(time.time_ns())[-5:-1]))
    else:
        srand(seed)

    i_ex, i_sted, _ = self.cache(datamap.pixelsize)
    pre_effective = self.get_effective(datamap.pixelsize, p_ex_roi[0, 0], p_sted_roi[0, 0])
    h, w = pre_effective.shape[0], pre_effective.shape[1]

    for (row, col) in pixel_list:
        pdt = pdt_roi[row, col]
        p_ex = p_ex_roi[row, col]
        p_sted = p_sted_roi[row, col]
        effective = self.get_effective(datamap.pixelsize, p_ex, p_sted)
        # i think resetting each time ensures that we are acquiring on the dmap while it is
        # being bleached. Either way, it doesn't affect speed, so I will keep it here
        bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int32)
        for key in bleached_sub_datamaps_dict:
            bleached_datamap += bleached_sub_datamaps_dict[key]

        value = 0.0
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                value += effective[sprime, tprime] * bleached_datamap[s, t]
                tprime += 1
            sprime += 1
        acquired_intensity[int(row / ratio), int(col / ratio)] = value

        if bleach:
            #
            for step in steps:
                bleach_func(self, i_ex, i_sted, p_ex, p_sted, step[row, col], bleached_sub_datamaps_dict, row, col, h, w, prob_ex,
                            prob_sted)
            sample_func(self, bleached_sub_datamaps_dict, row, col, h, w, prob_ex, prob_sted)