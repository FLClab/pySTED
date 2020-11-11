
import time
import numpy
cimport numpy
cimport cython

from libc.math cimport exp
from libc.stdlib cimport rand, srand, RAND_MAX

INTDTYPE = numpy.int32
FLOATDTYPE = numpy.float64

ctypedef numpy.int32_t INTDTYPE_t
ctypedef numpy.float64_t FLOATDTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def raster_func_wbleach_c(
    numpy.ndarray[FLOATDTYPE_t, ndim=2] acquired_intensity,
    numpy.ndarray[INTDTYPE_t, ndim=2] pixel_list,
    int ratio,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] effective,
    int rows_pad,
    int cols_pad,
    int laser_pad,
    numpy.ndarray[INTDTYPE_t, ndim=2] whole_datamap,
    int bleach,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] k_sted,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] pdt_roi,
    numpy.ndarray[INTDTYPE_t, ndim=2] bleached_datamap,
    int seed
    ):

    cdef int row, col
    cdef int s_from, s_to, t_from, t_to
    cdef int sprime, tprime
    cdef int h = effective.shape[0]
    cdef int w = effective.shape[0]
    cdef int current
    cdef int max_len = len(pixel_list)
    cdef FLOATDTYPE_t value
    cdef numpy.ndarray[INTDTYPE_t, ndim=2] wdmap
    cdef int sampled_value
    cdef int prob
    cdef FLOATDTYPE_t pdt
    cdef int rsamp

    if seed == 0:
        # if no seed is passed, calculates a 'pseudo-random' seed form the time in ns
        srand(int(str(time.time_ns())[-5:-1]))
    else:
        srand(seed)
    for i in range(max_len):
        row, col = pixel_list[i]

        value = 0
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                value += effective[sprime, tprime] * bleached_datamap[s, t]
                tprime += 1
            sprime += 1
        acquired_intensity[int(row / ratio), int(col / ratio)] = value

        pdt = pdt_roi[row, col]
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                # Updates probabilites
                prob_ex[s, t] = prob_ex[s, t] * exp(-1. * k_ex[sprime, tprime] * pdt)
                prob_sted[s, t] = prob_sted[s, t] * exp(-1. * k_sted[sprime, tprime] * pdt)

                # Calculates the binomial sampling
                sampled_value = 0
                current = bleached_datamap[s, t]
                prob = int(prob_ex[s, t] * prob_sted[s, t] * RAND_MAX)
                # For each count we sample a random variable
                for o in range(current):
                    rsamp = rand()
                    if rsamp <= prob:
                        sampled_value += 1
                bleached_datamap[s, t] = sampled_value

                tprime += 1
            sprime += 1

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def raster_func_c(
    numpy.ndarray[FLOATDTYPE_t, ndim=2] acquired_intensity,
    numpy.ndarray[INTDTYPE_t, ndim=2] pixel_list,
    int ratio,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] effective,
    int rows_pad,
    int cols_pad,
    int laser_pad,
    numpy.ndarray[INTDTYPE_t, ndim=2] whole_datamap,
    int bleach,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] k_sted,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] pdt_roi,
    numpy.ndarray[INTDTYPE_t, ndim=2] bleached_datamap,
    int seed
    ):

    cdef int row, col
    cdef int s_from, s_to, t_from, t_to
    cdef int sprime, tprime
    cdef int h = effective.shape[0]
    cdef int w = effective.shape[0]
    cdef int current
    cdef int max_len = len(pixel_list)
    cdef FLOATDTYPE_t value
    cdef numpy.ndarray[INTDTYPE_t, ndim=2] wdmap
    cdef int sampled_value
    cdef int prob
    cdef FLOATDTYPE_t pdt

    if seed == 0:
        # if no seed is passed, calculates a 'pseudo-random' seed form the time in ns
        srand(int(str(time.time_ns())[-5:-1]))
    else:
        srand(seed)

    for i in range(max_len):
        row, col = pixel_list[i]

        value = 0
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                value += effective[sprime, tprime] * whole_datamap[s, t]
                tprime += 1
            sprime += 1
        acquired_intensity[int(row / ratio), (col / ratio)] = value
