
import time
import numpy
import bleach_funcs
cimport numpy
cimport cython
import copy

from libc.math cimport exp
from libc.stdlib cimport rand, srand, RAND_MAX

INTDTYPE = numpy.int32
INT64DTYPE = numpy.int64
FLOATDTYPE = numpy.float64

ctypedef numpy.int32_t INTDTYPE_t
ctypedef numpy.int64_t INT64DTYPE_t
ctypedef numpy.float64_t FLOATDTYPE_t

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def default_update_survival_probabilities(object self,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] i_sted,
                   float p_ex,
                   float p_sted,
                   float step,
                   dict bleached_sub_datamaps_dict,
                   int row,
                   int col,
                   int h,
                   int w,
                   list mask,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex=None,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] k_sted=None,):
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
    #cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex, k_sted
    cdef int s, sprime, t, tprime
    cdef float prob
    cdef float rsamp
    cdef float maxval
    cdef float sampled_prob
    cdef int current
    cdef str key
    cdef float duty_cycle

    t0 = time.time()

    maxval = float(RAND_MAX)
#    from pprint import pprint
#    pprint(dict(
#        k_sted_mean = numpy.mean(k_sted),
#        k_sted_max = numpy.max(k_sted),
#        k_ex_mean = numpy.mean(k_ex)
#        ))
    if k_sted is None:
        photons_ex = self.fluo.get_photons(i_ex * p_ex, self.excitation.lambda_)
        duty_cycle = self.sted.tau * self.sted.rate
        photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle, self.sted.lambda_)
        k_sted = self.fluo.get_k_bleach(self.excitation.lambda_, self.sted.lambda_, photons_ex, photons_sted, self.sted.tau, 1/self.sted.rate, step, )
    if k_ex is None:
        k_ex = k_sted * 0.
    for (s, t) in mask:
        # sprime = s - row
        # tprime = t - col
        prob_ex[s, t] = prob_ex[s, t] * exp(-1. * k_ex[s, t] * step)
        prob_sted[s, t] = prob_sted[s, t] * exp(-1. * k_sted[s, t] * step)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def sample_molecules(object self,
                   dict bleached_sub_datamaps_dict,
                   int row,
                   int col,
                   int h,
                   int w,
                   list mask,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted):
    cdef int s, sprime, t, tprime, o
    cdef int sampled_value
    cdef float prob
    cdef float rsamp
    cdef float maxval
    cdef float sampled_prob
    cdef int current
    cdef numpy.ndarray[INT64DTYPE_t, ndim=2] datamap
    cdef str key
    cdef numpy.ndarray[INT64DTYPE_t, ndim=2] copied_datamap

    maxval = float(RAND_MAX)

    for key in bleached_sub_datamaps_dict:
        datamap = bleached_sub_datamaps_dict[key]
        for (sprime, tprime) in mask:
            s = sprime + row
            t = tprime + col
            current = datamap[s, t]
            if current > 0:
                # Calculates the binomial sampling
                sampled_value = 0
                prob = prob_ex[s, t] * prob_sted[s, t]
                # For each count we sample a random variable
                for o in range(current):
                    rsamp = rand()
                    sampled_prob = rsamp / maxval
                    if sampled_prob <= prob:
                        sampled_value += 1
                datamap[s, t] = sampled_value
        bleached_sub_datamaps_dict[key] = datamap
