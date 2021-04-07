
import time
import numpy
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
def default_bleach(object self,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] i_sted,
                   float p_ex,
                   float p_sted,
                   float pdt,
                   dict bleached_sub_datamaps_dict,
                   int row,
                   int col,
                   int h,
                   int w,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted):
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex, k_sted
    cdef int s, sprime, t, tprime
    cdef FLOATDTYPE_t value
    cdef int sampled_value
    cdef int prob
    cdef int rsamp
    cdef int current
    cdef str key
    cdef float duty_cycle

    # print("CUMaroonie")

    photons_ex = self.fluo.get_photons(i_ex * p_ex)
    k_ex = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)
    duty_cycle = self.sted.tau * self.sted.rate
    photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle)
    k_sted = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)

    for key in bleached_sub_datamaps_dict:
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                # Updates probabilites
                # I THINK I COMPUTE THIS WETHER THE PIXEL WAS EMPTY OR NOT?
                prob_ex[s, t] = prob_ex[s, t] * exp(-1. * k_ex[sprime, tprime] * pdt)
                prob_sted[s, t] = prob_sted[s, t] * exp(-1. * k_sted[sprime, tprime] * pdt)

                # only need to compute bleaching (resampling) if the pixel is not empty
                current = bleached_sub_datamaps_dict[key][s, t]
                if current > 0:
                    # Calculates the binomial sampling
                    sampled_value = 0
                    prob = int(prob_ex[s, t] * prob_sted[s, t] * RAND_MAX)
                    # For each count we sample a random variable
                    for o in range(current):
                        rsamp = rand()
                        if rsamp <= prob:
                            sampled_value += 1
                    bleached_sub_datamaps_dict[key][s, t] = sampled_value

                tprime += 1
            sprime += 1


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def test_bleach(object self,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] i_sted,
                   float p_ex,
                   float p_sted,
                   float pdt,
                   dict bleached_sub_datamaps_dict,
                   int row,
                   int col,
                   int h,
                   int w,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted):
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex, k_sted
    cdef int s, sprime, t, tprime
    cdef FLOATDTYPE_t value
    cdef int sampled_value
    cdef int prob
    cdef int rsamp
    cdef int current
    cdef str key
    cdef float duty_cycle

    # print("CUMaroonie")

    photons_ex = self.fluo.get_photons(i_ex * p_ex)
    k_ex = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)
    duty_cycle = self.sted.tau * self.sted.rate
    photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle)
    k_sted = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)
    # je pense que j'ai pas le choix de prendre les mêmes args que la default function? idk
    # le but de cette bleaching func est de bleacher vrm pas bcp quand c'est pas un flash et bleacher bcp quand oui :)

    for key in bleached_sub_datamaps_dict:
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                # Updates probabilites
                # I THINK I COMPUTE THIS WETHER THE PIXEL WAS EMPTY OR NOT?

                # only need to compute bleaching (resampling) if the pixel is not empty
                current = bleached_sub_datamaps_dict[key][s, t]
                if current > 0:
                    if key == "base":
                        # I think this amounts to linear, VERIFY THIS
                        prob_ex[s, t] = prob_ex[s, t] - 0.000000000000001
                        prob_sted[s, t] = prob_sted[s, t] - 0.000000000000001
                    elif key == "flashes":
                        # I think this amounts to exponential, VERIFY THIS
                        prob_ex[s, t] = prob_ex[s, t] * 0.9999999
                        prob_sted[s, t] = prob_sted[s, t] * 0.9999999
                    # Calculates the binomial sampling
                    sampled_value = 0
                    prob = int(prob_ex[s, t] * prob_sted[s, t] * RAND_MAX)
                    # For each count we sample a random variable
                    for o in range(current):
                        rsamp = rand()
                        if rsamp <= prob:
                            sampled_value += 1
                    bleached_sub_datamaps_dict[key][s, t] = sampled_value

                tprime += 1
            sprime += 1
