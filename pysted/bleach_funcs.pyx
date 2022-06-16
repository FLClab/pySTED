
import time
import numpy
import bleach_funcs
cimport numpy
cimport cython

from libc.math cimport exp
from libc.stdlib cimport rand, srand, RAND_MAX
from cython.parallel import prange

INTDTYPE = numpy.int64
FLOATDTYPE = numpy.float64

ctypedef numpy.int64_t INTDTYPE_t
ctypedef numpy.float64_t FLOATDTYPE_t

# # I KEEP THIS OLD VERSION HERE, COMMENTED, FOR FUTURE REFERENCE
# # @cython.boundscheck(False)  # turn off bounds-checking for entire function
# # @cython.wraparound(False)  # turn off negative index wrapping for entire function
# # def default_bleach(object self,
# #                    numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex,
# #                    numpy.ndarray[FLOATDTYPE_t, ndim=2] i_sted,
# #                    float p_ex,
# #                    float p_sted,
# #                    float pdt,
# #                    dict bleached_sub_datamaps_dict,
# #                    int row,
# #                    int col,
# #                    int h,
# #                    int w,
# #                    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
# #                    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted):
# #     cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
# #     cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex, k_sted
# #     cdef int s, sprime, t, tprime
# #     cdef FLOATDTYPE_t value
# #     cdef int sampled_value
# #     cdef float prob
# #     cdef float rsamp
# #     cdef float maxval
# #     cdef float sampled_prob
# #     cdef int current
# #     cdef str key
# #     cdef float duty_cycle
# #
# #     maxval = float(RAND_MAX)
# #
# #     photons_ex = self.fluo.get_photons(i_ex * p_ex)
# #     k_ex = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)
# #     duty_cycle = self.sted.tau * self.sted.rate
# #     photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle)
# #     k_sted = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)
# #
# #     for key in bleached_sub_datamaps_dict:
# #         sprime = 0
# #         for s in range(row, row + h):
# #             tprime = 0
# #             for t in range(col, col + w):
# #                 # Updates probabilites
# #                 # I THINK I COMPUTE THIS WETHER THE PIXEL WAS EMPTY OR NOT?
# #                 prob_ex[s, t] = prob_ex[s, t] * exp(-1. * k_ex[sprime, tprime] * pdt)
# #                 prob_sted[s, t] = prob_sted[s, t] * exp(-1. * k_sted[sprime, tprime] * pdt)
# #
# #                 # only need to compute bleaching (resampling) if the pixel is not empty
# #                 current = bleached_sub_datamaps_dict[key][s, t]
# #                 if current > 0:
# #                     # Calculates the binomial sampling
# #                     sampled_value = 0
# #                     prob = prob_ex[s, t] * prob_sted[s, t]
# #                     # For each count we sample a random variable
# #                     for o in range(current):
# #                         rsamp = rand()
# #                         sampled_prob = rsamp / maxval
# #                         if sampled_prob <= prob:
# #                             sampled_value += 1
# #                     bleached_sub_datamaps_dict[key][s, t] = sampled_value
# #
# #                 tprime += 1
# #             sprime += 1
#
# # The bleaching functions are split into a survival probabilities update part, and a
# # sampling part. This doesn't really matter for the default get_signal case,
# # but it is useful for other get_signal implems such as DyMin where the pdt is split into steps
#
# @cython.boundscheck(False)  # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# def test_update_survival_probabilities(object self,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=2] i_sted,
#                    float p_ex,
#                    float p_sted,
#                    float pdt,
#                    dict bleached_sub_datamaps_dict,
#                    int row,
#                    int col,
#                    int h,
#                    int w,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted):
#     cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
#     cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex, k_sted
#     cdef int s, sprime, t, tprime
#     cdef int prob
#     cdef int rsamp
#     cdef int current
#     cdef str key
#     cdef float duty_cycle
#
#
#     photons_ex = self.fluo.get_photons(i_ex * p_ex)
#     k_ex = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)
#     duty_cycle = self.sted.tau * self.sted.rate
#     photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle)
#     k_sted = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)
#     # je pense que j'ai pas le choix de prendre les mêmes args que la default function? idk
#     # le but de cette bleaching func est de bleacher vrm pas bcp quand c'est pas un flash et bleacher bcp quand oui :)
#
#     for key in bleached_sub_datamaps_dict:
#         sprime = 0
#         for s in range(row, row + h):
#             tprime = 0
#             for t in range(col, col + w):
#                 # Updates probabilites
#                 # I THINK I COMPUTE THIS WETHER THE PIXEL WAS EMPTY OR NOT?
#
#                 # only need to compute bleaching (resampling) if the pixel is not empty
#                 current = bleached_sub_datamaps_dict[key][s, t]
#                 if current > 0:
#                     if key == "base":
#                         # I think this amounts to linear, VERIFY THIS
#                         prob_ex[s, t] = prob_ex[s, t] - 0.000000000000001
#                         prob_sted[s, t] = prob_sted[s, t] - 0.000000000000001
#                     elif key == "flashes":
#                         # I think this amounts to exponential, VERIFY THIS
#                         prob_ex[s, t] = prob_ex[s, t] * 0.9999999
#                         prob_sted[s, t] = prob_sted[s, t] * 0.9999999
#
#                 tprime += 1
#             sprime += 1
#
# @cython.boundscheck(False)  # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# def default_bleach_multisteps(object self,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=2] i_sted,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=1] p_ex,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=1] p_sted,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=1] pdt,
#                    dict bleached_sub_datamaps_dict,
#                    int row,
#                    int col,
#                    int h,
#                    int w,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted):
#     cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
#     cdef numpy.ndarray[FLOATDTYPE_t, ndim=3] k_ex, k_sted
#     cdef numpy.ndarray[FLOATDTYPE_t, ndim=1] tmp
#     cdef int s, sprime, t, tprime, i
#     cdef FLOATDTYPE_t value
#     cdef int sampled_value
#     cdef float prob
#     cdef float rsamp
#     cdef float maxval
#     cdef float sampled_prob
#     cdef float expsum
#     cdef int current
#     cdef str key
#     cdef float duty_cycle
#
#     maxval = float(RAND_MAX)
#
#     k_ex = numpy.zeros((len(p_ex), i_ex.shape[0], i_ex.shape[1]), dtype=i_ex.dtype)
#     for i in range(len(p_ex)):
#       photons_ex = self.fluo.get_photons(i_ex * p_ex[i])
#       k_ex[i] = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)
#
#     k_sted = numpy.zeros((len(p_sted), i_sted.shape[0], i_sted.shape[1]), dtype=i_sted.dtype)
#     for i in range(len(p_sted)):
#       duty_cycle = self.sted.tau * self.sted.rate
#       photons_sted = self.fluo.get_photons(i_sted * p_sted[i] * duty_cycle)
#       k_sted[i] = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)
#
#     for key in bleached_sub_datamaps_dict:
#         sprime = 0
#         for s in range(row, row + h):
#             tprime = 0
#             for t in range(col, col + w):
#                 # Updates probabilites
#                 # I THINK I COMPUTE THIS WETHER THE PIXEL WAS EMPTY OR NOT?
#                 expsum = 0.
#                 for i in range(len(p_sted)):
#                   expsum = expsum + k_ex[i, sprime, tprime] * pdt[i]
#                 prob_ex[s, t] = prob_ex[s, t] * exp(-1. * expsum)
#                 expsum = 0.
#                 for i in range(len(p_sted)):
#                   expsum = expsum + k_sted[i, sprime, tprime] * pdt[i]
#                 prob_sted[s, t] = prob_sted[s, t] * exp(-1. * expsum)
#
#                 # only need to compute bleaching (resampling) if the pixel is not empty
#                 current = bleached_sub_datamaps_dict[key][s, t]
#                 if current > 0:
#                     # Calculates the binomial sampling
#                     sampled_value = 0
#                     prob = prob_ex[s, t] * prob_sted[s, t]
#                     # For each count we sample a random variable
#                     for o in range(current):
#                         rsamp = rand()
#                         sampled_prob = rsamp / maxval
#                         if sampled_prob <= prob:
#                             sampled_value += 1
#                     bleached_sub_datamaps_dict[key][s, t] = sampled_value
#
#                 tprime += 1
#             sprime += 1

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def default_update_survival_probabilities(object self,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] i_sted,
                   float p_ex,
                   float p_sted,
                   float step,
                   numpy.ndarray[INTDTYPE_t, ndim=3] bleached_sub_datamaps_dict,
                   int row,
                   int col,
                   int h,
                   int w,
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
    cdef int key
    cdef float duty_cycle
    cdef float tmp

    maxval = float(RAND_MAX)
    if k_sted is None:
        photons_ex = self.fluo.get_photons(i_ex * p_ex, self.excitation.lambda_)
        duty_cycle = self.sted.tau * self.sted.rate
        photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle, self.sted.lambda_)
        k_sted = self.fluo.get_k_bleach(self.excitation.lambda_, self.sted.lambda_, photons_ex, photons_sted, self.sted.tau, 1/self.sted.rate, step, )
    if k_ex is None:
        k_ex = k_sted * 0.

    # for key in range(len(bleached_sub_datamaps_dict)):
    for s in prange(row, row + h, nogil=True):
        sprime = s - row
        for t in range(col, col + w):
            tprime = t - col
            # Updates probabilites
            # I THINK I COMPUTE THIS WETHER THE PIXEL WAS EMPTY OR NOT?
            prob_ex[s, t] = prob_ex[s, t] * exp(-1. * k_ex[sprime, tprime] * step)
            prob_sted[s, t] = prob_sted[s, t] * exp(-1. * k_sted[sprime, tprime] * step)


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def sample_molecules(object self,
                   numpy.ndarray[INTDTYPE_t, ndim=3] bleached_sub_datamaps_dict,
                   int row,
                   int col,
                   int h,
                   int w,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted):
    cdef int s, sprime, t, tprime, o
    cdef int sampled_value
    cdef float prob
    cdef float rsamp
    cdef float maxval
    cdef float sampled_prob
    cdef int current
    cdef int key

    maxval = float(RAND_MAX)

    for key in range(bleached_sub_datamaps_dict.shape[0]):
        for s in range(row, row + h):
            for t in range(col, col + w):
                current = bleached_sub_datamaps_dict[key, s, t]
                if current > 0:
                    # binomial(bleached_sub_datamaps_dict, current, key, s, t, maxval, prob_ex, prob_sted)
                    sampled_value = 0
                    prob = prob_ex[s, t] * prob_sted[s, t]
                    # For each count we sample a random variable
                    for o in range(current):
                        rsamp = rand()
                        sampled_prob = rsamp / maxval
                        if sampled_prob <= prob:
                            sampled_value = sampled_value + 1

                    bleached_sub_datamaps_dict[key, s, t] = sampled_value

# @cython.boundscheck(False)  # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# def sample_molecules(object self,
#                    dict bleached_sub_datamaps_dict,
#                    int row,
#                    int col,
#                    int h,
#                    int w,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
#                    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted):
#     cdef int s, t
#     cdef float maxval
#     cdef str key
#
#     maxval = float(RAND_MAX)
#
#     for key in bleached_sub_datamaps_dict:
#         for s in range(row, row + h):
#             for t in range(col, col + w):
#                 binomial(bleached_sub_datamaps_dict, key, s, t, maxval, prob_ex, prob_sted)

# @cython.boundscheck(False)  # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# def binomial(dict bleached_sub_datamaps_dict, int current, str key, int s, int t, float maxval,
#             numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
#             numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted
#     ):
#     """
#     Samples from the datamap using a binomial
#     """
#     cdef int o
#     # cdef int current
#     cdef float prob
#     cdef float rsamp
#     cdef int sampled_value
#     cdef float sampled_prob
#
#     # current = bleached_sub_datamaps_dict[key][s, t]
#     # if current > 0:
#     # Calculates the binomial sampling
#     sampled_value = 0
#     prob = prob_ex[s, t] * prob_sted[s, t]
#     # For each count we sample a random variable
#     for o in prange(current, nogil=True):
#         rsamp = rand()
#         sampled_prob = rsamp / maxval
#         if sampled_prob <= prob:
#             sampled_value += 1
#
#     bleached_sub_datamaps_dict[key][s, t] = sampled_value

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def binomial(dict bleached_sub_datamaps_dict, int current, str key, int s, int t, float maxval,
            numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
            numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted
    ):
    """
    Samples from the datamap using a binomial
    """
    cdef int o
    # cdef int current
    cdef float prob
    cdef float rsamp
    cdef int sampled_value
    cdef float sampled_prob

    # current = bleached_sub_datamaps_dict[key][s, t]
    # if current > 0:
    # Calculates the binomial sampling
    sampled_value = 0
    prob = prob_ex[s, t] * prob_sted[s, t]
    # For each count we sample a random variable
    for o in range(current):
        rsamp = rand()
        sampled_prob = rsamp / maxval
        if sampled_prob <= prob:
            sampled_value += 1

    bleached_sub_datamaps_dict[key][s, t] = sampled_value
