
import time
import numpy
import bleach_funcs
cimport numpy
cimport cython
import copy


from libc.math cimport exp
from libc.stdlib cimport rand, srand, RAND_MAX

INTDTYPE = numpy.int32
FLOATDTYPE = numpy.float64

ctypedef numpy.int32_t INTDTYPE_t
ctypedef numpy.int64_t INT64DTYPE_t
ctypedef numpy.float64_t FLOATDTYPE_t

# I KEEP THIS OLD VERSION HERE, COMMENTED, FOR FUTURE REFERENCE
# @cython.boundscheck(False)  # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# def default_bleach(object self,
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
#     cdef FLOATDTYPE_t value
#     cdef int sampled_value
#     cdef float prob
#     cdef float rsamp
#     cdef float maxval
#     cdef float sampled_prob
#     cdef int current
#     cdef str key
#     cdef float duty_cycle
#
#     maxval = float(RAND_MAX)
#
#     photons_ex = self.fluo.get_photons(i_ex * p_ex)
#     k_ex = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)
#     duty_cycle = self.sted.tau * self.sted.rate
#     photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle)
#     k_sted = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)
#
#     for key in bleached_sub_datamaps_dict:
#         sprime = 0
#         for s in range(row, row + h):
#             tprime = 0
#             for t in range(col, col + w):
#                 # Updates probabilites
#                 # I THINK I COMPUTE THIS WETHER THE PIXEL WAS EMPTY OR NOT?
#                 prob_ex[s, t] = prob_ex[s, t] * exp(-1. * k_ex[sprime, tprime] * pdt)
#                 prob_sted[s, t] = prob_sted[s, t] * exp(-1. * k_sted[sprime, tprime] * pdt)
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

# The bleaching functions are split into a survival probabilities update part, and a
# sampling part. This doesn't really matter for the default get_signal case,
# but it is useful for other get_signal implems such as DyMin where the pdt is split into steps

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def test_update_survival_probabilities(object self,
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
    cdef int prob
    cdef int rsamp
    cdef int current
    cdef str key
    cdef float duty_cycle


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

                tprime += 1
            sprime += 1

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def default_bleach_multisteps(object self,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] i_sted,
                   numpy.ndarray[FLOATDTYPE_t, ndim=1] p_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=1] p_sted,
                   numpy.ndarray[FLOATDTYPE_t, ndim=1] pdt,
                   dict bleached_sub_datamaps_dict,
                   int row,
                   int col,
                   int h,
                   int w,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
                   numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted):
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=3] k_ex, k_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=1] tmp
    cdef int s, sprime, t, tprime, i
    cdef FLOATDTYPE_t value
    cdef int sampled_value
    cdef float prob
    cdef float rsamp
    cdef float maxval
    cdef float sampled_prob
    cdef float expsum
    cdef int current
    cdef str key
    cdef float duty_cycle

    maxval = float(RAND_MAX)

    k_ex = numpy.zeros((len(p_ex), i_ex.shape[0], i_ex.shape[1]), dtype=i_ex.dtype)
    for i in range(len(p_ex)):
      photons_ex = self.fluo.get_photons(i_ex * p_ex[i])
      k_ex[i] = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)

    k_sted = numpy.zeros((len(p_sted), i_sted.shape[0], i_sted.shape[1]), dtype=i_sted.dtype)
    for i in range(len(p_sted)):
      duty_cycle = self.sted.tau * self.sted.rate
      photons_sted = self.fluo.get_photons(i_sted * p_sted[i] * duty_cycle)
      k_sted[i] = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)

    for key in bleached_sub_datamaps_dict:
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                # Updates probabilites
                # I THINK I COMPUTE THIS WETHER THE PIXEL WAS EMPTY OR NOT?
                expsum = 0.
                for i in range(len(p_sted)):
                  expsum = expsum + k_ex[i, sprime, tprime] * pdt[i]
                prob_ex[s, t] = prob_ex[s, t] * exp(-1. * expsum)
                expsum = 0.
                for i in range(len(p_sted)):
                  expsum = expsum + k_sted[i, sprime, tprime] * pdt[i]
                prob_sted[s, t] = prob_sted[s, t] * exp(-1. * expsum)

                # only need to compute bleaching (resampling) if the pixel is not empty
                current = bleached_sub_datamaps_dict[key][s, t]
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
                    bleached_sub_datamaps_dict[key][s, t] = sampled_value

                tprime += 1
            sprime += 1

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
#    print("checkpoint1=", time.time()-t0); t0 = time.time()
    for key in bleached_sub_datamaps_dict:
#        sprime = 0
        for i, s in enumerate(range(row, row + h)):
#            tprime = 0
            for j, t in enumerate(range(col, col + w)):
                # Updates probabilites
                # I THINK I COMPUTE THIS WETHER THE PIXEL WAS EMPTY OR NOT?
#                prob_ex[s, t] = prob_ex[s, t] * exp(-1. * k_ex[sprime, tprime] * step)
#                prob_sted[s, t] = prob_sted[s, t] * exp(-1. * k_sted[sprime, tprime] * step)
                 prob_ex[i, j] = prob_ex[i, j] * exp(-1. * k_ex[i, j] * step)
                 prob_sted[i, j] = prob_sted[i, j] * exp(-1. * k_sted[i, j] * step)
#                 nana=1*3**8
#                print(key, s, t)
                #if prob_ex[s, t] <1 or prob_sted[s, t]<1:
#                if k_sted[sprime, tprime] > 0.1 :
#                if k_sted[sprime, tprime] > 1000 :
#                if prob_sted[s, t] < -0.99983:
#                    print("#--debug--")
#                    from pprint import pprint
#                    debug_dict = dict(
#                            key=key,
#                            sprime=sprime,
#                            tprime=tprime,
#                            row=row,
#                            col=col,
#                            k_sted=k_sted[sprime, tprime],
#                            k_sted_max = k_sted.max(),
#                            k_sted_mean = k_sted.mean(),
#                            k_ex=k_ex[sprime, tprime],
#                            k_ex_max = k_ex.max(),
#                            k_ex_mean = k_ex.mean(),
#                            p_ex=p_ex,
#                            p_sted=p_sted,
#                            step=step,
#                            prob_ex = prob_ex[s, t],
#                            prob_sted = prob_sted[s, t],
#                            infered_prob_sted=numpy.exp(-k_sted[sprime, tprime]*step),
#                            )
#                    pprint(debug_dict)
#                    import pdb; pdb.set_trace()
#                    print("#--debug--")
#                tprime += 1
#            sprime += 1

#    print("checkpoint2=", time.time()-t0); t0 = time.time()

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def sample_molecules(object self,
                   dict bleached_sub_datamaps_dict,
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
    cdef str key
    cdef numpy.ndarray[INT64DTYPE_t, ndim=2] copied_datamap

    maxval = float(RAND_MAX)
    
    
#    fastmap = numpy.ones(bleached_sub_datamaps_dict['base'].shape)

    for key in bleached_sub_datamaps_dict:
        copied_datamap = copy.deepcopy(bleached_sub_datamaps_dict[key])   
#        copied_datamap = numpy.ones(bleached_sub_datamaps_dict[key].shape)
        for i, s in enumerate(range(row, row + h)):
            for j, t in enumerate(range(col, col + w)):
#                current = bleached_sub_datamaps_dict[key][s, t]
                current = copied_datamap[s,t]
#                current = 1 
                if current > 0:
                    # Calculates the binomial sampling
                    sampled_value = 0
#                    prob = prob_ex[s, t] * prob_sted[s, t]
                    prob = prob_ex[i, j] * prob_sted[i, j]
                    # For each count we sample a random variable
                    for o in range(current):
                        rsamp = rand()
                        sampled_prob = rsamp / maxval
                        if sampled_prob <= prob:
                            sampled_value += 1



##                    if sampled_value >= 10:
#                    if sampled_value >= 10000:
##                    if sampled_value < current:
#                        from pprint import pprint
#                        print("#--debug--")
#                        debug_dict = dict(
#                                key=key,
#                                row=row,
#                                h=h,
#                                col=col,
#                                w=w,
#                                maxval=maxval,
#                                sampled_value=sampled_value,
#                                s=s,
#                                t=t,
#                                current=current,
#                                prob=prob,
#                                sampled_prob=sampled_prob,
#                                infered_k=-numpy.log(prob)/15e-6
#                                )
#                        print(debug_dict)
#                        import pdb; pdb.set_trace()
#                        print("#---------")


#                    bleached_sub_datamaps_dict[key][s, t] = sampled_value
                    copied_datamap[s, t] = sampled_value
        bleached_sub_datamaps_dict[key] = copied_datamap
