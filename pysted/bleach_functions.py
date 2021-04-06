
"""
This module contains different bleaching functions that can be used when acquiring signal and bleaching a Datamap

Code written by Benoit Turcotte, benoit.turcotte.4@ulaval.ca, October 2020
For use by FLClab (@CERVO) authorized people
"""

import numpy
from functools import partial


def default_bleach(i_ex, i_sted, fluo, excitation, sted, p_ex, p_sted, pdt, prob_ex, prob_sted, region):
    """
    jesus christ 11 param??
    I want this to be the default the bleaching that is currently used inside microscope.get_signal_and_bleach.
    Is there anything else to say for now?
    :return:
    """

    photons_ex = fluo.get_photons(i_ex * p_ex)
    k_ex = fluo.get_k_bleach(excitation.lambda_, photons_ex)

    duty_cycle = sted.tau * sted.rate
    photons_sted = fluo.get_photons(i_sted * p_sted * duty_cycle)
    k_sted = fluo.get_k_bleach(sted.lambda_, photons_sted)

    prob_ex[region] *= numpy.exp(-k_ex * pdt)
    prob_sted[region] *= numpy.exp(-k_sted * pdt)

    return prob_ex, prob_sted


def fuck_tout(prob_ex, prob_sted, region, **kwargs):
    prob_ex[region] = 0
    prob_sted[region] = 0
    return prob_ex, prob_sted


def fifty_fifty(prob_ex, prob_sted, region, **kwargs):
    prob_ex[region] = 0.5
    prob_sted[region] = 0.5
    return prob_ex, prob_sted


def sted_exc(p_ex, p_sted, prob_ex, prob_sted, region, **kwargs):
    """
    Meant to replicate the 'if only STED, barely bleaches, if only exc, bleaches a bit, if both, bleaches a lot
    should probably add in the p_ex, p_sted somewhere in the calculation
    The values I put here work well when generating a laser with dpxsz = 50 nm, but I need to find a way to make
    it appropriate no matter the dpxsz used...
    """
    if p_ex == 0 and p_sted == 0:
        # in this case the survival probabilities do not change
        pass
    elif p_ex == 0 and p_sted != 0:
        # barely bleaches
        prob_sted[region] *= 0.999999
    elif p_ex != 0 and p_sted == 0:
        # bleaches a bit
        prob_ex[region] *= 0.999993
    else:
        prob_sted[region] *= 0.999999
        prob_ex[region] *= 0.99999
    return prob_ex, prob_sted


# C'EST DUMB ÇA!!! IL FAUT QUE MES BLEACHING FUNCTIONS SOIENT DES C FUNC, SINON ÇA DEFEAT LE PURPOUSE DE FAIRE DU CYTHON
def test(self, i_ex, i_sted, p_ex, p_sted, pdt, bleached_sub_datamaps_dict, row, col, h, w, prob_ex, prob_sted, ):
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
                prob_ex[s, t] = prob_ex[s, t] * numpy.exp(-1. * k_ex[sprime, tprime] * pdt)
                prob_sted[s, t] = prob_sted[s, t] * numpy.exp(-1. * k_sted[sprime, tprime] * pdt)

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

# if you add a function, add it to the dict so it gets detected
functions_dict = {"default_bleach": partial(default_bleach), "fuck_tout": partial(fuck_tout),
                  "fifty_fifty": partial(fifty_fifty), "sted_exc": partial(sted_exc)}
