
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


def fuck_tout(p_ex, p_sted, pdt, prob_ex, prob_sted, region):
    prob_ex[region] = 0
    prob_sted[region] = 0
    return prob_ex, prob_sted


def fifty_fifty(p_ex, p_sted, pdt, prob_ex, prob_sted, region):
    prob_ex[region] = 0.5
    prob_sted[region] = 0.5
    return prob_ex, prob_sted


# if you add a function, add it to the dict so it gets detected
functions_dict = {"default_bleach": partial(default_bleach), "fuck_tout": partial(fuck_tout),
                  "fifty_fifty": partial(fifty_fifty)}
