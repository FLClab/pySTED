
"""
This module contains different bleaching functions that can be used when acquiring signal and bleaching a Datamap

Code written by Benoit Turcotte, benoit.turcotte.4@ulaval.ca, October 2020
For use by FLClab (@CERVO) authorized people
"""

import numpy
from matplotlib import pyplot


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

def fuck_tout():
    prob_ex = 0
    prob_sted = 0
    return prob_ex, prob_sted