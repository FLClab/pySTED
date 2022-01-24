import os
import tifffile

import numpy as np
import pandas as pd

from load_utils import *
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from gym_sted.utils import get_foreground
from pysted import base
from skopt import gp_minimize


"""
The goal here is to verify that I am able to optimise correctly using gp_minimize
To achieve this, I will generate a microscope with certain parameters. I will then randomly generate
a datamap with a certain amount of fluorophores in each bead, and image them with the microscope
using a certain set of parameters (and then try with randomly sampling imaging parameters each time?)
Then the function to optimize will try and guess the number of fluorophores it needs to put in each bead
to recreate a datamap that will give it an acquisition that is as similar to the first one as possible
"""


def optimize_n_molecs(n_fluorophores_guess):
    """
    XD
    :param ?: xD!
    :returns: xD!
    """
    # print(f"function called :)")
    molecules_disposition_gt = np.zeros((64, 64))
    molecules_disposition_guess = np.zeros((64, 64))
    random_bead_positions = np.random.randint(0, 64, size=(10, 2))
    n_fluorophores_gt = 150
    for row, col in random_bead_positions:
        molecules_disposition_gt[row, col] += n_fluorophores_gt
        molecules_disposition_guess[row, col] += n_fluorophores_guess

    egfp = {
        "lambda_": 635e-9,  # TODO: verify ok to change like that...
        "qy": 0.6,  # COPIED FROM BEFORE
        "sigma_abs": {
            635: 0.1e-21,  # Table S3, Oracz et al., nature 2017   ALBERT USES 0.1e-21
            750: 3.5e-25,  # (1 photon exc abs) Table S3, Oracz et al., nature 2017
        },
        "sigma_ste": {
            750: 2.8e-22,  # Table S3, Oracz et al., nature 2017   ALBERT USES 4.8e-22
        },
        "sigma_tri": 10.14e-21,  # COPIED FROM BEFORE
        #     "tau": 3e-09,
        "tau": 3.5e-9,  # @646nm, ATTO Fluorescent labels, ATTO-TEC GmbH catalog 2016/2018
        "tau_vib": 1.0e-12,  # t_vib, Table S3, Oracz et al., nature 2017
        "tau_tri": 1.2e-6,  # COPIED FROM BEFORE
        "phy_react": {
            635: 0.008e-6,  # ALBERT USES 0.008e-5
            750: 0.008e-9,  # ALBERT USES 0.008e-8
        },
        "k_isc": 0.48e+6  # COPIED FROM BEFORE
    }

    pixelsize = 20e-9
    laser_ex = base.GaussianBeam(635e-9)
    laser_sted = base.DonutBeam(750e-9, zero_residual=0)
    detector = base.Detector(noise=True, background=4)
    objective = base.Objective()
    fluo = base.Fluorescence(**egfp)
    # PK JDOIS METTRE LOAD_CACHE=TRUE ??? MAKES NO SENSE ???
    microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
    i_ex, i_sted, _ = microscope.cache(pixelsize, save_cache=False)

    dmap_gt = base.TemporalDatamap(molecules_disposition_gt, pixelsize, None)
    dmap_gt.set_roi(i_ex, "max")

    dmap_guess = base.TemporalDatamap(molecules_disposition_guess, pixelsize, None)
    dmap_guess.set_roi(i_ex, "max")

    p_ex_max = 100 * 1.5135e-6
    p_sted_max = 100 * 1.7681e-3
    conf_params = {  # I should ask Albert what the params for conf1 / conf2 were
        "pdt": 20e-6,
        "p_ex": 0.15 * p_ex_max,
        "p_sted": 0.0
    }

    sted_params = {
        "pdt": np.random.uniform(10e-6, 150e-6),
        "p_ex": np.random.uniform(0, 1) * p_ex_max,
        "p_sted": np.random.uniform(0, 1) * p_sted_max
    }

    # eventually I will want to modify this to do multiple acquisitions to factor in noisiness of acqs

    gt_conf1, _, _ = microscope.get_signal_and_bleach(dmap_gt, dmap_gt.pixelsize, **conf_params,
                                                      bleach=False, update=False)
    gt_sted, gt_bleached, _ = microscope.get_signal_and_bleach(dmap_gt, dmap_gt.pixelsize, **sted_params,
                                                               bleach=True, update=True)
    gt_conf2, _, _ = microscope.get_signal_and_bleach(dmap_gt, dmap_gt.pixelsize, **conf_params,
                                                      bleach=False, update=False)

    guess_conf1, _, _ = microscope.get_signal_and_bleach(dmap_guess, dmap_guess.pixelsize, **conf_params,
                                                         bleach=False, update=False)
    guess_sted, guess_bleached, _ = microscope.get_signal_and_bleach(dmap_guess, dmap_guess.pixelsize, **sted_params,
                                                                     bleach=True, update=True)
    guess_conf2, _, _ = microscope.get_signal_and_bleach(dmap_guess, dmap_guess.pixelsize, **sted_params,
                                                         bleach=False, update=False)

    # for now only use the avg photon count on the fg as value with which to optimize
    gt_sted_fg_bool = get_foreground(gt_sted)
    gt_sted_fg = gt_sted * gt_sted_fg_bool

    guess_sted_fg_bool = get_foreground(guess_sted)
    guess_sted_fg = guess_sted * guess_sted_fg_bool

    gt_avg_fg_photoncount = np.sum(gt_sted_fg) / np.sum(gt_sted_fg_bool)   # I think so?
    guess_avg_fg_photoncount = np.sum(guess_sted_fg) / np.sum(guess_sted_fg_bool)

    # we want to minimize the difference between these two values
    objective = np.abs(guess_avg_fg_photoncount - gt_avg_fg_photoncount)

    return objective


if __name__ == "__main__":
    # np.random.seed(42)
    # obj = optimize_n_molecs(150)

    n_calls_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n_repetitions = 5
    # n_calls_list = [5, 6]
    # n_repetitions = 2
    n_molecule_guess_values = np.zeros((len(n_calls_list), n_repetitions))
    objective_values = np.zeros((len(n_calls_list), n_repetitions))
    for row, n_calls in enumerate(n_calls_list):
        for col in range(n_repetitions):
            print(f"n_calls = {n_calls}, repetition {col + 1}")
            res = gp_minimize(optimize_n_molecs,
                              [(0, 300)],
                              acq_func='EI',
                              n_calls=n_calls,
                              n_random_starts=5)

            n_molecule_guess_values[row, col] = res.x[0]
            objective_values[row, col] = res.fun

    save_path = os.path.expanduser(os.path.join("~", "Documents", "research", "h22_code", "gym-sted-dev",
                                                "audurand_pysted", "tests_bt_2", "fluo_func_optim",
                                                "results", "n_molecs_optim_rand_params")) + "/"

    np.save(save_path + "n_molecules_guess_values.npy", n_molecule_guess_values)
    np.save(save_path + "objective_values.npy", objective_values)
