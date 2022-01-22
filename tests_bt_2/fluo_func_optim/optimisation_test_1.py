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


# path I use on my laptop, this needs to be modified depending on the device you are running the script on
DATA_MAIN = os.path.expanduser(os.path.join("~", "Documents", "research", "h22_code", "albert_beads_exps",
                               "bandit-optimization-experiments", "2021-09-14_grid_articacts"))
DATA_SUBDIRS = [
    "/four_params_five_reps_2timesDwell/", "/four_params_five_reps_smallerDwell/",
    "/four_params_five_reps_smallerDwell_day2/", "/four_params_five_reps_tetraspec/",
    "/four_params_five_reps_tetraspec_smaller_dwell/"
]

def fluo_param_optim_1(optim_param_vect):
    """
    version 1 of the function :
    This function optimizes the sigma_abs (exc) and phy_react (exc and sted) params of the fluorescence paramters

    This function needs to take as entry the fluorophore parameters optimize and returns a scalar which needs to be
    optimized (1 - A*B type beat)
    :param 1: A vector ordered by [sigma_abs (exc), phy_react (exc), phy_react (sted)]

    ... if I understand correctly, I can't add other args to the function to optimize than the vector of params
        this means I will have to recode this function if I'm running it on another device for the paths to work,
        and I need to code 4 versions of it :
            [sigma_abs + phy_react; sigma_abs + phy_react + qy; sigma_abs + phy_react + sigma_ste;
             sigma_abs + phy_react + qy + sigma_ste]
    ... There must be a better way ?

    ~*! this link is important !*~
    https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html
    """
    global DATA_MAIN
    global DATA_SUBDIRS

    # load a random acquisition and params
    data_paths = [DATA_MAIN + data_subdir for data_subdir in DATA_SUBDIRS]
    real_acquisition = microscopy_random_data_loader(data_paths)

    molecules_disposition = generate_dmap_from_real_img(real_acquisition["sted"])

    egfp = {
        "lambda_": 635e-9,  # TODO: verify ok to change like that...
        "qy": 0.6,  # COPIED FROM BEFORE
        "sigma_abs": {
            635: optim_param_vect[0],  # Table S3, Oracz et al., nature 2017   ALBERT USES 0.1e-21
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
            635: optim_param_vect[1],  # ALBERT USES 0.008e-5
            750: optim_param_vect[2],  # ALBERT USES 0.008e-8
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
    dmap = base.TemporalDatamap(molecules_disposition, pixelsize, None)
    dmap.set_roi(i_ex, "max")

    # params with which the real image was taken
    p_ex_max = 100 * 1.5135e-6
    p_sted_max = 100 * 1.7681e-3

    # The real life params with which the sampled image was acquired
    alb_params = {
        "pdt": real_acquisition["pdt"],
        "p_ex": 0.01 * real_acquisition["p_ex"] * p_ex_max,
        "p_sted": 0.01 * real_acquisition["p_sted"] * p_sted_max
    }

    conf_params = {  # I should ask Albert what the params for conf1 / conf2 were
        "pdt": 20e-6,
        "p_ex": 0.15 * p_ex_max,
        "p_sted": 0.0
    }

    pixel_list = line_step_pixel_list_builder(dmap, line_step=real_acquisition["line_step"])

    # acquire a confocal before and after, acquire an image with the same param as the real image was acquired with
    simul_conf1, _, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                         bleach=False, update=False)
    simul_sted, simul_bleached, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **alb_params,
                                                                     bleach=True, update=True,
                                                                     pixel_list=pixel_list, filter_bypass=True)
    simul_conf2, _, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                         bleach=False, update=False)

    simul_conf1_fg_bool = get_foreground(simul_conf1)
    simul_conf1_fg = simul_conf1 * simul_conf1_fg_bool
    simul_conf2_fg_bool = get_foreground(simul_conf2)
    simul_conf2_fg = simul_conf2 * simul_conf2_fg_bool

    conf1_fg_bool = get_foreground(real_acquisition["conf1"])
    conf1_fg = real_acquisition["conf1"] * conf1_fg_bool
    conf2_fg_bool = get_foreground(real_acquisition["conf2"])
    conf2_fg = real_acquisition["conf2"] * conf2_fg_bool

    photobleaching_real = 100 * np.sum(conf2_fg) / np.sum(conf1_fg)
    photobleaching_simul = 100 * np.sum(simul_conf2_fg) / np.sum(simul_conf1_fg)

    photoncount_real = np.sum(real_acquisition["sted"])
    photoncount_simul = np.sum(simul_sted)

    photobleaching_obj = 1 - (np.abs(photobleaching_real - photobleaching_simul) / photobleaching_real)
    photoncount_obj = 1 - (np.abs(photoncount_real - photoncount_simul) / photoncount_real)

    # VOIR CONVO DISCORD AVEC ANTHONY POUR SAVOIR WHAT TO DO NEXT

    # est-ce que -1 * A * B causerait des problèmes si les 2 objectifs sont négatifs? what would that mean/imply ??
    objective = -1 * photoncount_obj * photobleaching_obj
    print(f"photobleaching_REAL = {photobleaching_real}, photobleaching_SIMUL = {photobleaching_simul}")
    print(f"photobleaching objective = {photobleaching_obj}")
    print(f"--------------------------------------------------------------------------------------------")
    print(f"photoncount_REAL = {photoncount_real}, photoncount_SIMUL = {photoncount_simul}")
    print(f"photoncount objective = {photoncount_obj}")
    print(f"--------------------------------------------------------------------------------------------")
    print(f"objective = {objective}")

    return objective


if __name__ == "__main__":
    obj_test = fluo_param_optim_1([0.42e-22, 0.002e-7, 0.002e-10])

    # optim = gp_minimize(
    #     fluo_param_optim_1,
    #     [(0.42e-22, 0.42e-20), (0.002e-9, 0.002e-5), (0.002e-12, 0.002e-8)],
    #     acq_func="EI",
    #     n_calls=2,
    #     n_random_starts=1
    # )

    # print(optim)