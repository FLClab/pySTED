import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from pysted import base, utils
from pysted import exp_data_gen as dg

from gym_sted.rewards.objectives_timed import find_nanodomains, Signal_Ratio, Resolution
from gym_sted.utils import get_foreground
import metrics


default_egfp = {
    "lambda_": 535e-9,
    "qy": 0.6,
    "sigma_abs": {
        488: 0.08e-21,
        575: 0.02e-21
    },
    "sigma_ste": {
        575: 3.0e-22,
    },
    "sigma_tri": 10.14e-21,
    "tau": 3e-09,
    "tau_vib": 1.0e-12,
    "tau_tri": 1.2e-6,
    "phy_react": {
        488: 0.008e-5,
        575: 0.008e-8
    },
    "k_isc": 0.48e+6
}

# fluo params dict to modify and test
egfp = {
    "lambda_": 535e-9,
    "qy": 0.6,
    "sigma_abs": {
        488: 0.08e-21,
        575: 0.02e-21
    },
    "sigma_ste": {
        575: 3.0e-22,
    },
    "sigma_tri": 10.14e-21,
    "tau": 3e-09,
    "tau_vib": 1.0e-12,
    "tau_tri": 1.2e-6,
    "phy_react": {
        488: 0.008e-5,
        575: 0.008e-8
    },
    "k_isc": 0.48e+6,
}

optim_params = {
    "pdt": 10e-6,
    "p_ex": 0.25e-3,
    "p_sted": 87.5e-3
}

pixelsize = 20e-9
# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()

fluo_default = base.Fluorescence(**default_egfp)
microscope_default = base.Microscope(laser_ex, laser_sted, detector, objective, fluo_default, load_cache=True)
i_ex, i_sted, _ = microscope_default.cache(pixelsize, save_cache=True)

fluo_test = base.Fluorescence(**egfp)
microscope_test = base.Microscope(laser_ex, laser_sted, detector, objective, fluo_test, load_cache=True)

n_molecs_base = 5
n_molecs_in_domain_base = 135
min_dist = 50
multiplier = 10

shroom_base = dg.Synapse(n_molecs_base, mode="mushroom", seed=42)
shroom_base.add_nanodomains(10, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain_base, seed=42,
                            valid_thickness=7)
dmap_base = base.TemporalDatamap(shroom_base.frame, pixelsize, shroom_base)
dmap_base.set_roi(i_ex, "max")

shroom_x10 = dg.Synapse(n_molecs_base * multiplier, mode="mushroom", seed=42)
shroom_x10.add_nanodomains(10, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain_base * multiplier, seed=42,
                           valid_thickness=7)
dmap_x10 = base.TemporalDatamap(shroom_x10.frame, pixelsize, shroom_x10)
dmap_x10.set_roi(i_ex, "max")

acq_default, bleached_default, _ = microscope_default.get_signal_and_bleach(dmap_base, dmap_base.pixelsize,
                                                                            **optim_params,
                                                                            bleach=True, update=False)

acq_x10, bleached_x10, _ = microscope_test.get_signal_and_bleach(dmap_x10, dmap_x10.pixelsize,
                                                                 **optim_params,
                                                                 bleach=True, update=False)

fig, axes = plt.subplots(2, 2)

def_acq_imshow = axes[0, 0].imshow(acq_default)
axes[0, 0].set_title(f"Default parameters")
fig.colorbar(def_acq_imshow, ax=axes[0, 0], fraction=0.05, pad=0.05)

x10_acq_imshow = axes[0, 1].imshow(acq_x10)
axes[0, 1].set_title(f"x10 parameters")
fig.colorbar(x10_acq_imshow, ax=axes[0, 1], fraction=0.05, pad=0.05)

def_bleached_imshow = axes[1, 0].imshow(bleached_default["base"][dmap_base.roi])
fig.colorbar(def_bleached_imshow, ax=axes[1, 0], fraction=0.05, pad=0.05)

x10_bleached_imshow = axes[1, 1].imshow(bleached_x10["base"][dmap_x10.roi])
fig.colorbar(x10_bleached_imshow, ax=axes[1, 1], fraction=0.05, pad=0.05)

plt.show()
