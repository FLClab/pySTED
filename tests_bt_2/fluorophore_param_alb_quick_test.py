import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from pysted import base, utils
from pysted import exp_data_gen as dg

from gym_sted.rewards.objectives_timed import find_nanodomains, Signal_Ratio, Resolution
from gym_sted.utils import get_foreground
import metrics


def rescale_func(val_to_scale, current_range_min, current_range_max, new_min, new_max):
    return ( (val_to_scale - current_range_min) / (current_range_max - current_range_min) ) * (new_max - new_min) + new_min

print("Setting up the microscope ...")
# Microscope stuff

egfp = {
#     "lambda": 535e-9,
    "lambda_": 635e-9, # TODO: verify ok to change like that...
    "qy": 0.6, # COPIED FROM BEFORE
    "sigma_abs": {
        635: 0.1e-21, #Table S3, Oracz et al., nature 2017
        750: 3.5e-25,  # (1 photon exc abs) Table S3, Oracz et al., nature 2017
    },
    "sigma_ste": {
        750: 4.8e-22, #Table S3, Oracz et al., nature 2017
    },
    "sigma_tri": 10.14e-21, # COPIED FROM BEFORE
#     "tau": 3e-09,
    "tau" : 3.5e-9, # @646nm, ATTO Fluorescent labels, ATTO-TEC GmbH catalog 2016/2018
    "tau_vib": 1.0e-12, #t_vib, Table S3, Oracz et al., nature 2017
    "tau_tri": 1.2e-6, # COPIED FROM BEFORE
    "phy_react": {
#         488: 0.008e-5,
#         575: 0.008e-8,
        635: 0.008e-5, # COPIED FROM BEFORE
        750:  0.008e-8, # COPIED FROM BEFORE
    },
    "k_isc": 0.48e+6 # COPIED FROM BEFORE
}

action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350.0e-3},
    "p_ex" : {"low" : 0., "high" : 250.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}

pixelsize = 20e-9

snr_evaluator = Signal_Ratio(percentile=75)
resolution_evaluator = Resolution(pixelsize=pixelsize)

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(635e-9)
laser_sted = base.DonutBeam(750e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
i_ex, i_sted, _ = microscope.cache(pixelsize, save_cache=True)

conf_params = {
    "P_EX": 25.0e-6,
    "PDT": 10.0e-6,
    "P_STED": 0.0
}
optim_params = {
    "pdt": 10e-6,
    "p_ex": 0.25e-3,
    "p_sted": 87.5e-3
}

multiplier = 1
shroom = dg.Synapse(5 * multiplier, mode="mushroom", seed=42)

n_molecs_in_domain = 135
min_dist = 50
shroom.add_nanodomains(10, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain * multiplier, seed=42,
                       valid_thickness=7)

dmap = base.TemporalSynapseDmap(shroom.frame, pixelsize, shroom)
dmap.set_roi(i_ex, "max")

acq, bleached, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **optim_params,
                                                    bleach=True, update=False)

fig, axes = plt.subplots(1, 3)

dmap_before_imshow = axes[0].imshow(dmap.whole_datamap[dmap.roi], vmax=np.max(dmap.whole_datamap))
axes[0].set_title(f"Datamap before acquisition")
fig.colorbar(dmap_before_imshow, ax=axes[0], fraction=0.05, pad=0.05)

acq_imshow = axes[1].imshow(acq, cmap="hot")
axes[1].set_title(f"Acquired signal")
fig.colorbar(acq_imshow, ax=axes[1], fraction=0.05, pad=0.05)

dmap_after_imshow = axes[2].imshow(bleached["base"][dmap.roi], vmax=np.max(dmap.whole_datamap))
axes[2].set_title(f"Datamap after acquisition")
fig.colorbar(dmap_after_imshow, ax=axes[2], fraction=0.05, pad=0.05)

plt.show()
plt.close(fig)
