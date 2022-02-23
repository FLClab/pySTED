import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from pysted import base, utils
from pysted import exp_data_gen as dg

from gym_sted.rewards.objectives_timed import find_nanodomains, Signal_Ratio, Resolution
from gym_sted.utils import get_foreground
import metrics


"""
faire des acqs back to back avec les params optimaux et voir le f1 score à chaque acq
"""


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

# CES PARAMS LÀ ONT LAIR GOOD, IL ME RESTE À CHECKER LE BLEACH FUTUR SI IL SE COMPORTE COMME JE VOUDRAIS
# fluo params dict to modify and test
egfp_x10 = {
    # wavelength of the fluorophores, doesn't seem to change much if we stay close-ish (very large) to the beam
    # wavelengths, but going too far (i.e. 100e-9) renders the fluorophores much less "potent"
    "lambda_": 535e-9,
    "qy": 0.6,   # increasing increases the number of photons, decreasing decreases it
    "sigma_abs": {
        # beam - molecule interaction for excitation
        # modifying 575 (STED) doesn't seem to have much impact?
        # increasing 488 (EXC) increases number of photns, while decreasing it decreases the number of photons
        488: 0.03e-22,
        575: 0.02e-21
    },
    "sigma_ste": {
        # beam - molecule interaction for STED
        # decreasing the value decreases the STED effect, making the img more confocal for same STED power
        # increasing the value increases STED effect without increasing photobleaching cause by STED
        575: 2.0e-22,
    },
    "sigma_tri": 10.14e-21,
    "tau": 3e-09,   # decreasing reduces STED effect while leaving its photobleaching the same, increasing does ?
    "tau_vib": 1.0e-12,   # decreasing reduces STED effect while leaving its photobleaching the same, increasing does ?
    "tau_tri": 1.2e-6,   # decreasing decreases photobleaching, increasing increases photobleaching ?
    "phy_react": {
        488: 0.0008e-6,   # photobleaching caused by exc beam, lower = less photobleaching
        575: 0.00185e-8    # photobleaching cuased by sted beam, lower = less photobleaching
    },
    "k_isc": 0.48e+6,
}

egfp_x100 = {
    # wavelength of the fluorophores, doesn't seem to change much if we stay close-ish (very large) to the beam
    # wavelengths, but going too far (i.e. 100e-9) renders the fluorophores much less "potent"
    "lambda_": 535e-9,
    "qy": 0.6,   # increasing increases the number of photons, decreasing decreases it
    "sigma_abs": {
        # beam - molecule interaction for excitation
        # modifying 575 (STED) doesn't seem to have much impact?
        # increasing 488 (EXC) increases number of photns, while decreasing it decreases the number of photons
        488: 0.0275e-23,
        575: 0.02e-21
    },
    "sigma_ste": {
        # beam - molecule interaction for STED
        # decreasing the value decreases the STED effect, making the img more confocal for same STED power
        # increasing the value increases STED effect without increasing photobleaching cause by STED
        575: 2.0e-22,
    },
    "sigma_tri": 10.14e-21,
    "tau": 3e-09,   # decreasing reduces STED effect while leaving its photobleaching the same, increasing does ?
    "tau_vib": 1.0e-12,   # decreasing reduces STED effect while leaving its photobleaching the same, increasing does ?
    "tau_tri": 1.2e-6,   # decreasing decreases photobleaching, increasing increases photobleaching ?
    "phy_react": {
        488: 0.0008e-6,   # photobleaching caused by exc beam, lower = less photobleaching
        575: 0.00185e-8   # photobleaching cuased by sted beam, lower = less photobleaching
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

fluo_x10 = base.Fluorescence(**egfp_x10)
microscope_x10 = base.Microscope(laser_ex, laser_sted, detector, objective, fluo_x10, load_cache=True)
i_ex, i_sted, _ = microscope_x10.cache(pixelsize, save_cache=True)

fluo_x100 = base.Fluorescence(**egfp_x100)
microscope_x100 = base.Microscope(laser_ex, laser_sted, detector, objective, fluo_x100, load_cache=True)
i_ex, i_sted, _ = microscope_x100.cache(pixelsize, save_cache=True)

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

shroom_x100 = dg.Synapse(n_molecs_base * multiplier ** 2, mode="mushroom", seed=42)
shroom_x100.add_nanodomains(10, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain_base * multiplier ** 2,
                            seed=42, valid_thickness=7)
dmap_x100 = base.TemporalDatamap(shroom_x100.frame, pixelsize, shroom_x100)
dmap_x100.set_roi(i_ex, "max")

nd_gt_positions = np.copy(np.array(dmap_base.synapses.nanodomains_coords))

n_acqs = 5
acqs = []
bleached = []
fig, axes = plt.subplots(2, n_acqs)
dmap_copy = np.copy(dmap_x100.whole_datamap[dmap_x100.roi])
for i in range(n_acqs):
    print(f"acq {i+1} of {n_acqs}")
    acq, bleached_dmap, _ = microscope_x100.get_signal_and_bleach(dmap_x100, dmap_x100.pixelsize,
                                                             **optim_params,
                                                             bleach=True, update=True, bleach_mode="proportional")

    acqs.append(np.copy(acq))
    bleached.append(np.copy(bleached_dmap["base"][dmap_base.roi]))

    # COMPUTE LE F1_SCORE ICI :)
    nd_guess_positions = find_nanodomains(acq, dmap_base.pixelsize)
    detector = metrics.CentroidDetectionError(nd_gt_positions, nd_guess_positions, 2,
                                              algorithm="hungarian")
    f1_score = detector.f1_score
    photobleaching = (np.sum(dmap_copy) - np.sum(bleached[i])) / np.sum(dmap_copy)

    imshow_acq = axes[0, i].imshow(acqs[i])
    axes[0, i].set_title(f"acq {i}, \n f1_score = {f1_score}")
    fig.colorbar(imshow_acq, ax=axes[0, i], fraction=0.05, pad=0.05)

    imshow_bleached = axes[1, i].imshow(bleached[i])
    axes[1, i].set_title(f"bleached {i}, \n photobleaching = {round(photobleaching, 4)}")
    fig.colorbar(imshow_bleached, ax=axes[1, i], fraction=0.05, pad=0.05)

plt.show()