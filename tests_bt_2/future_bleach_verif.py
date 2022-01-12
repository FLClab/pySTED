import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from pysted import base, utils
from pysted import exp_data_gen as dg

from gym_sted.rewards.objectives_timed import find_nanodomains, Signal_Ratio, Resolution
from gym_sted.utils import get_foreground
import metrics


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
confocal_params = {
    "pdt": 10e-6,
    "p_ex": 0.05e-3,
    "p_sted": 0.0
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
n_molecs_in_domain_base = 0   # 135
min_dist = 50
multiplier = 10

shroom_x10 = dg.Synapse(n_molecs_base * multiplier, mode="mushroom", seed=42)
shroom_x10.add_nanodomains(10, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain_base * multiplier, seed=42,
                           valid_thickness=7)
dmap_x10 = base.TemporalSynapseDmap(shroom_x10.frame, pixelsize, shroom_x10)
dmap_x10.set_roi(i_ex, "max")
dmap_x10.create_t_stack_dmap_smooth(2000000, delay=2)

shroom_x100 = dg.Synapse(n_molecs_base * multiplier ** 2, mode="mushroom", seed=42)
shroom_x100.add_nanodomains(10, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain_base * multiplier ** 2,
                            seed=42, valid_thickness=7)
dmap_x100 = base.TemporalSynapseDmap(shroom_x100.frame, pixelsize, shroom_x100)
dmap_x100.set_roi(i_ex, "max")
dmap_x100.create_t_stack_dmap_smooth(2000000, delay=2)

mode = "x10"
if mode == "x10":
    chosen_dmap = dmap_x10
    chosen_microscope = microscope_x10
elif mode == "x100":
    chosen_dmap = dmap_x100
    chosen_microscope = microscope_x100
else:
    chosen_dmap = None
    chosen_microscope = None

acqs = []
for t in range(chosen_dmap.flash_tstack.shape[0]):
    for i in range(7):
        chosen_dmap.update_whole_datamap(i)
        chosen_dmap.update_dicts({"flashes": i})
        plt.imshow(chosen_dmap.whole_datamap[chosen_dmap.roi])
        plt.title(f"t = {i},"
                  f"\n max = {np.max(chosen_dmap.whole_datamap[chosen_dmap.roi])}")
        plt.show()

    chosen_dmap.update_whole_datamap(t)
    chosen_dmap.update_dicts({"flashes": t})

    acq, bleached, _ = chosen_microscope.get_signal_and_bleach(chosen_dmap, chosen_dmap.pixelsize, **confocal_params,
                                                               bleach=True, update=True, bleach_mode="proportional")

    acqs.append(acq)

    fig, axes = plt.subplots(1, len(acqs))
    for j, acq_to_show in enumerate(acqs):
        if len(acqs) > 1:
            axes[j].imshow(acq_to_show, vmin=0, vmax=np.max(np.asarray(acqs)))
            axes[j].set_title(f"t = {j}")
            fig.suptitle(f"ratio between 2 latest acqs = {np.sum(acqs[-1]) / np.sum(acqs[-2])}")
        else:
            axes.imshow(acq_to_show, vmin=0, vmax=np.max(np.asarray(acqs)))
            axes.set_title(f"t = {j}")
    plt.show()
    plt.close(fig)

