import os
import numpy as np
from matplotlib import pyplot as plt

from pysted import base, utils
from pysted import exp_data_gen as dg

from gym_sted.rewards.objectives_timed import find_nanodomains, Signal_Ratio, Resolution
from gym_sted.utils import get_foreground
import metrics


"""
plan pour cette fig:
Montrer une datamap, une acq confocale et une acq sted sur cette datamap, la datamap bleaché pour
chaque acq
"""

def rescale_func(val_to_scale, current_range_min, current_range_max, new_min, new_max):
    return ( (val_to_scale - current_range_min) / (current_range_max - current_range_min) ) * (new_max - new_min) + new_min


path_to_save_dir = os.path.join(os.path.expanduser('~'), "Documents", "research", "NeurIPS", "aaai_paper", "data_gen",
                                "data_for_fig_1_v1", "b_v1_data")

data_to_save_dict = {
    "datamaps before": "/datamaps_before_acq",
    "nd gt": "/nd_gt_positions",
    "confoc acq": "/confocal_acquisition",
    "sted acq": "/sted_acquisitions",
    "nd guess confocal": "/nd_guess_positions_confocal",
    "nd guess sted": "/nd_guess_positions_sted",
    "f1_scores confocal": "/f1_scores_confocal",
    "f1_scores sted": "/f1_scores_sted",
    "% resolved nd confocal": "/ratio_resolved_nd_confocal",
    "% resolved nd sted": "/ratio_resolved_nd_sted",
    "datamaps after confocal": "/datamaps_after_confocal",
    "datamaps after sted": "/datamaps_after_sted",
    "photobleaching confocal": "/photobleaching_confocal",
    "photobleaching sted": "/photobleaching_sted",
    "SNR confocal": "/snr_confocal",
    "SNR sted": "/snr_sted",
    "resolution confocal": "/resolution_confocal",
    "resolution sted": "/resolution_sted",
}

# for key in data_to_save_dict.keys():
#     if not os.path.exists(path_to_save_dir + data_to_save_dict[key]):
#         os.makedirs(path_to_save_dir + data_to_save_dict[key])

print("Setting up the microscope ...")
# Microscope stuff
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
    "k_isc": 0.48e+6
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
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
i_ex, i_sted, _ = microscope.cache(pixelsize, save_cache=True)

confoc_ratio, sted_ratio = 0.6, 0.25
pdt_mult = 1

mode = "same"

if mode == "diff":
    conf_params = {
        "p_ex": 25.0e-6,
        "pdt": pdt_mult * 10.0e-6,
        "p_sted": 0.0
    }
    save_dir = path_to_save_dir + "/diff_params"
elif mode == "same":
    conf_params = {
        "p_ex": confoc_ratio * action_spaces["p_ex"]["high"],
        "pdt": pdt_mult * 10.0e-6,
        "p_sted": 0.0
    }
    save_dir = path_to_save_dir + "/same_params"

sted_params = {
    # "p_ex": 25.0e-6,
    "p_ex": confoc_ratio * action_spaces["p_ex"]["high"],
    "pdt": pdt_mult * 10.0e-6,
    "p_sted": sted_ratio * action_spaces["p_sted"]["high"]
}

print(action_spaces["p_ex"]["high"])
print(0.25 * action_spaces["p_sted"]["high"])
exit()

datamaps_before, datamaps_after_confoc, datamaps_after_sted = [], [], []
nd_gt_positions, nd_guess_positions_confoc, nd_guess_positions_sted = [], [], []
acq_confocals, acq_steds = [], []
f1_scores_confoc, f1_scores_sted, ratio_resolved_nds_confoc, ratio_resolved_nds_sted = [], [], [], []
# photobleachings_confoc, snrs_confoc, resolutions_confoc = [], [], []
# photobleachings_sted, snrs_sted, resolutions_sted = [], [], []

n_repetitions = 10   # I want num = 10, 1 for test
seeds = [s for s in range(n_repetitions)]
for i in range(n_repetitions):
    print(f"starting acq {i+1} of {n_repetitions}")

    # first step is to generate the datamap
    shroom = dg.Synapse(5, mode="mushroom", seed=seeds[i])

    n_molecs_in_domain = 135
    min_dist = 50
    shroom.add_nanodomains(10, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain, seed=seeds[i],
                           valid_thickness=7)
    # shroom.rotate_and_translate()

    dmap = base.TemporalDatamap(shroom.frame, pixelsize, shroom)
    dmap.set_roi(i_ex, "max")

    datamaps_before.append(np.copy(dmap.whole_datamap[dmap.roi]))
    nd_gt_positions.append(np.copy(np.array(dmap.synapses.nanodomains_coords)))

    confoc_acq, confoc_bleached, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                                      bleach=True, update=False, seed=seeds[i])
    confoc_bleached = confoc_bleached["base"][dmap.roi]

    acq_confocals.append(np.copy(confoc_acq))
    datamaps_after_confoc.append(np.copy(confoc_bleached))

    nd_guess_positions_confoc.append(find_nanodomains(confoc_acq, dmap.pixelsize))

    detector_confoc = metrics.CentroidDetectionError(nd_gt_positions[i], nd_guess_positions_confoc[i], 2,
                                                     algorithm="hungarian")
    f1_scores_confoc.append(detector_confoc.f1_score)
    ratio_resolved_nds_confoc.append(detector_confoc.true_positive / nd_gt_positions[i].shape[0])

    sted_acq, sted_bleached, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **sted_params,
                                                                  bleach=True, update=False, seed=seeds[i])
    sted_bleached = sted_bleached["base"][dmap.roi]

    acq_steds.append(np.copy(sted_acq))
    datamaps_after_sted.append(np.copy(sted_bleached))

    nd_guess_positions_sted.append(find_nanodomains(sted_acq, dmap.pixelsize))

    detector_sted = metrics.CentroidDetectionError(nd_gt_positions[i], nd_guess_positions_sted[i], 2,
                                                   algorithm="hungarian")
    f1_scores_sted.append(detector_sted.f1_score)
    ratio_resolved_nds_sted.append(detector_sted.true_positive / nd_gt_positions[i].shape[0])

    # fig, axes = plt.subplots(2, 3)
    #
    # axes[0, 0].imshow(dmap.whole_datamap[dmap.roi], cmap='hot')
    # axes[0, 0].set_title(f"Datamap before acquisition")
    #
    # axes[0, 1].imshow(confoc_acq, cmap='hot', vmin=0, vmax=np.max(confoc_acq))
    # axes[0, 1].set_title(f"Confocal acquisition")
    #
    # axes[0, 2].imshow(confoc_bleached, cmap='hot')
    # axes[0, 2].set_title(f"Datamap after confocal")
    #
    # axes[1, 0].imshow(dmap.whole_datamap[dmap.roi], cmap='hot')
    #
    # axes[1, 1].imshow(sted_acq, cmap='hot', vmin=0, vmax=np.max(confoc_acq))
    # axes[1, 1].set_title(f"STED acquisition")
    #
    # axes[1, 2].imshow(sted_bleached, cmap='hot')
    # axes[1, 2].set_title(f"Datamap after STED")
    #
    # plt.show()
    # plt.close(fig)

# datamaps_before, datamaps_after_confoc, datamaps_after_sted = [], [], []
# nd_gt_positions, nd_guess_positions_confoc, nd_guess_positions_sted = [], [], []
# acq_confocals, acq_steds = [], []
# f1_scores_confoc, f1_scores_sted, ratio_resolved_nds_confoc, ratio_resolved_nds_sted = [], [], [], []

datamaps_before = np.asarray(datamaps_before)
datamaps_after_confoc = np.asarray(datamaps_after_confoc)
datamaps_after_sted = np.asarray(datamaps_after_sted)
nd_gt_positions = np.asarray(nd_gt_positions)
nd_guess_positions_confoc = np.asarray(nd_guess_positions_confoc)
nd_guess_positions_sted = np.asarray(nd_guess_positions_sted)
acq_confocals = np.asarray(acq_confocals)
acq_steds = np.asarray(acq_steds)
f1_scores_confoc = np.asarray(f1_scores_confoc)
f1_scores_sted = np.asarray(f1_scores_sted)
ratio_resolved_nds_confoc = np.asarray(ratio_resolved_nds_confoc)
ratio_resolved_nds_sted = np.asarray(ratio_resolved_nds_sted)

np.save(save_dir + "/datamaps_before.npy", datamaps_before)
np.save(save_dir + "/datamaps_after_confoc.npy", datamaps_after_confoc)
np.save(save_dir + "/datamaps_after_sted.npy", datamaps_after_sted)
np.save(save_dir + "/nd_gt_positions.npy", nd_gt_positions)
np.save(save_dir + "/nd_guess_positions_confoc.npy", nd_guess_positions_confoc)
np.save(save_dir + "/nd_guess_positions_sted.npy", nd_guess_positions_sted)
np.save(save_dir + "/acq_confocals.npy", acq_confocals)
np.save(save_dir + "/acq_steds.npy", acq_steds)
np.save(save_dir + "/f1_scores_confoc.npy", f1_scores_confoc)
np.save(save_dir + "/f1_scores_sted.npy", f1_scores_sted)
np.save(save_dir + "/ratio_resolved_nds_confoc.npy", ratio_resolved_nds_confoc)
np.save(save_dir + "/ratio_resolved_nds_sted.npy", ratio_resolved_nds_sted)

