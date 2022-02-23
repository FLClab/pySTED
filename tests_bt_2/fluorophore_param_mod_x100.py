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
PLAN
STEP 1 :
redo the code that allows me compute F1-score for various params
    - run it for the 'optimal' params, results should be approx same as in paper figure
STEP 2 :
Modify the script :
    - runs multiple times on a static dmap, looks at photobleaching, photon count and f1-score at each acq
    - run on default dmap params, default fluo params
    - then multiply dmap params by 10, and attempt to modify fluo params such that photon count of 1st img
      stays similar, but photobleaching is reduced
Once this works, modify to run temporally?
"""

def rescale_func(val_to_scale, current_range_min, current_range_max, new_min, new_max):
    return ( (val_to_scale - current_range_min) / (current_range_max - current_range_min) ) * (new_max - new_min) + new_min

print("Setting up the microscope ...")
# Microscope stuff

# DO NOT MODIFY THE VALUES IN THIS DICT!
# Modify the values in egfp to test stuff out :)
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

# these are the fluo params I have selected for when the base amount of molecules is 500 and the peaks are 14000 (x100)
egfp = {
    "lambda_": 535e-9,
    "qy": 0.6,
    "sigma_abs": {
        488: 0.0275e-23,
        575: 0.02e-21
    },
    "sigma_ste": {
        575: 2.0e-22,
    },
    "sigma_tri": 10.14e-21,
    "tau": 3e-09,
    "tau_vib": 1.0e-12,
    "tau_tri": 1.2e-6,
    "phy_react": {
        488: 0.0008e-6,
        575: 0.00185e-8
    },
    "k_isc": 0.48e+6,
}

if egfp == default_egfp:
    print("using default fluo params")
else:
    print("testing new fluo params")

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

conf_params = {
    "P_EX": 25.0e-6,
    "PDT": 10.0e-6,
    "P_STED": 0.0
}

multipliers = np.array([0., 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.])
# multipliers = np.array([0., 1.])

pdt_values = multipliers * action_spaces["pdt"]["high"]
pdt_values = rescale_func(pdt_values, np.min(pdt_values), np.max(pdt_values),
                          action_spaces["pdt"]["low"], action_spaces["pdt"]["high"])
p_ex_values = multipliers * action_spaces["p_ex"]["high"]
p_sted_values = multipliers * action_spaces["p_sted"]["high"]

# pdt_values = np.array([10.0e-6])
# p_ex_values = np.array([0.25e-3])
# p_sted_values = np.array([87.5e-3])

fluo_params_save_dir = "./tests_bt_2/save_dir"
if not os.path.exists(fluo_params_save_dir):
    os.mkdir(fluo_params_save_dir)

n_acqs_per_rep = 1
n_repetitions = 10   # I want num = 10, 2 for test
f1_score_values = np.zeros((pdt_values.shape[0], p_ex_values.shape[0], p_sted_values.shape[0], n_repetitions))
n_molecules_values = np.zeros((pdt_values.shape[0], p_ex_values.shape[0], p_sted_values.shape[0], n_repetitions, n_acqs_per_rep + 1))
n_photons_values = np.zeros((pdt_values.shape[0], p_ex_values.shape[0], p_sted_values.shape[0], n_repetitions))
seeds = [s for s in range(n_repetitions)]
for pdt_idx, pdt in enumerate(pdt_values):
    for pex_idx, p_ex in enumerate(p_ex_values):
        for psted_idx, p_sted in enumerate(p_sted_values):
            print(f"starting {n_repetitions} acquisitions with pdt = {pdt}, p_ex = {p_ex}, p_sted = {p_sted}")
            nd_gt_positions, nd_guess_positions = [], []
            for i in range(n_repetitions):
                n_molecs = []
                molec_mult = 100
                # first step is to generate the datamap
                shroom = dg.Synapse(5 * molec_mult, mode="mushroom", seed=seeds[i])

                n_molecs_in_domain = 135 * molec_mult
                min_dist = 50
                shroom.add_nanodomains(10, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain, seed=seeds[i],
                                       valid_thickness=7)
                # shroom.rotate_and_translate()

                dmap = base.TemporalDatamap(shroom.frame, pixelsize, shroom)
                dmap.set_roi(i_ex, "max")
                nd_gt_positions.append(np.copy(np.array(dmap.synapses.nanodomains_coords)))
                dmap_copy = np.copy(dmap.whole_datamap[dmap.roi])

                # for snr / resolution evaluation
                conf1, bleached_conf1, _ = microscope.get_signal_and_bleach(
                    dmap, dmap.pixelsize, conf_params["PDT"], conf_params["P_EX"], conf_params["P_STED"],
                    bleach=False, update=False, seed=seeds[i]
                )

                n_molecs_init = np.sum(dmap.whole_datamap)
                n_molecs.append(n_molecs_init)

                acq, bleached_dict, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, pdt, p_ex, p_sted,
                                                                         bleach=True, update=True, seed=seeds[i])

                # fig, axes = plt.subplots(1, 3)
                #
                # axes[0].imshow(dmap_copy)
                # axes[1].imshow(acq)
                # axes[2].imshow(dmap.whole_datamap[dmap.roi])
                #
                # plt.show()

                nd_guess_positions.append(find_nanodomains(acq, dmap.pixelsize))

                detector = metrics.CentroidDetectionError(nd_gt_positions[i], nd_guess_positions[i], 2,
                                                          algorithm="hungarian")
                f1_score_values[pdt_idx, pex_idx, psted_idx, i] = detector.f1_score
                n_photons_values[pdt_idx, pex_idx, psted_idx, i] = np.sum(acq)

                n_molecs_post = np.sum(dmap.whole_datamap)
                n_molecs.append(n_molecs_post)

                n_molecules_values[pdt_idx, pex_idx, psted_idx, i, :] = np.asarray(n_molecs)

                # foreground on confocal image
                fg_c = get_foreground(conf1)
                # foreground on sted image
                if np.any(acq):
                    fg_s = get_foreground(acq)
                else:
                    fg_s = np.ones_like(fg_c)
                # remove STED foreground points not in confocal foreground, if any
                fg_s *= fg_c

                conf2, bleached_conf1, _ = microscope.get_signal_and_bleach(
                    dmap, dmap.pixelsize, conf_params["PDT"], conf_params["P_EX"], conf_params["P_STED"],
                    bleach=False, update=False, seed=seeds[i]
                )

            # iirc ça c'est pour save les positions dans un array pour que ça aille une taille bien def
            # max_n_guesses = 0
            # for guesses in nd_guess_positions:
            #     if guesses.shape[0] > max_n_guesses:
            #         max_n_guesses = guesses.shape[0]
            # for idx, guesses in enumerate(nd_guess_positions):
            #     if guesses.shape[0] < max_n_guesses and guesses.shape[0] != 0:
            #         n_appends = max_n_guesses - guesses.shape[0]
            #         vals_to_append = []
            #         for j in range(n_appends):
            #             vals_to_append.append([999, 999])
            #         guesses = np.append(guesses, vals_to_append, axis=0)
            #     elif guesses.shape[0] < max_n_guesses and guesses.shape[0] == 0:
            #         guesses = []
            #         n_appends = max_n_guesses
            #         for j in range(n_appends):
            #             guesses.append([999, 999])
            #         guesses = np.asarray(guesses)
            #     nd_guess_positions[idx] = guesses

f1_score_values_avg = np.squeeze(np.mean(f1_score_values, axis=-1))
n_photons_values_avg = np.squeeze(np.mean(n_photons_values, axis=-1))
n_molecules_values_avg = np.squeeze(np.mean(n_molecules_values, axis=-2))

# save the data in an appropriate dir
if egfp == default_egfp:
    # save stuff in fluo_params_save_dir + "/fluo_default", along with saving the dict of default fluo params
    save_dir = fluo_params_save_dir + "/fluo_default"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
else:
    # save stuff in a dir named fluo_params_save_dir + "/fluo_<X>", along with saving the dict of used fluo params
    existing_save_dirs = [x[0] for x in os.walk(fluo_params_save_dir) if 'fluo_' in x[0] and 'default' not in x[0]]
    existing_dir_indices = [int(x[-1]) for x in existing_save_dirs]
    if len(existing_dir_indices) == 0:
        save_dir_idx = 1
    else:
        save_dir_idx = np.max(existing_dir_indices) + 1
    save_dir = fluo_params_save_dir + f"/fluo_{save_dir_idx}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

# ... I think the way I am saving my data is bad, I am waiting too long with too much data loaded before saving it,
# this will take up too much RAM...
# save the og arrays and the avg to fluo_params_save_dir
np.save(save_dir + "/f1_score_values_baseline.npy", f1_score_values)
np.save(save_dir + "/f1_score_values_avg_baseline.npy", f1_score_values_avg)
np.save(save_dir + "/n_photons_values_baseline.npy", n_photons_values)
np.save(save_dir + "/n_photons_values_avg_baseline.npy", n_photons_values_avg)
np.save(save_dir + "/n_molecules_values_baseline.npy", n_molecules_values)
np.save(save_dir + "/n_molecules_values_avg_baseline.npy", n_molecules_values_avg)
dict_file = open(save_dir + "/fluo_params.pkl", "wb")
pickle.dump(egfp, dict_file)
dict_file.close()

print(f1_score_values_avg)
print(n_photons_values_avg)
print(n_molecules_values_avg)

