import os
import numpy as np
from matplotlib import pyplot as plt

from pysted import base, utils
from pysted import exp_data_gen as dg

from gym_sted.rewards.objectives_timed import find_nanodomains, Signal_Ratio, Resolution
from gym_sted.utils import get_foreground
import metrics


"""
Voir pages 5-7 de mon doc "Papier aaai" sur mon ipad, j'explique ce que je veux
 - Pour la fig C, je veux faire un cube de données montrant le % de NDs résolus / f1_score obtenu en fonction de
   pdt, p_ex, p_sted pour un phy_react fixe
   
 - Pour la fig D, je veux choisir la best combinaison (pdt, p_ex) selon ce que j'ai obtenu pour la figure C ^
   et montrer comment les 3 objectifs (SNR, résolution, Bleach) varient en fonction de p_sted. Je ferais différentes
   courbes pour différents phy_react
   
Comme D dépend de C, je vais le scripte pour C dans ce fichier. 
*** JE VEUX TOUT ENREGISTRER ***
 - datamps générées
    - positions des NDs
 - images acquises
 - pdt, p_ex, p_sted utilisés
 - % NDs résolus
    - positions des guess de NDs
 - f1_score
 - datamaps après l'acq
 - valeurs photobleaching, SNR, résolution obtenus
 - am I forgetting something ?
 
structure d'enregistrement des données :
 - faire un folder pour chaque chose que j'ai listé en haut
 - enregistrer le data dans le folder approprié (xd), avec un nom du genre
    <DATA>_pdt_<PDT VALUE>_pex_<P_EX VALUE>_psted_<P_STED VALUE>.npy
 - je vais répeter (?) fois pour chaque tuple de params, donc enregistrer ^ comme des np arrays de (?)
"""

def rescale_func(val_to_scale, current_range_min, current_range_max, new_min, new_max):
    return ( (val_to_scale - current_range_min) / (current_range_max - current_range_min) ) * (new_max - new_min) + new_min


path_to_save_dir = os.path.join(os.path.expanduser('~'), "Documents", "research", "NeurIPS", "aaai_paper", "data_gen",
                                "data_for_fig_1_v1", "c_v1_data")

# make directories for all the data that will be saved
data_to_save_dict = {
    "datamaps before": "/datamaps_before_acq",
    "nd gt": "/nd_gt_positions",
    "acquired images": "/acquired_images",
    "% resolved nd": "/ratio_resolved_nd",
    "nd guess": "/nd_guess_positions",
    "f1_scores": "/f1_scores",
    "datamaps after": "/datamaps_after_acq",
    "photobleaching": "/photobleaching",
    "SNR": "/snr",
    "resolution": "/resolution",
    "confocals before": "/confocals_before",
    "confocals after": "/confocals_after"
}

for key in data_to_save_dict.keys():
    if not os.path.exists(path_to_save_dir + data_to_save_dict[key]):
        os.makedirs(path_to_save_dir + data_to_save_dict[key])

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

conf_params = {
    "P_EX": 25.0e-6,
    "PDT": 10.0e-6,
    "P_STED": 0.0
}


# vérifier avec Anthony si ça semble être une bonne discrétisation :)
# pdt_values = np.linspace(action_spaces["pdt"]["low"], action_spaces["pdt"]["high"], 2)   # I want num = 11, 2 for test
multipliers = np.array([0, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.])
pdt_values = multipliers * action_spaces["pdt"]["high"]
pdt_values = rescale_func(pdt_values, np.min(pdt_values), np.max(pdt_values),
                          action_spaces["pdt"]["low"], action_spaces["pdt"]["high"])
p_ex_values = multipliers * action_spaces["p_ex"]["high"]
p_sted_values = multipliers * action_spaces["p_sted"]["high"]

# pdt_values = [15e-6]
# p_ex_values = [0.0001]
# p_sted_values = [0.05]

n_repetitions = 10   # I want num = 10, 2 for test
seeds = [s for s in range(n_repetitions)]
for pdt in pdt_values:
    for p_ex in p_ex_values:
        for p_sted in p_sted_values:
            print(f"starting {n_repetitions} acquisitions with pdt = {pdt}, p_ex = {p_ex}, p_sted = {p_sted}")
            datamaps_before, datamaps_after = [], []
            nd_gt_positions, nd_guess_positions = [], []
            acquired_images = []
            f1_scores, ratio_resolved_nds = [], []
            photobleachings, snrs, resolutions = [], [], []
            confocals_before, confocals_after = [], []
            for i in range(n_repetitions):

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

                # for snr / resolution evaluation
                conf1, bleached_conf1, _ = microscope.get_signal_and_bleach(
                    dmap, dmap.pixelsize, conf_params["PDT"], conf_params["P_EX"], conf_params["P_STED"],
                    bleach=False, update=False, seed=seeds[i]
                )

                n_molecs_init = np.sum(dmap.whole_datamap)

                acq, bleached_dict, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, pdt, p_ex, p_sted,
                                                                         bleach=True, update=True, seed=seeds[i])

                n_molecs_post = np.sum(dmap.whole_datamap)

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
                confocals_before.append(conf1)
                confocals_after.append(conf2)

                acquired_images.append(np.copy(acq))
                nd_guess_positions.append(find_nanodomains(acq, dmap.pixelsize))

                detector = metrics.CentroidDetectionError(nd_gt_positions[i], nd_guess_positions[i], 2,
                                                          algorithm="hungarian")
                f1_scores.append(detector.f1_score)
                ratio_resolved_nds.append(detector.true_positive / nd_gt_positions[i].shape[0])
                datamaps_after.append(np.copy(dmap.whole_datamap[dmap.roi]))

                # reste juste le photobleaching, SNR, résolution
                photobleaching = 1 - n_molecs_post / n_molecs_init
                snr = snr_evaluator.evaluate(acq, conf1, fg_s, fg_c, n_molecs_init, n_molecs_post,
                                             dmap)
                resolution = resolution_evaluator.evaluate(acq, conf1, fg_s, fg_c, n_molecs_init, n_molecs_post,
                                                    dmap)

                photobleachings.append(photobleaching)
                snrs.append(snr)
                resolutions.append(resolution)

            max_n_guesses = 0
            for guesses in nd_guess_positions:
                if guesses.shape[0] > max_n_guesses:
                    max_n_guesses = guesses.shape[0]
            for idx, guesses in enumerate(nd_guess_positions):
                if guesses.shape[0] < max_n_guesses and guesses.shape[0] != 0:
                    n_appends = max_n_guesses - guesses.shape[0]
                    vals_to_append = []
                    for j in range(n_appends):
                        vals_to_append.append([999, 999])
                    guesses = np.append(guesses, vals_to_append, axis=0)
                elif guesses.shape[0] < max_n_guesses and guesses.shape[0] == 0:
                    guesses = []
                    n_appends = max_n_guesses
                    for j in range(n_appends):
                        guesses.append([999, 999])
                    guesses = np.asarray(guesses)
                nd_guess_positions[idx] = guesses

            # exit()

            # convert into npy arrays, save to the right place with the right name :)
            datamaps_before = np.asarray(datamaps_before)
            datamaps_after = np.asarray(datamaps_after)
            nd_gt_positions = np.asarray(nd_gt_positions)
            nd_guess_positions = np.asarray(nd_guess_positions)
            acquired_images = np.asarray(acquired_images)
            f1_scores = np.asarray(f1_scores)
            ratio_resolved_nds = np.asarray(ratio_resolved_nds)
            photobleachings = np.asarray(photobleachings)
            snrs = np.asarray(snrs)
            resolutions = np.asarray(resolutions)
            confocals_before = np.asarray(confocals_before)
            confocals_after = np.asarray(confocals_after)

            np.save(path_to_save_dir + data_to_save_dict["datamaps before"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", datamaps_before)
            np.save(path_to_save_dir + data_to_save_dict["datamaps after"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", datamaps_after)
            np.save(path_to_save_dir + data_to_save_dict["nd gt"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", nd_gt_positions)
            np.save(path_to_save_dir + data_to_save_dict["nd guess"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", nd_guess_positions)
            np.save(path_to_save_dir + data_to_save_dict["acquired images"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", acquired_images)
            np.save(path_to_save_dir + data_to_save_dict["f1_scores"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", f1_scores)
            np.save(path_to_save_dir + data_to_save_dict["% resolved nd"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", ratio_resolved_nds)
            np.save(path_to_save_dir + data_to_save_dict["photobleaching"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", photobleachings)
            np.save(path_to_save_dir + data_to_save_dict["SNR"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", snrs)
            np.save(path_to_save_dir + data_to_save_dict["resolution"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", resolutions)
            np.save(path_to_save_dir + data_to_save_dict["confocals before"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", confocals_before)
            np.save(path_to_save_dir + data_to_save_dict["confocals after"] +
                    f"/pdt_{pdt}_pex_{p_ex}_psted_{p_sted}.npy", confocals_after)

print("~*! ---------------- !*~")
print(f"all done without crashing ^_^")
print("~*! ---------------- !*~")
