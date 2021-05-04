import numpy as np
from matplotlib import pyplot as plt
import tqdm
from pysted import base, utils, temporal


# le but ici est d'ajouter l'élément temporel (les flash de Ca2+)  à l'élément spatial. C'est une bonne opportunité
# pour continuer à faire des fonctions englobantes pour mes tests

# Get light curves stuff to generate the flashes later
event_file_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
video_file_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"

sampled_curve = utils.flash_generator_old(event_file_path, video_file_path)

# Generate a datamap
ensemble_func, synapses_list = utils.generate_synaptic_fibers((256, 256), (100, 255), (3, 10), (2, 5))

poils_frame = ensemble_func.return_frame().astype(int)

# Microscope stuff
egfp = {"lambda_": 535e-9,
        "qy": 0.6,
        "sigma_abs": {488: 1.15e-20,
                      575: 6e-21},
        "sigma_ste": {560: 1.2e-20,
                      575: 6.0e-21,
                      580: 5.0e-21},
        "sigma_tri": 1e-21,
        "tau": 3e-09,
        "tau_vib": 1.0e-12,
        "tau_tri": 5e-6,
        "phy_react": {488: 1e-4,   # 1e-4
                      575: 1e-8},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 10e-9
dpxsz = 10e-9
bleach = False
p_ex = 1e-6
p_sted = 30e-3
pdt = 10e-6
size = 64 + (2 * 22 + 1)
roi = 'max'
seed = True

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
# zero_residual controls how much of the donut beam "bleeds" into the the donut hole
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
# noise allows noise on the detector, background adds an average photon count for the empty pixels
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap = base.Datamap(poils_frame, dpxsz)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, bleach_func="default_bleach")
i_ex, _, _ = microscope.cache(datamap.pixelsize)
datamap.set_roi(i_ex, roi)

# randomly select 1 synapse that I will flash
flat_synapses_list = [item for sublist in synapses_list for item in sublist]
random_synapse = flat_synapses_list[np.random.randint(0, len(flat_synapses_list))]
rr, cc = random_synapse.return_shape(shape=poils_frame.shape)
isolated_synapse = np.copy(datamap.whole_datamap[datamap.roi]).astype(int)
isolated_synapse -= poils_frame
isolated_synapse[rr.astype(int), cc.astype(int)] += 5

# acquire the images during the flash :)
datamap.whole_datamap = datamap.whole_datamap.astype(int)
sampled_curve = utils.rescale_data(sampled_curve, to_int=True, divider=3)   # ??? ajouter ça à ma func de light curve sinon ça marchera pas
len_sequence = len(sampled_curve)
frozen_datamap = np.copy(datamap.whole_datamap[datamap.roi])
save_path = "D:/SCHOOL/Maitrise/H2021/Recherche/data_generation/flashing_generated_map/test1/"
list_datamaps, list_confocals, list_steds = [], [], []
for i in tqdm.trange(len_sequence):
    # multiplier 1 synapse par la light curve
    datamap.whole_datamap[datamap.roi] = np.copy(frozen_datamap)   # essayer np.copy?
    datamap.whole_datamap[datamap.roi] -= isolated_synapse
    datamap.whole_datamap[datamap.roi] += isolated_synapse * sampled_curve[i]

    roi_save_copy = np.copy(datamap.whole_datamap[datamap.roi])

    confoc_acq, _ = microscope.get_signal_and_bleach_fast(datamap, datamap.pixelsize, pdt, p_ex, 0.0,
                                                          pixel_list=None, bleach=bleach, update=False)
    sted_acq, _ = microscope.get_signal_and_bleach_fast(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
                                                        pixel_list=None, bleach=bleach, update=False)

    list_datamaps.append(roi_save_copy)
    list_confocals.append(confoc_acq)
    list_steds.append(sted_acq)

min_datamap, max_datamap = np.min(list_datamaps), np.max(list_datamaps)
min_confocal, max_confocal = np.min(list_confocals), np.max(list_confocals)
min_sted, max_sted = np.min(list_steds), np.max(list_steds)
for idx in range(len(list_datamaps)):
    plt.imsave(save_path + f"datamaps/{idx}.png", list_datamaps[idx], vmin=min_datamap, vmax=max_datamap)
    plt.imsave(save_path + f"confocals/{idx}.png", list_confocals[idx], vmin=min_confocal, vmax=max_confocal)
    plt.imsave(save_path + f"steds/{idx}.png", list_steds[idx], vmin=min_sted, vmax=max_sted)
