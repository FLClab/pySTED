import numpy as np
from matplotlib import pyplot as plt
import tqdm
from pysted import base, utils, temporal


# le but ici est d'ajouter l'élément temporel (les flash de Ca2+)  à l'élément spatial. C'est une bonne opportunité
# pour continuer à faire des fonctions englobantes pour mes tests

# Get light curves stuff to generate the flashes later
event_file_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
video_file_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"

# Generate a datamap
frame_shape = (100, 100)
ensemble_func, synapses_list = utils.generate_synaptic_fibers(frame_shape, (25, 74), (3, 10), (2, 5))

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

# Build a dictionnary corresponding synapses to a bool saying if they are currently flashing or not
# They all start not flashing
flat_synapses_list = [item for sublist in synapses_list for item in sublist]
synpase_flashing_dict, synapse_flash_idx_dict, synapse_flash_curve_dict, isolated_synapses_frames = {}, {}, {}, {}

for idx_syn in range(len(flat_synapses_list)):
    synpase_flashing_dict[idx_syn] = False
    synapse_flash_idx_dict[idx_syn] = 0
    rr, cc = flat_synapses_list[idx_syn].return_shape(shape=poils_frame.shape)
    isolated_synapses_frames[idx_syn] = np.zeros(frame_shape).astype(int)
    isolated_synapses_frames[idx_syn][rr.astype(int), cc.astype(int)] += 5


flash_prob = 0.05   # every iteration, all synapses will have a 5% to start flashing

# start acquisition loop
frozen_datamap = np.copy(datamap.whole_datamap[datamap.roi])
len_sequence = 100
save_path = "D:/SCHOOL/Maitrise/H2021/Recherche/data_generation/multiple_random_flashes/test1/"
list_datamaps, list_confocals, list_steds = [], [], []
for i in tqdm.trange(len_sequence):
    print(f"acq {i + 1} of {len_sequence}")
    datamap.whole_datamap[datamap.roi] = np.copy(frozen_datamap)  # essayer np.copy?

    # loop through all synapses, make some start to flash, randomly, maybe
    for idx_syn in range(len(flat_synapses_list)):
        if np.random.binomial(1, flash_prob) and synpase_flashing_dict[idx_syn] is False:
            # can start the flash
            synpase_flashing_dict[idx_syn] = True
            synapse_flash_idx_dict[idx_syn] = 1
            sampled_curve = utils.flash_generator_old(event_file_path, video_file_path)
            synapse_flash_curve_dict[idx_syn] = utils.rescale_data(sampled_curve, to_int=True, divider=3)

        if synpase_flashing_dict[idx_syn]:
            datamap.whole_datamap[datamap.roi] -= isolated_synapses_frames[idx_syn]
            datamap.whole_datamap[datamap.roi] += isolated_synapses_frames[idx_syn] * \
                                                  synapse_flash_curve_dict[idx_syn][synapse_flash_idx_dict[idx_syn]]
            synapse_flash_idx_dict[idx_syn] += 1
            if synapse_flash_idx_dict[idx_syn] >= 40:
                synapse_flash_idx_dict[idx_syn] = 0
                synpase_flashing_dict[idx_syn] = False

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
