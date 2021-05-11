"""
The goal in this file is to play around with anthony's code for synapse generation, see what I can do :)
"""

import math
import numpy as np
import random
from matplotlib import pyplot as plt
import tqdm
from pysted import base, utils, temporal
from scipy.spatial.distance import cdist

#--------------------------------------- Fiber + Polygons --------------------------------------------------------------
"""
image = np.zeros((256, 256))

# si je comprends bien, je dois créer un obj Ensemble, ainsi que des objets Fiber et Polygon
# Je dois aussi comprendre comment un obj NodeCombiner et Synapse fonctionne

# create fiber from random func?
# faire ça comme ça ici permet de m'assurer que ma fibre sera tjrs dans l'image :)
min_nodes, max_nodes = 100, 200
min_array, max_array = np.asarray((min_nodes, min_nodes)), np.asarray((max_nodes, max_nodes))
# jpense jpeux controller l'angle aussi
fibre_rand_test = temporal.Fiber(random_params={"num_points": (min_nodes, max_nodes),
                                                "pos": [np.zeros((1, 2)) + min_array,
                                                        image.shape - max_array],
                                                "scale": (1, 5)})

func_fiber, func_polygons = utils.generate_fiber_with_synapses(image.shape, 100, 200, 5, 10, polygon_scale=(5, 10))

n_frames = 1
roi = ((0, 0), image.shape)   # jtrouve que la façon de gérer la shape de l'ensemble est weird
ensemble_test2 = temporal.Ensemble(roi=roi)
ensemble_test2.append(func_fiber)
for polygon in func_polygons:
    ensemble_test2.append(polygon)

frame2 = ensemble_test2.return_frame()
plt.imshow(frame2)
plt.show()
"""
#--------------------------------------- Fiber + fibers ----------------------------------------------------------------
"""
# ici je veux plotter une fibre centrale et essayer d'ajouter des plus petites fibres perpendiculaires sur la fibre

image = np.zeros((256, 256))

min_nodes, max_nodes = 100, 200
min_array, max_array = np.asarray((min_nodes, min_nodes)), np.asarray((max_nodes, max_nodes))
fibre_rand = temporal.Fiber(random_params={"num_points": (min_nodes, max_nodes),
                                                "pos": [np.zeros((1, 2)) + min_array,
                                                        image.shape - max_array],
                                                "scale": (1, 5)})

sec_fibers = utils.generate_secondary_fibers(image.shape, fibre_rand, 5, 3, sec_len=(10, 20))

n_frames = 1
roi = ((0, 0), image.shape)   # jtrouve que la façon de gérer la shape de l'ensemble est weird
ensemble_test = temporal.Ensemble(roi=roi)
ensemble_test.append(fibre_rand)
for sec_fiber in sec_fibers:
    ensemble_test.append(sec_fiber)

frame = ensemble_test.return_frame()
plt.imshow(frame)
plt.show()
"""
#--------------------------------------- Fiber + sec fibers + synapses -------------------------------------------------
"""
# ici je veux générer une fibre et trouver comment je peux générer les sous fibres perpendiculaires-ish à la principale

image = np.zeros((256, 256))

# generate the main fiber
# si je mets 200, 256 ça fait que des fois il finit jamais, ? pt pcq 256 >= image.shape
# il faut que je mette max_nodes < image.shape sinon ça fuck
# aussi je pense que ce que je fais live ne marchera pas sur une image pas carrée?
# cette partie là faudrait que je repasse dessus c'est weird
min_nodes, max_nodes = 200, 255   # si je mets 200, 300 ça fait que des fois il finit jamais, ? pt pcq 300 > image.shape
min_array, max_array = np.asarray((min_nodes, min_nodes)), np.asarray((max_nodes, max_nodes))
fibre_rand = temporal.Fiber(random_params={"num_points": (min_nodes, max_nodes),
                                                "pos": [np.zeros((1, 2)) + min_array,
                                                        image.shape - max_array],
                                                "scale": (1, 5)})

# generate the secondary fibers branching out from the main fiber
sec_fibers = utils.generate_secondary_fibers(image.shape, fibre_rand, 5, 3, sec_len=(10, 20))

# generate synapses attached to the secondary fibers
synapses_lists = []
for secondary_fiber in sec_fibers:
    ith_fiber_synapses = utils.generate_synapses_on_fiber(image.shape, secondary_fiber, 2, 1, synapse_scale=(5, 5))
    synapses_lists.append(ith_fiber_synapses)

n_frames = 1
roi = ((0, 0), image.shape)   # jtrouve que la façon de gérer la shape de l'ensemble est weird
ensemble_test = temporal.Ensemble(roi=roi)
ensemble_test.append(fibre_rand)
for idx, sec_fiber in enumerate(sec_fibers):
    ensemble_test.append(sec_fiber)
    for synapse in synapses_lists[idx]:
        ensemble_test.append(synapse)

frame = ensemble_test.return_frame()

plt.imshow(frame)
plt.show()
"""
#--------------------------------------- Testing encompasing function --------------------------------------------------
"""
# testing my generation function
# des fois il fait juste jamais finir? Je sais pas trop pourquoi, mais mon hypothèse c'est
# qu'il essait de placer des trucs out of bounds et il reste pris dans une loop infinie
# je pourrais pt ajouter de quoi que genre s'il fait + que X iter de la loop pour placer une
# fibre / synapse et qu'il ne parvient pas il fait juste give up ou qqchose
ensemble_func, synapses_list = utils.generate_synaptic_fibers((256, 256), (100, 255), (3, 10), (2, 5))

func_frame = ensemble_func.return_frame()

synapses_ensemble = temporal.Ensemble(roi=((0, 0), (256, 256)))
for sub_list in synapses_list:
    for synapse in sub_list:
        synapses_ensemble.append(synapse)

synapses_frame = synapses_ensemble.return_frame()

third_frame_test = np.zeros(func_frame.shape)
synapses_dict = ensemble_func.generate_objects_dict(obj_type="synapses")
for i, key in enumerate(synapses_dict):
    rr, cc = synapses_dict[key].return_shape(shape=third_frame_test.shape)
    third_frame_test[rr.astype(int), cc.astype(int)] = i

fig, axes = plt.subplots(1, 3)

all_frame = axes[0].imshow(func_frame)
axes[0].set_title(f"Fibers + Synapses")
fig.colorbar(all_frame, ax=axes[0], fraction=0.04, pad=0.05)

synapses_frame = axes[1].imshow(synapses_frame)
axes[1].set_title(f"Synapses")
fig.colorbar(synapses_frame, ax=axes[1], fraction=0.04, pad=0.05)

third_frame = axes[2].imshow(third_frame_test)
axes[2].set_title(f"Adding synapses manually")
fig.colorbar(third_frame, ax=axes[2], fraction=0.04, pad=0.05)

plt.show()
"""
#--------------------------------------- Adding Ca2+ Flashes -----------------------------------------------------------

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
