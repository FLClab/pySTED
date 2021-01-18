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
