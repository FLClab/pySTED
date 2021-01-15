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
func_frame = utils.generate_synaptic_fibers((256, 256), (100, 255), (3, 10), (2, 5))
# func_frame = utils.generate_synaptic_fibers((256, 256), (200, 255), 5, 2)

plt.imshow(func_frame)
plt.show()
