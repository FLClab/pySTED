"""
The goal in this file is to play around with anthony's code for synapse generation, see what I can do :)
"""

import numpy as np
import random
from matplotlib import pyplot as plt
import tqdm
from pysted import base, utils, temporal
from scipy.spatial.distance import cdist

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

# error message pops up sometimes, maybe adding the out of bounds protection will fix that :)
n_polygones = 5
min_distance = 10
n_added = 0
sampled_nodes = np.empty((0, 2))
while n_added != n_polygones:
    sampled_node = np.asarray(random.sample(list(fibre_rand_test.nodes_position), 1)[0].astype(int))
    if np.less_equal(sampled_node, 0).any() or np.greater_equal(sampled_node, image.shape - np.ones((1, 1))).any():
        continue
    # print(sampled_node)
    # exit()
    if n_added == 0:
        sampled_nodes = np.append(sampled_nodes, sampled_node)
        sampled_nodes = np.expand_dims(sampled_nodes, 0).astype(int)
        n_added += 1
        continue
    # comparer la distance du point samplé à tous les points dans la liste
    # vérifier qu'elle est plus grande que min_distance pour tous les points déjà présents,
    # si c'est le cas, l'ajouter à la liste, sinon continuer le while :)
    else:
        sample_to_verify = np.expand_dims(np.copy(sampled_node), axis=0).astype(int)
        sampled_nodes = np.append(sampled_nodes, sample_to_verify, axis=0).astype(int)
        distances = cdist(sampled_nodes, sampled_nodes)
        distances[n_added, n_added] = min_distance + 1
        if np.less_equal(distances[n_added, :], min_distance).any():
            # at least 1 elt is closer than 10 pixels to an already present elt so remove it :)
            sampled_nodes = np.delete(sampled_nodes, n_added, axis=0)
        else:
            # good to add to the list
            n_added += 1


polygon_rand_list = []
for node in sampled_nodes:
    polygon = temporal.Polygon(random_params={"pos": [node, node],
                                              "scale": (5, 10)})
    polygon_rand_list.append(polygon)

n_frames = 1
roi = ((0, 0), image.shape)   # jtrouve que la façon de gérer la shape de l'ensemble est weird
ensemble_test = temporal.Ensemble(roi=roi)
ensemble_test.append(fibre_rand_test)
for polygon in polygon_rand_list:
    # pass
    ensemble_test.append(polygon)

frame = ensemble_test.return_frame()
plt.imshow(frame)
plt.show()

# print("begins")
# frames = ensemble_test.generate_sequence(n_frames)
# print("ends")
#
# for frame in frames:
#     plt.imshow(frame)
#     plt.show()
#
# print("bitch?")
