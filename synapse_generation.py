"""
The goal in this file is to play around with anthony's code for synapse generation, see what I can do :)
"""

import numpy as np
import random
from matplotlib import pyplot as plt
import tqdm
from pysted import base, utils, temporal

image = np.zeros((256, 256))

# si je comprends bien, je dois créer un obj Ensemble, ainsi que des objets Fiber et Polygon
# Je dois aussi comprendre comment un obj NodeCombiner et Synapse fonctionne

# create fiber from random func?
fibre_rand_test = temporal.Fiber(random_params={"num_points": (30, 50),
                                                "pos": [(0, 0), image.shape]})

n_polygones = 5

sampled_nodes = random.sample(list(fibre_rand_test.nodes_position), n_polygones)

# create polygon from random func?
polygon_rand_test = temporal.Polygon(random_params={"pos": [(0, 0), image.shape]})

# create a synapse
# l'obj Synapse semple servir à faire pousser des polygones sur des fibres
# dans mon cas, pour l'instant, on travaille sur des datamaps statiques dans le temps
# (rien qui pousse, mais des flashs / déplacements de calcium)
# donc je ne pense pas que je devrais travailler avec un objet synapse
# synapse_test = temporal.Synapse()

polygon_rand_list = []
for node in sampled_nodes:
    polygon = temporal.Polygon(random_params={"pos": [node, node]})
    polygon_rand_list.append(polygon)

n_frames = 1
roi = ((0, 0), image.shape)   # jtrouve que la façon de gérer la shape de l'ensemble est weird
ensemble_test = temporal.Ensemble(roi=roi)
ensemble_test.append(fibre_rand_test)
for polygon in polygon_rand_list:
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
