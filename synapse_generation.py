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

func_fiber, func_polygons = utils.generate_fiber_with_synapses(image.shape, 100, 200, 5, 10, polygon_scale=(3, 6))

n_frames = 1
roi = ((0, 0), image.shape)   # jtrouve que la façon de gérer la shape de l'ensemble est weird
ensemble_test2 = temporal.Ensemble(roi=roi)
ensemble_test2.append(func_fiber)
for polygon in func_polygons:
    ensemble_test2.append(polygon)

frame2 = ensemble_test2.return_frame()
plt.imshow(frame2)
plt.show()