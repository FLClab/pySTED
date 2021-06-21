import numpy as np
import tqdm
from pysted import base, utils
import os
import argparse
from matplotlib import pyplot as plt


seed = 52
curves_path = "flash_files/events_curves.npy"
# Generate a datamap
frame_shape = (64, 64)
ensemble_func, synapses_list = utils.generate_synaptic_fibers(frame_shape, (9, 55), (3, 10), (2, 5),
                                                              seed=seed)
# Build a dictionnary corresponding synapses to a bool saying if they are currently flashing or not
# They all start not flashing
flat_synapses_list = [item for sublist in synapses_list for item in sublist]

poils_frame = ensemble_func.return_frame().astype(int)

save_path = r"D:\SCHOOL\Maitrise\E2021\research\iid_pres\figs\simulated_data"
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(poils_frame, cmap="inferno", vmax=10)
plt.show()
# fig.savefig(save_path + f"/dmap_seed_{seed}.png")