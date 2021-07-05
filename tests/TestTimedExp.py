import numpy as np
import tqdm
from pysted import base, utils, exp_data_gen
import time
import os
import argparse
from matplotlib import pyplot as plt
import sys

# save_path = os.path.join(os.path.expanduser('~'), "Documents", "research", "NeurIPS", "exp_runtimes")

hand_crafted_light_curve = utils.hand_crafted_light_curve(delay=2, n_decay_steps=10, n_molecules_multiplier=14)

plt.plot(hand_crafted_light_curve)
plt.title(f"shape = {hand_crafted_light_curve.shape}")
plt.show()
exit()

time_quantum_us = 1
master_clock = base.Clock(time_quantum_us)
exp_time = 10000000   # we want our experiment to last 1000000 us, or 1s

light_curves_path = f"flash_files/events_curves.npy"
shroom = exp_data_gen.Synapse(5, mode="mushroom", seed=42)
n_molecs_in_domain = 10
min_dist = 100
shroom.add_nanodomains(40, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain, seed=42, valid_thickness=3)
for nanodomain in shroom.nanodomains:
    nanodomain.add_flash_curve(light_curves_path, seed=42)

flash_step = 0
# print(normalized_light_curve[flash_step])
for i in tqdm.trange(exp_time):
    master_clock.update_time()
    shroom.flash_nanodomains(master_clock.current_time, master_clock.time_quantum_us)
