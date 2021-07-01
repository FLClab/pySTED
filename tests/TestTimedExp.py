import numpy as np
import tqdm
from pysted import base, utils
import time
import os
import argparse
from matplotlib import pyplot as plt
import sys

# save_path = os.path.join(os.path.expanduser('~'), "Documents", "research", "NeurIPS", "exp_runtimes")
# test = np.zeros((10, 10))
# np.save(save_path + "/yo", test)
# exit()

time_quantum_us = 1
master_clock = base.Clock(time_quantum_us)

time_start = time.time()
exp_time = 1000000   # we want our experiment to last 1000000 us, or 1s
for i in tqdm.trange(exp_time):
    master_clock.update_time()
print(f"took {time.time() - time_start} s to run {exp_time} time steps of only time bank additions")
