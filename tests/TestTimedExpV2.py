import numpy as np
import tqdm
from pysted import base, utils, exp_data_gen
import time
import os
import argparse
from matplotlib import pyplot as plt
import sys
import argparse

# parser = argparse.ArgumentParser(description="Example of experiment script")
# parser.add_argument("--save_path", type=str, default="", help="Where to save the files")
# args = parser.parse_args()
#
# save_path = utils.make_path_sane(args.save_path)
# if not os.path.exists(save_path):
#     os.mkdir(save_path)

hand_crafted_light_curve = utils.hand_crafted_light_curve(delay=2, n_decay_steps=10, n_molecules_multiplier=14)

test_random_agent = base.RandomActionSelector(0, 0, 0)

for i in range(100):
    test_random_agent.select_action()

