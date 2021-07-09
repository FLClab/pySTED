import numpy as np
import tqdm
from pysted import base, utils, exp_data_gen
import time
import os
import argparse
from matplotlib import pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser(description="Example of experiment script")
parser.add_argument("--save_path", type=str, default="", help="Where to save the files")
args = parser.parse_args()

save_path = utils.make_path_sane(args.save_path)
if not os.path.exists(save_path):
    os.mkdir(save_path)

hand_crafted_light_curve = utils.hand_crafted_light_curve(delay=2, n_decay_steps=10, n_molecules_multiplier=14)


time_quantum_us = 1
master_clock = base.Clock(time_quantum_us)
# exp_time = 1000000   # we want our experiment to last 1000000 us, or 1s
exp_time = 500000   # testing with bleach is much longer :)

light_curves_path = f"flash_files/events_curves.npy"
shroom = exp_data_gen.Synapse(5, mode="mushroom", seed=42)
n_molecs_in_domain = 0
min_dist = 100
shroom.add_nanodomains(40, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain, seed=42, valid_thickness=3)
shroom.frame = shroom.frame.astype(int)

print("Setting up the microscope ...")
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
        "phy_react": {488: 1e-6,   # 1e-4
                      575: 1e-10},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 20e-9
bleach = False
p_ex = 1e-6
p_sted = 30e-3
pdt = 10e-6

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
i_ex, _, _ = microscope.cache(pixelsize, save_cache=True)
temporal_synapse_dmap = base.TemporalSynapseDmap(shroom.frame, datamap_pixelsize=20e-9, synapse_obj=shroom)
temporal_synapse_dmap.set_roi(i_ex, intervals='max')

decay_time_us = 1000000   # 1 seconde
temporal_synapse_dmap.create_t_stack_dmap(decay_time_us)

action_selector = base.RandomActionSelector(pdt, p_ex, p_sted)
temporal_exp = base.TemporalExperiment(master_clock, microscope, temporal_synapse_dmap)

acqs, dmaps, actions = temporal_exp.launch_experiment(exp_time, action_selector)
print(f"dmaps_array.shape = {dmaps.shape}")
print(f"confocal_acquisitions.shape = {acqs.shape}")
print(f"len(selected_actions) = {len(actions)}")

