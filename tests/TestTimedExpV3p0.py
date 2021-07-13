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
exp_time = 500000   # testing with bleach is much longer :)

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
p_ex_val = 1e-6
p_sted_val = 30e-3
pdt_val = 100e-6

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
i_ex, _, _ = microscope.cache(pixelsize, save_cache=True)
# temporal_synapse_dmap = base.TemporalSynapseDmap(shroom.frame, datamap_pixelsize=20e-9, synapse_obj=shroom)
# temporal_synapse_dmap.set_roi(i_ex, intervals='max')
molecs_disp = np.ones((64, 64))
temporal_dmap = base.TestTemporalDmap(molecs_disp, datamap_pixelsize=20e-9)
temporal_dmap.set_roi(i_ex, intervals='max')

decay_time_us = 1000000   # 1 seconde
# temporal_synapse_dmap.create_t_stack_dmap(decay_time_us)
temporal_dmap.create_t_stack_dmap(decay_time_us)

agent = base.RandomActionSelector(pdt_val, p_ex_val, p_sted_val, temporal_dmap.whole_datamap[temporal_dmap.roi].shape)
temporal_exp = base.TemporalExperimentV2p0(microscope, temporal_dmap)

acqs, dmaps, actions = temporal_exp.launch_experiment(exp_time, agent)

print(f"acqs shape = {acqs.shape}")
print(f"dmaps shape = {dmaps.shape}")
print(f"len actions = {len(actions)}")

np.save(save_path + "/test_acqs_v3p0", acqs)
np.save(save_path + "/test_dmaps_v3p0", dmaps)
textfile = open(save_path + "/test_selected_actions_v3p0.txt", "w")
for action in actions:
    textfile.write(action + "\n")
textfile.close()
