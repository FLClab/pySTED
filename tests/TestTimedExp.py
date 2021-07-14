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
# exp_time = 819200   # testing with bleach is much longer :)
exp_time = 300000

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
pdt_val = 10e-6
# pdt_val = np.random.randint(0, 100, size=(64, 64)) * 1e-6

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
i_ex, _, _ = microscope.cache(pixelsize, save_cache=True)
molecs_disp = np.ones((64, 64))
shroom = exp_data_gen.Synapse(5, mode="mushroom", seed=42)
n_molecs_in_domain = 0
min_dist = 100
shroom.add_nanodomains(40, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain, seed=42, valid_thickness=3)
shroom.frame = shroom.frame.astype(int)

# here I want to make it look like multiple episodes are happening and in each episode an experiment that lasts exp_time
# is launched
agent = base.RandomActionSelector(pdt_val, p_ex_val, p_sted_val, roi_shape=molecs_disp.shape)
n_episodes = 1
for n in range(n_episodes):
    action_counter = 0
    # create an experiment object with a new clock and datamap
    # temporal_dmap = base.TestTemporalDmap(molecs_disp, datamap_pixelsize=20e-9)
    temporal_dmap = base.TemporalSynapseDmap(shroom.frame, datamap_pixelsize=20e-9, synapse_obj=shroom)
    temporal_dmap.set_roi(i_ex, intervals='max')

    decay_time_us = 1000000  # 1 seconde
    temporal_dmap.create_t_stack_dmap(decay_time_us)
    temporal_dmap.update_whole_datamap(0)

    clock = base.Clock(time_quantum_us=time_quantum_us)

    temporal_exp = base.TemporalExperiment(clock, microscope, temporal_dmap, exp_runtime=exp_time, bleach=True)
    acquisitions = []
    actions = []
    dmaps_after_actions = []
    while clock.current_time < temporal_exp.exp_runtime:
        agent.select_action()
        actions.append(agent.action_selected)
        acq = temporal_exp.play_action(agent.current_action_pdt, agent.current_action_p_ex, agent.current_action_p_sted)
        dmaps_after_actions.append(np.copy(temporal_dmap.whole_datamap))
        acquisitions.append(np.copy(acq))
        action_counter += 1
        print(f"!!!!!!! {action_counter} action(s) completed !!!!!!!")

    for i in range(len(acquisitions)):
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(acquisitions[i])
        axes[0].set_title(f"i = {i}, action = {actions[i]}")

        axes[1].imshow(dmaps_after_actions[i])
        axes[1].set_title(f"dmap after actions")

        plt.show()
        plt.close()

