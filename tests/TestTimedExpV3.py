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
# for t in range(temporal_dmap.flash_tstack.shape[0]):
#     plt.imshow(temporal_dmap.flash_tstack[t])
#     plt.title(f"t = {t}, value = {np.max(temporal_dmap.flash_tstack[t])}")
#     plt.show()
# exit()
dmap_update_times = np.arange(temporal_dmap.time_usec_between_flash_updates, exp_time + 1,
                              temporal_dmap.time_usec_between_flash_updates)

# this here will be my experiment loop
t = 0
flash_tstep = 0
indices = {"flashes": flash_tstep}
action_selected, action_completed = None, False
n_actions = 3
valid_actions = {0: "confocal", 1: "sted", 2: "wait"}
intensity = np.zeros(temporal_dmap.whole_datamap[temporal_dmap.roi].shape).astype(float)
acquisitions, selected_actions = [], []
while t < exp_time:
    # for the first test I just want to sample actions, not the params
    # once this works, I will randomly sample a pixel dwell time from a certain interval
    if action_selected is None or action_completed:
        # action_selected = valid_actions[np.random.randint(0, n_actions)]
        action_selected = valid_actions[0]
        if action_selected == "confocal":
            pdt = np.ones(temporal_dmap.whole_datamap[temporal_dmap.roi].shape) * pdt_val
            p_ex = p_ex_val
            p_sted = 0.0
        elif action_selected == "sted":
            pdt = np.ones(temporal_dmap.whole_datamap[temporal_dmap.roi].shape) * pdt_val
            p_ex = p_ex_val
            p_sted = p_sted_val
        elif action_selected == "wait":
            pdt = np.ones(temporal_dmap.whole_datamap[temporal_dmap.roi].shape) * pdt_val
            p_ex = 0.0
            p_sted = 0.0
        else:
            raise ValueError("Impossible action selected :)")
        action_completed = False
        selected_actions.append(action_selected)

    # compute the acquisition time for the selected action in usec since the time quantum is 1usec
    action_required_time = np.sum(pdt) * 1e6   # this assumes a pdt_val given in sec * 1e-6
    action_completed_time = t + action_required_time
    time_steps_covered_by_acq = np.arange(t, action_completed_time)
    dmap_times = []
    for i in time_steps_covered_by_acq:
        if i % temporal_dmap.time_usec_between_flash_updates == 0 and i != 0:
            dmap_times.append(i)

    # if len(dmap_times) == 0, this means the acquisition is not interupted and we can just do it whole
    # if not, then we need to split the acquisition
    if len(dmap_times) == 0:
        acq, bleached, intensity = microscope.get_signal_and_bleach(temporal_dmap, temporal_dmap.pixelsize, pdt, p_ex,
                                                                    p_sted, indices=indices,
                                                                    acquired_intensity=intensity, bleach=False,
                                                                    update=True)
        acquisitions.append(np.copy(acq))
        t += action_required_time
        action_completed = True
        intensity = np.zeros(temporal_dmap.whole_datamap[temporal_dmap.roi].shape).astype(float)
    else:
        # assume raster pixel scan
        pixel_list = utils.pixel_sampling(intensity, mode="all")
        flash_t_step_pixel_idx_dict = {}
        n_keys = 0
        first_key = flash_tstep
        for i in range(len(dmap_times) + 1):
            pdt_cumsum = np.cumsum(pdt * 1e6)
            if i < len(dmap_times) and dmap_times[i] >= exp_time:
                # the datamap would update, but the experiment will be over before then
                update_pixel_idx = np.argwhere(pdt_cumsum + t > exp_time)[0, 0]
                flash_t_step_pixel_idx_dict[flash_tstep] = update_pixel_idx
                if flash_tstep > first_key:
                    flash_t_step_pixel_idx_dict[flash_tstep] += flash_t_step_pixel_idx_dict[flash_tstep - 1]
                t = exp_time
                break
            elif i < len(dmap_times):   # mid update split
                update_pixel_idx = np.argwhere(pdt_cumsum + t > dmap_update_times[flash_tstep])[0, 0]
                flash_t_step_pixel_idx_dict[flash_tstep] = update_pixel_idx
                if flash_tstep > first_key:
                    flash_t_step_pixel_idx_dict[flash_tstep] += flash_t_step_pixel_idx_dict[flash_tstep - 1]
                n_keys += 1
                flash_tstep += 1
                t += pdt_cumsum[update_pixel_idx - 1]
            else:   # from last update to the end of acq
                update_pixel_idx = pdt_cumsum.shape[0] - 1
                flash_t_step_pixel_idx_dict[flash_tstep] = update_pixel_idx
                t += pdt_cumsum[update_pixel_idx] - pdt_cumsum[flash_t_step_pixel_idx_dict[flash_tstep - 1] - 1]
        key_counter = 0
        for key in flash_t_step_pixel_idx_dict:
            if key_counter == 0:
                acq_pixel_list = pixel_list[0:flash_t_step_pixel_idx_dict[key]]
            elif key_counter == n_keys:
                acq_pixel_list = pixel_list[flash_t_step_pixel_idx_dict[key - 1]:flash_t_step_pixel_idx_dict[key] + 1]
            else:
                acq_pixel_list = pixel_list[flash_t_step_pixel_idx_dict[key - 1]:flash_t_step_pixel_idx_dict[key]]
            if len(acq_pixel_list) == 0:   # acq is over time to go home
                break
            key_counter += 1
            indices = {"flashes": key}
            temporal_dmap.update_whole_datamap(key)
            temporal_dmap.update_dicts(indices)
            acq, bleached, intensity = microscope.get_signal_and_bleach(temporal_dmap, temporal_dmap.pixelsize,
                                                                        pdt, p_ex, p_sted, indices=indices,
                                                                        acquired_intensity=intensity, bleach=False,
                                                                        update=True, pixel_list=acq_pixel_list)

        acquisitions.append(np.copy(acq))
        action_completed = True
        intensity = np.zeros(temporal_dmap.whole_datamap[temporal_dmap.roi].shape).astype(float)


acquisitions = np.array(acquisitions)
print(f"acquisitions.shape = {acquisitions.shape}")
np.save(save_path + "/test_acqs_v3", acquisitions)

print(f"len(selected_actions) = {len(selected_actions)}")
textfile = open(save_path + "/test_selected_actions_v3.txt", "w")
for action in selected_actions:
    textfile.write(action + "\n")
textfile.close()

for i in range(acquisitions.shape[0]):
    plt.imshow(acquisitions[i])
    plt.title(i)
    plt.show()
