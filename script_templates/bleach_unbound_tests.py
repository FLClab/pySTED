import numpy as np
import tqdm
from pysted import base, utils
import os
import argparse
from matplotlib import pyplot as plt


print("Setting up the datamap and its flashes ...")
# Get light curves stuff to generate the flashes later
event_file_path = "flash_files/stream1_events.txt"
video_file_path = "flash_files/stream1.tif"

# Generate a datamap
frame_shape = (64, 64)
ensemble_func, synapses_list = utils.generate_synaptic_fibers(frame_shape, (9, 55), (3, 10), (2, 5),
                                                              seed=27)
# Build a dictionnary corresponding synapses to a bool saying if they are currently flashing or not
# They all start not flashing
flat_synapses_list = [item for sublist in synapses_list for item in sublist]

poils_frame = ensemble_func.return_frame().astype(int)

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
        "phy_react": {488: 1e-12,   # 1e-4
                      575: 1e-16},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 10e-9
confoc_pxsize = 30e-9   # confoc ground truths will be taken at a resolution 3 times lower than sted scans
dpxsz = 10e-9
bleach = True
p_ex = 1e-6
p_sted = 30e-3
min_pdt = 1e-6   # le min pdt est 1 us
pdt = 10e-6
higher_pdt_pixels = utils.pixel_sampling(poils_frame, mode="checkers")
roi = 'max'
acquisition_time = 5
flash_prob = 0.05   # every iteration, all synapses will have a 5% to start flashing
flash_seed = 42

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
temporal_datamap = base.TemporalDatamap(poils_frame, dpxsz, flat_synapses_list)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, bleach_func="default_bleach")
i_ex, _, _ = microscope.cache(temporal_datamap.pixelsize)

temporal_datamap = base.TemporalDatamap(poils_frame, dpxsz, flat_synapses_list)
temporal_datamap.set_roi(i_ex, roi)

temporal_datamap.create_t_stack_dmap(acquisition_time, pdt, (10, 1.5), event_file_path, video_file_path, flash_prob,
                                     i_ex, roi)

# set up variables for acquisition loop
t_stack_idx = 0
frozen_datamap = np.copy(temporal_datamap.whole_datamap[temporal_datamap.roi])
n_time_steps, n_tsteps_per_flash_step = utils.compute_time_correspondances((10, 1.5), acquisition_time, min_pdt, mode="pdt")
ratio = utils.pxsize_ratio(confoc_pxsize, temporal_datamap.pixelsize)
confoc_n_rows, confoc_n_cols = int(np.ceil(frame_shape[0] / ratio)), int(np.ceil(frame_shape[1] / ratio))
actions_required_pixels = {"confocal": confoc_n_rows * confoc_n_cols, "sted": frame_shape[0] * frame_shape[1]}
imaged_pixels = 0
action_selected = "confocal"
action_completed = False
pixels_for_current_action = actions_required_pixels[action_selected]
confoc_intensity = np.zeros((confoc_n_rows, confoc_n_cols)).astype(float)
sted_intensity = np.zeros(frozen_datamap.shape).astype(float)
list_datamaps = [np.copy(frozen_datamap)]
list_confocals = [np.zeros(confoc_intensity.shape)]
list_steds = [np.zeros(sted_intensity.shape)]
idx_type = {}
confocal_starting_pixel, sted_starting_pixel = [0, 0], [0, 0]

# verif that no values in the pdt_array are lower than the min pdt
min_pdt_selected = np.min(pdt)
if min_pdt_selected < min_pdt:
    # TODO : raise error or something not sure how I want to handle it
    print("hey!")
    exit()
confocal_pixel_list = utils.generate_raster_pixel_list(frame_shape[0] * frame_shape[1], [0, 0], frozen_datamap)
confocal_pixel_list = utils.pixel_list_filter(frozen_datamap, confocal_pixel_list, confoc_pxsize,
                                              temporal_datamap.pixelsize, output_empty=True)
confoc_pdt_array = np.ones(frame_shape) * pdt
sted_pdt_array = np.copy(confoc_pdt_array)
actions_required_time = {"confocal": np.sum(confoc_pdt_array), "sted": np.sum(sted_pdt_array)}
time_for_current_action = actions_required_time[action_selected]
time_spent_imaging = 0
# first action is always a confocal
pixel_list = confocal_pixel_list
pixel_list_time_idx = 0

# start acquisition loop
print("Starting the experiment loop")
np.random.seed(flash_seed)
np.random.RandomState(flash_seed)
for t_step_idx in tqdm.trange(n_time_steps):
    microscope.time_bank += min_pdt
    next_pixel_to_img = pixel_list[pixel_list_time_idx]
    if action_selected == "confocal":
        if microscope.time_bank - confoc_pdt_array[tuple(next_pixel_to_img)] >= 0:
            microscope.time_bank -= confoc_pdt_array[tuple(next_pixel_to_img)]
            pixel_list_time_idx += 1
            microscope.pixel_bank += 1
    elif action_selected == "sted":
        if microscope.time_bank - sted_pdt_array[tuple(next_pixel_to_img)] >= 0:
            microscope.time_bank -= sted_pdt_array[tuple(next_pixel_to_img)]
            pixel_list_time_idx += 1
            microscope.pixel_bank += 1
    # ici il va y avoir un elif pour XbyX sted

    # verify if the current action is interrupted by a flash step
    if t_step_idx % n_tsteps_per_flash_step == 0:

        # il faut que je fasse l'acquisition avant de maj la datamap
        if microscope.pixel_bank >= 1:
            if action_selected == "confocal":
                confoc_acq, confoc_intensity, temporal_datamap, imaged_pixel_list = \
                    utils.action_execution(action_selected, frame_shape, confocal_starting_pixel, confoc_pxsize,
                                           temporal_datamap, frozen_datamap, microscope,
                                           confoc_pdt_array, p_ex, 0.0, confoc_intensity, bleach)

            elif action_selected == "sted":
                sted_acq, sted_intensity, temporal_datamap, imaged_pixel_list = \
                    utils.action_execution(action_selected, frame_shape, sted_starting_pixel,
                                           temporal_datamap.pixelsize, temporal_datamap,
                                           frozen_datamap, microscope, sted_pdt_array, p_ex, p_sted, sted_intensity,
                                           bleach)

            # shift the starting pixel
            if action_selected == "confocal":
                confocal_starting_pixel = imaged_pixel_list[-1]
                confocal_starting_pixel = utils.set_starting_pixel(confocal_starting_pixel, frame_shape, ratio=ratio)
            elif action_selected == "sted":
                sted_starting_pixel = imaged_pixel_list[-1]
                sted_starting_pixel = utils.set_starting_pixel(sted_starting_pixel, frame_shape)

            # empty the pixel bank after the acquisition
            imaged_pixels += microscope.pixel_bank
            microscope.empty_pixel_bank()
            pixel_list_time_idx = 0

            if imaged_pixels == actions_required_pixels[action_selected]:
                action_completed = True

            # là c'est ici que je doit gérer le bleaching
            # si je comprends bien ce que j'ai fait ( :) ), si le bleach est à OFF je fais juste pass ?
            # et si le bleach est à ON, il faut que j'update la base_datamap et les flash_tstack:
            # whole_datamap devrait s'est fait bleaché, et whole_datamap = base + flash
            if not bleach:
                temporal_datamap.list_dmaps[t_stack_idx] = np.copy(frozen_datamap)
            else:
                if t_stack_idx < len(temporal_datamap.list_dmaps) - 1:
                    temporal_datamap.bleach_future(t_stack_idx)

        # update la datamap selon le flash
        temporal_datamap.whole_datamap = temporal_datamap.base_datamap + temporal_datamap.flash_tstack[t_stack_idx]
        t_stack_idx += 1

        plt.imshow(temporal_datamap.whole_datamap[temporal_datamap.roi])
        plt.show()
