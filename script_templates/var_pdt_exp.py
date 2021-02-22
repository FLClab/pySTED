import numpy as np
import tqdm
from pysted import base, utils
import os
import argparse
from matplotlib import pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# add arg parser handling
parser = argparse.ArgumentParser(description="Example of experiment script")
parser.add_argument("--save_path", type=str, default="", help="Where to save the files")
parser.add_argument("--bleach", type=str2bool, default=False, help="Whether or not bleaching is on or not")
parser.add_argument("--dmap_seed", type=int, default=None, help="Whether or not the dmap is created using a seed")
parser.add_argument("--flash_seed", type=int, default=None, help="Whether or not the flashes are controlled by a seed")
parser.add_argument("--acq_time", type=int, default=5, help="Acquisition time (in seconds)")
args = parser.parse_args()


save_path = args.save_path
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Get light curves stuff to generate the flashes later
event_file_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
video_file_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"

# Generate a datamap
frame_shape = (64, 64)
ensemble_func, synapses_list = utils.generate_synaptic_fibers(frame_shape, (9, 55), (3, 10), (2, 5),
                                                              seed=args.dmap_seed)
# Build a dictionnary corresponding synapses to a bool saying if they are currently flashing or not
# They all start not flashing
flat_synapses_list = [item for sublist in synapses_list for item in sublist]

poils_frame = ensemble_func.return_frame().astype(int)

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
bleach = args.bleach
p_ex = 1e-6
p_sted = 30e-3
# set un min pdt
# gérer ma loop en fonction de ce min pdt
# pouvoir avoir pdt comme matrice
min_pdt = 1e-6   # le min pdt est 1 us
# pdt = 10e-6   # pour (10, 1.5) ça me donne 15k pixels par iter
pdt = np.ones(frame_shape) * min_pdt
higher_pdt_pixels = utils.pixel_sampling(poils_frame, mode="checkers")
for row, col in higher_pdt_pixels:
    pdt[row, col] = 10e-6
roi = 'max'
acquisition_time = args.acq_time
flash_prob = 0.05   # every iteration, all synapses will have a 5% to start flashing
flash_seed = args.flash_seed

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
# zero_residual controls how much of the donut beam "bleeds" into the the donut hole
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
# noise allows noise on the detector, background adds an average photon count for the empty pixels
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
frozen_datamap = np.copy(temporal_datamap.list_dmaps[t_stack_idx].whole_datamap[temporal_datamap.roi])
# faudrait que j'utilise min_pdt ici
n_time_steps, n_tsteps_per_flash_step = utils.compute_time_correspondances((10, 1.5), acquisition_time, min_pdt, mode="pdt")
ratio = utils.pxsize_ratio(confoc_pxsize, temporal_datamap.pixelsize)
confoc_n_rows, confoc_n_cols = int(np.ceil(frame_shape[0] / ratio)), int(np.ceil(frame_shape[1] / ratio))
actions_required_pixels = {"confocal": confoc_n_rows * confoc_n_cols, "sted": frame_shape[0] * frame_shape[1]}
imaged_pixels = 0   # can cap at either n_pixels_confoc or n_pixels_sted
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

# ------------------ GETSION DE L'IMPLEM D'UN PDT VARIABLE ----------------------------------------
# verif that no values in the pdt_array are lower than the min pdt
min_pdt_selected = np.min(pdt)
if min_pdt_selected < min_pdt:
    # TODO : raise error or something not sure how I want to handle it
    print("hey!")
    exit()
time_bank = 0
confocal_pixel_list = utils.generate_raster_pixel_list(frame_shape[0] * frame_shape[1], [0, 0], frozen_datamap)
confocal_pixel_list = utils.pixel_list_filter(frozen_datamap, confocal_pixel_list, confoc_pxsize,
                                              temporal_datamap.pixelsize, output_empty=True)
confoc_pdt_array = np.zeros(frame_shape)
for row, col in confocal_pixel_list:
    confoc_pdt_array[row, col] = pdt[row, col]
actions_required_time = {"confocal": np.sum(confoc_pdt_array), "sted": np.sum(pdt)}
time_for_current_action = actions_required_time[action_selected]
time_spent_imaging = 0

# start acquisition loop
np.random.seed(flash_seed)
np.random.RandomState(flash_seed)
for t_step_idx in tqdm.trange(n_time_steps):
    # faut pas que j'add 1 pixel au pixel_bank à chaque iter
    # how the fuck do I manage time
    time_bank += min_pdt
    if action_selected == "confocal":
        # print(time_bank - confoc_pdt_array[confocal_starting_pixel])
        if time_bank - confoc_pdt_array[tuple(confocal_starting_pixel)] >= 0:
            microscope.pixel_bank += 1
            time_bank -= confoc_pdt_array[tuple(confocal_starting_pixel)]
    elif action_selected == "sted":
        if time_bank - pdt[tuple(sted_starting_pixel)] >= 0:
            microscope.pixel_bank += 1
            time_bank -= pdt[tuple(sted_starting_pixel)]

    if t_step_idx % n_tsteps_per_flash_step == 0:

        if microscope.pixel_bank >= 1:
            if action_selected == "confocal":
                confoc_acq, confoc_intensity, temporal_datamap.list_dmaps[t_stack_idx], pixel_list = \
                    utils.action_execution(action_selected, frame_shape, confocal_starting_pixel, confoc_pxsize,
                                           temporal_datamap.list_dmaps[t_stack_idx], frozen_datamap, microscope, pdt,
                                           p_ex, 0.0, confoc_intensity, bleach)

            elif action_selected == "sted":
                sted_acq, sted_intensity, temporal_datamap.list_dmaps[t_stack_idx], pixel_list = \
                    utils.action_execution(action_selected, frame_shape, sted_starting_pixel,
                                           temporal_datamap.pixelsize, temporal_datamap.list_dmaps[t_stack_idx],
                                           frozen_datamap, microscope, pdt, p_ex, p_sted, sted_intensity, bleach)

            # shift the starting pixel
            if action_selected == "confocal":
                confocal_starting_pixel = pixel_list[-1]
                confocal_starting_pixel = utils.set_starting_pixel(confocal_starting_pixel, frame_shape, ratio=ratio)
            elif action_selected == "sted":
                sted_starting_pixel = pixel_list[-1]
                sted_starting_pixel = utils.set_starting_pixel(sted_starting_pixel, frame_shape)

            # empty the pixel bank after the acquisition
            imaged_pixels += microscope.pixel_bank
            microscope.empty_pixel_bank()

            if imaged_pixels == actions_required_pixels[action_selected]:
                action_completed = True

            if not bleach:
                temporal_datamap.list_dmaps[t_stack_idx] = np.copy(frozen_datamap)
            else:
                if t_stack_idx < len(temporal_datamap.list_dmaps) - 1:
                    temporal_datamap.bleach_future(t_stack_idx)
        # get a copy of the datamap to add to a list to save later
        # IL FAUDRAIT QUE JE SET LES PROCHAINS FRAME DANS LA temporal_datamap.list_dmaps À LA VERSION BLEACHÉE QUE J'AI
        # EU LÀ, MAIS GENRE C'EST COMPLIQUÉ FUCK
        t_stack_idx += 1
        roi_save_copy = np.copy(temporal_datamap.list_dmaps[t_stack_idx].
                                whole_datamap[temporal_datamap.list_dmaps[t_stack_idx].roi])
        list_datamaps.append(roi_save_copy)
        idx_type[t_step_idx] = "datamap"
        # print(f"flash updated, now at idx {t_stack_idx}")

    # Regarder il me manque combien de pixels
    pixels_needed_to_complete_acq = pixels_for_current_action - imaged_pixels
    # time_needed_to_complete_acq = time_for_current_action -

    if microscope.pixel_bank == pixels_needed_to_complete_acq:

        if action_selected == "confocal":
            confoc_acq, confoc_intensity, temporal_datamap.list_dmaps[t_stack_idx], pixel_list = \
                utils.action_execution(action_selected, frame_shape, confocal_starting_pixel, confoc_pxsize,
                                       temporal_datamap.list_dmaps[t_stack_idx], frozen_datamap, microscope, pdt, p_ex,
                                       0.0, confoc_intensity, bleach)

        elif action_selected == "sted":
            sted_acq, sted_intensity, temporal_datamap.list_dmaps[t_stack_idx], pixel_list = \
                utils.action_execution(action_selected, frame_shape, sted_starting_pixel, temporal_datamap.pixelsize,
                                       temporal_datamap.list_dmaps[t_stack_idx], frozen_datamap, microscope, pdt, p_ex,
                                       p_sted, sted_intensity, bleach)

        if bleach and t_stack_idx < len(temporal_datamap.list_dmaps) - 1:
            temporal_datamap.bleach_future(t_stack_idx)

        # shift the starting pixel
        if action_selected == "confocal":
            confocal_starting_pixel = pixel_list[-1]
            confocal_starting_pixel = utils.set_starting_pixel(confocal_starting_pixel, frame_shape, ratio=ratio)
        elif action_selected == "sted":
            sted_starting_pixel = pixel_list[-1]
            sted_starting_pixel = utils.set_starting_pixel(sted_starting_pixel, frame_shape)

        # empty the pixel bank after the acquisition
        imaged_pixels += microscope.pixel_bank
        microscope.empty_pixel_bank()

        action_completed = True

    if action_completed:
        # add acquisition to be saved
        if action_selected == "confocal":
            list_confocals.append(np.copy(confoc_acq))
            idx_type[t_step_idx] = "confocal"
        elif action_selected == "sted":
            list_steds.append(np.copy(sted_acq))
            idx_type[t_step_idx] = "sted"

        # select the new action based off the previous
        # (so for now this is confocal -> sted -> confocal)
        action_completed = False
        if action_selected == "confocal":
            action_selected = "sted"
        elif action_selected == "sted":
            action_selected = "confocal"

        imaged_pixels = 0
        pixels_for_current_action = actions_required_pixels[action_selected]

# make stacks for datamaps, confocals and steds, and save them
datamaps_stack = np.stack(list_datamaps)
confocals_stack = np.stack(list_confocals)
steds_stack = np.stack(list_steds)
np.save(save_path + "/datamaps", datamaps_stack)
np.save(save_path + "/confocals", confocals_stack)
np.save(save_path + "/steds", steds_stack)

# write the lines in the script file for video generation
ffmpeg_file_path = save_path + "/in.ffconcat"
file = open(ffmpeg_file_path, "a")
file.write("ffconcat version 1.0\n")
file.write("file 0.png\n")
file.write(f"duration 5\n")
file.close()
keys_list = sorted(idx_type.keys())
for idx, key in enumerate(keys_list):
    if idx_type[key] == "datamap":
        list_datamaps.pop(0)
    elif idx_type[key] == "confocal":
        list_confocals.pop(0)
    elif idx_type[key] == "sted":
        list_steds.pop(0)
    else:
        print(f"FORBIDDEN UNKNOWN")

    if key != keys_list[-1]:
        # make the calculations for times to write in script file
        duration = (keys_list[idx + 1] - key) * min_pdt * 10  # right?
        # write the lines to script file
        file = open(ffmpeg_file_path, "a")
        file.write(f"file {key + 1}.png\n")
        file.write(f"duration {duration}\n")
        file.close()
    else:
        for i in range(2):   # do it twice or else it skips the last frame
            duration = 10
            # write the lines to script file
            file = open(ffmpeg_file_path, "a")
            file.write(f"file {key + 1}.png\n")
            file.write(f"duration {duration}\n")
            file.close()

np.save(save_path + "/idx_type_dict", idx_type)
# so this should be everything for the experiment part of the script
