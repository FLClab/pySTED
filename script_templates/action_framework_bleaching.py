import numpy as np
from matplotlib import pyplot as plt
import tqdm
from pysted import base, utils, temporal
import os
import shutil

"""
Le but de se script est de faire le framework pour une acquisition avec prise de décision.
Prise de décision ici signifie dire au microscope quels pixels imager selon l'information qu'on a.
L'information qu'on a correspondera à un scan confocal à plus basse résolution que les scans steds.
Après chaque décision et action, on enregistre l'état de l'acquisition 
Version 1 (simple):
    La prise de décision est simplement de faire un raster scan complet.
    Donc dans la loop, on va : 
        faire une acq confoc à basse résolution
        prendre une décision (juste faire un raster scan for now)
        Faire l'action
        enregistrer la dernière image sted, datamap, ...

FUCK LA VERSION 1, si je veux que ça marche je pense que j'ai pas le choix de faire des iterations
dans le time step du microscope. Par contre ça va me faire une loop de genre 100000000000000000000
itérations, alors comment est-ce que je veux gérer ça?
"""

# Get light curves stuff to generate the flashes later
event_file_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
video_file_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"

# Generate a datamap
frame_shape = (64, 64)
ensemble_func, synapses_list = utils.generate_synaptic_fibers(frame_shape, (9, 55), (3, 10), (2, 5), seed=None)

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
        "phy_react": {488: 1e-8,  # 1e-4
                      575: 1e-12},  # 1e-8
        "k_isc": 0.26e6}
pixelsize = 10e-9
confoc_pxsize = 30e-9  # confoc ground truths will be taken at a resolution 3 times lower than sted scans
dpxsz = 10e-9
bleach = True
p_ex = 1e-6
p_sted = 30e-3
p_sted_array = np.reshape(np.linspace(0, p_sted * 5, 64 * 64), poils_frame.shape)
p_ex_array = np.reshape(np.linspace(0, p_ex * 3, 64 * 64), poils_frame.shape)
pdt = 10e-6  # pour (10, 1.5) ça me donne 15k pixels par iter
# pdt = 0.3   # pour (10, 1.5) ça me donne 0.5 pixels par iter
size = 64 + (2 * 22 + 1)
roi = 'max'
seed = True

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
# zero_residual controls how much of the donut beam "bleeds" into the the donut hole
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
# noise allows noise on the detector, background adds an average photon count for the empty pixels
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap = base.Datamap(poils_frame, dpxsz)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, bleach_func="default_bleach")
i_ex, _, _ = microscope.cache(datamap.pixelsize)
datamap.set_roi(i_ex, roi)

# Build a dictionnary corresponding synapses to a bool saying if they are currently flashing or not
# They all start not flashing
flat_synapses_list = [item for sublist in synapses_list for item in sublist]

synpase_flashing_dict, synapse_flash_idx_dict, synapse_flash_curve_dict, isolated_synapses_frames = \
    utils.generate_synapse_flash_dicts(flat_synapses_list, frame_shape)

# start acquisition loop

save_path = "D:/SCHOOL/Maitrise/H2021/Recherche/data_generation/video_tests_varying_params/bleaching_2/"
acquisition_time = 5  ## in seconds
flash_prob = 0.05  # every iteration, all synapses will have a 5% to start flashing
frozen_datamap = np.copy(datamap.whole_datamap[datamap.roi])
n_time_steps, n_pixel_per_flash_step = utils.compute_time_correspondances((10, 1.5), acquisition_time, pdt, mode="pdt")
print(f"n_pixel_per_flash_step = {n_pixel_per_flash_step}")
ratio = utils.pxsize_ratio(confoc_pxsize, datamap.pixelsize)
confoc_n_rows, confoc_n_cols = int(np.ceil(frame_shape[0] / ratio)), int(np.ceil(frame_shape[1] / ratio))
actions_required_pixels = {"confocal": confoc_n_rows * confoc_n_cols, "sted": frame_shape[0] * frame_shape[1]}
imaged_pixels = 0  # can cap at either n_pixels_confoc or n_pixels_sted
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
for pixel_idx in tqdm.trange(n_time_steps):
    microscope.pixel_bank += 1
    if pixel_idx % n_pixel_per_flash_step == 0:

        if action_selected == "confocal":
            pixel_list = utils.generate_raster_pixel_list(frame_shape[0] * frame_shape[1], confocal_starting_pixel,
                                                          frozen_datamap)
            pixel_list = utils.pixel_list_filter(frozen_datamap, pixel_list, confoc_pxsize, datamap.pixelsize,
                                                 output_empty=True)

            # Cut elements before the starting pixel from the list
            start_idx = pixel_list.index(tuple(confocal_starting_pixel))
            pixel_list = pixel_list[start_idx:]
            pixel_list = pixel_list[:microscope.pixel_bank]

        elif action_selected == "sted":
            # faire la sélection que je fais déjà plus haut
            pixel_list = utils.generate_raster_pixel_list(microscope.pixel_bank, sted_starting_pixel, frozen_datamap)
            pixel_list = utils.pixel_list_filter(frozen_datamap, pixel_list, datamap.pixelsize, datamap.pixelsize,
                                                 output_empty=True)

        # faire le scan confocal sur la pixel_list
        # faire un if pour confocal ou sted
        if action_selected == "confocal":
            confoc_acq, bleached, confoc_intensity = microscope.get_signal_and_bleach_fast(datamap, confoc_pxsize, pdt, p_ex,
                                                                                    0.0,
                                                                                    acquired_intensity=confoc_intensity,
                                                                                    pixel_list=pixel_list,
                                                                                    bleach=bleach, update=False,
                                                                                    filter_bypass=True)
            datamap.whole_datamap = np.copy(bleached)

        elif action_selected == "sted":
            sted_acq, bleached, sted_intensity = microscope.get_signal_and_bleach_fast(datamap, datamap.pixelsize, pdt, p_ex,
                                                                                p_sted,
                                                                                acquired_intensity=sted_intensity,
                                                                                pixel_list=pixel_list,
                                                                                bleach=bleach, update=False,
                                                                                filter_bypass=True)
            datamap.whole_datamap = np.copy(bleached)

        # shift the starting pixel
        if action_selected == "confocal":
            confocal_starting_pixel = pixel_list[-1]
            # confocal_starting_pixel = utils.set_starting_pixel(confocal_starting_pixel, (confoc_n_rows, confoc_n_cols))
            confocal_starting_pixel = utils.set_starting_pixel(confocal_starting_pixel, frame_shape, ratio=ratio)
        elif action_selected == "sted":
            sted_starting_pixel = pixel_list[-1]
            sted_starting_pixel = utils.set_starting_pixel(sted_starting_pixel, frame_shape)

        # empty the pixel bank after the acquisition
        imaged_pixels += microscope.pixel_bank
        microscope.empty_pixel_bank()

        if imaged_pixels == actions_required_pixels[action_selected]:
            action_completed = True

        # datamap.whole_datamap[datamap.roi] = np.copy(frozen_datamap)
        # loop through all synapses, make some start to flash, randomly, maybe
        for idx_syn in range(len(flat_synapses_list)):
            if np.random.binomial(1, flash_prob) and synpase_flashing_dict[idx_syn] is False:
                # can start the flash
                synpase_flashing_dict[idx_syn] = True
                synapse_flash_idx_dict[idx_syn] = 1
                sampled_curve = utils.flash_generator(event_file_path, video_file_path)
                synapse_flash_curve_dict[idx_syn] = utils.rescale_data(sampled_curve, to_int=True, divider=3)

            if synpase_flashing_dict[idx_syn]:
                datamap.whole_datamap[datamap.roi] -= isolated_synapses_frames[idx_syn]
                datamap.whole_datamap[datamap.roi] += isolated_synapses_frames[idx_syn] * \
                                                      synapse_flash_curve_dict[idx_syn][synapse_flash_idx_dict[idx_syn]]
                synapse_flash_idx_dict[idx_syn] += 1
                if synapse_flash_idx_dict[idx_syn] >= 40:
                    synapse_flash_idx_dict[idx_syn] = 0
                    synpase_flashing_dict[idx_syn] = False

        # get a copy of the datamap to add to a list to save later
        roi_save_copy = np.copy(datamap.whole_datamap[datamap.roi])
        # plt.imshow(roi_save_copy)
        # plt.show()
        list_datamaps.append(roi_save_copy)
        idx_type[pixel_idx] = "datamap"

    # Regarder il me manque combien de pixels
    pixels_needed_to_complete_acq = pixels_for_current_action - imaged_pixels

    if microscope.pixel_bank == pixels_needed_to_complete_acq:

        if action_selected == "confocal":

            pixel_list = utils.generate_raster_pixel_list(frame_shape[0] * frame_shape[1], sted_starting_pixel,
                                                          frozen_datamap)
            pixel_list = utils.pixel_list_filter(frozen_datamap, pixel_list, confoc_pxsize, datamap.pixelsize,
                                                 output_empty=True)

            # Cut elements before the starting pixel from the list
            start_idx = pixel_list.index(tuple(confocal_starting_pixel))
            pixel_list = pixel_list[start_idx:]
            pixel_list = pixel_list[:microscope.pixel_bank]
            # print(pixel_list[0])

        elif action_selected == "sted":
            # faire la sélection que je fais déjà plus haut
            pixel_list = utils.generate_raster_pixel_list(microscope.pixel_bank, sted_starting_pixel, frozen_datamap)
            pixel_list = utils.pixel_list_filter(frozen_datamap, pixel_list, datamap.pixelsize, datamap.pixelsize,
                                                 output_empty=True)

        # faire le scan confocal sur la pixel_list
        # faire un if pour confocal ou sted
        if action_selected == "confocal":
            confoc_acq, bleached, confoc_intensity = microscope.get_signal_and_bleach_fast(datamap, confoc_pxsize, pdt, p_ex,
                                                                                    0.0,
                                                                                    acquired_intensity=confoc_intensity,
                                                                                    pixel_list=pixel_list,
                                                                                    bleach=bleach, update=False,
                                                                                    filter_bypass=True)
            datamap.whole_datamap = np.copy(bleached)

        elif action_selected == "sted":
            sted_acq, bleached, sted_intensity = microscope.get_signal_and_bleach_fast(datamap, datamap.pixelsize, pdt, p_ex,
                                                                                p_sted,
                                                                                acquired_intensity=sted_intensity,
                                                                                pixel_list=pixel_list,
                                                                                bleach=bleach, update=False,
                                                                                filter_bypass=True)
            datamap.whole_datamap = np.copy(bleached)

        # shift the starting pixel
        if action_selected == "confocal":
            confocal_starting_pixel = pixel_list[-1]
            # confocal_starting_pixel = utils.set_starting_pixel(confocal_starting_pixel, (confoc_n_rows, confoc_n_cols))
            confocal_starting_pixel = utils.set_starting_pixel(confocal_starting_pixel, frame_shape, ratio=ratio)
        elif action_selected == "sted":
            sted_starting_pixel = pixel_list[-1]
            sted_starting_pixel = utils.set_starting_pixel(sted_starting_pixel, frame_shape)

        # empty the pixel bank after the acquisition
        imaged_pixels += microscope.pixel_bank
        microscope.empty_pixel_bank()

        # l'acquisition est finit, ce qui veut dire que que je dois faire des choses là :)
        action_completed = True

    if action_completed:
        # add acquisition to be saved
        if action_selected == "confocal":
            list_confocals.append(np.copy(confoc_acq))
            idx_type[pixel_idx] = "confocal"
        elif action_selected == "sted":
            list_steds.append(np.copy(sted_acq))
            idx_type[pixel_idx] = "sted"

        # select the new action based off the previous
        # (so for now this is confocal -> sted -> confocal)
        action_completed = False
        if action_selected == "confocal":
            action_selected = "sted"
            # confoc_intensity = np.zeros((confoc_n_rows, confoc_n_cols)).astype(float)
        elif action_selected == "sted":
            action_selected = "confocal"
            # sted_intensity = np.zeros(frozen_datamap.shape).astype(float)

        imaged_pixels = 0
        pixels_for_current_action = actions_required_pixels[action_selected]
        # confoc_intensity = np.zeros((confoc_n_rows, confoc_n_cols)).astype(float)
        # sted_intensity = np.zeros(frozen_datamap.shape).astype(float)

# save_path = "D:/SCHOOL/Maitrise/H2021/Recherche/data_generation/time_integration/ffmpeg_raster_video/test1/"
ffmpeg_file_path = save_path + "in.ffconcat"

min_datamap, max_datamap = np.min(list_datamaps), np.max(list_datamaps)
min_confocal, max_confocal = np.min(list_confocals), np.max(list_confocals)
min_sted, max_sted = np.min(list_steds), np.max(list_steds)

# for idx in range(len(list_datamaps)):
#     plt.imsave(save_path + f"datamaps/{idx}.png", list_datamaps[idx], vmin=min_datamap, vmax=max_datamap)
# for idx in range(len(list_confocals)):
#     plt.imsave(save_path + f"confocals/{idx}.png", list_confocals[idx], vmin=min_confocal, vmax=max_confocal)
# for idx in range(len(list_steds)):
#     plt.imsave(save_path + f"steds/{idx}.png", list_steds[idx], vmin=min_sted, vmax=max_sted)
# plt.imsave(save_path + "flat_datamap.png", poils_frame)

# Generate the first figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

dmap_imshow = axes[0].imshow(list_datamaps[0], vmin=min_datamap, vmax=max_datamap)
axes[0].set_title(f"Datamap")
fig.colorbar(dmap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

confocal_imshow = axes[1].imshow(list_confocals[0], vmin=min_confocal, vmax=max_confocal)
axes[1].set_title(f"Confocal (Ground truth) \n "
                  f"1/3 the resolution of STED")
fig.colorbar(confocal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

sted_imshow = axes[2].imshow(list_steds[0], vmin=min_sted, vmax=max_sted)
axes[2].set_title(f"STED acquisition")
fig.colorbar(sted_imshow, ax=axes[2], fraction=0.04, pad=0.05)

fig.suptitle(f"Acquisition will start in 5 seconds")
plt.savefig(save_path + f"0.png")
plt.close()

# write the lines in the script file for video generation
file = open(ffmpeg_file_path, "a")
file.write("ffconcat version 1.0\n")
file.write("file 0.png\n")
file.write(f"duration 5\n")
file.close()

# loop through the keys in the dict to correctly order the picture updates
# faire un tqdm range serait pas pire :)
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
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

        dmap_imshow = axes[0].imshow(list_datamaps[0], vmin=min_datamap, vmax=max_datamap)
        axes[0].set_title(f"Datamap")
        fig.colorbar(dmap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

        confocal_imshow = axes[1].imshow(list_confocals[0], vmin=min_confocal, vmax=max_confocal)
        axes[1].set_title(f"Confocal (Ground truth) \n "
                          f"1/3 the resolution of STED")
        fig.colorbar(confocal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

        sted_imshow = axes[2].imshow(list_steds[0], vmin=min_sted, vmax=max_sted)
        axes[2].set_title(f"STED acquisition")
        fig.colorbar(sted_imshow, ax=axes[2], fraction=0.04, pad=0.05)

        fig.suptitle(f"Acquisition has started")
        plt.savefig(save_path + f"{key + 1}.png")
        plt.close()

        # make the calculations for times to write in script file
        duration = (keys_list[idx + 1] - key) * pdt * 10  # right?

        # write the lines to script file
        file = open(ffmpeg_file_path, "a")
        file.write(f"file {key + 1}.png\n")
        file.write(f"duration {duration}\n")  # need to modify this so it computes the right time
        file.close()

    else:
        for i in range(2):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

            dmap_imshow = axes[0].imshow(list_datamaps[0], vmin=min_datamap, vmax=max_datamap)
            axes[0].set_title(f"Datamap")
            fig.colorbar(dmap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

            confocal_imshow = axes[1].imshow(list_confocals[0], vmin=min_confocal, vmax=max_confocal)
            axes[1].set_title(f"Confocal (Ground truth) \n "
                              f"1/3 the resolution of STED")
            fig.colorbar(confocal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

            sted_imshow = axes[2].imshow(list_steds[0], vmin=min_sted, vmax=max_sted)
            axes[2].set_title(f"STED acquisition")
            fig.colorbar(sted_imshow, ax=axes[2], fraction=0.04, pad=0.05)

            fig.suptitle(f"Acquisition has ended")
            plt.savefig(save_path + f"{key + 1}.png")
            plt.close()

            # make the calculations for times to write in script file
            duration = 10  # right?

            # write the lines to script file
            file = open(ffmpeg_file_path, "a")
            file.write(f"file {key + 1}.png\n")
            file.write(f"duration {duration}\n")  # need to modify this so it computes the right time
            file.close()

# calculer le temps total du video en secondes et l'enregistrer qqpart pour
total_duration = 5 + (keys_list[-1] * pdt * 10) + 10

# copy the audio file over, :)
mp3_file = "D:/SCHOOL/Maitrise/H2021/Recherche/data_generation/time_integration/ffmpeg_video_tests/Trance_009_Sound_System_Dreamscape_HD.mp3"
shutil.copy(mp3_file, save_path)

# call cmd lines to make the video
# EVERYTHING IS WORKING WELL EXCEPT THAT THE LAST FRAME WONT DISPLAY? I COULD HACK MY WAY BY APPENDING THE LAST FRAME
# ONE MORE TIME I GUESS IDK
os.chdir(save_path)
os.system(
    fr'cmd /c "ffmpeg -i in.ffconcat -i Trance_009_Sound_System_Dreamscape_HD.mp3 -c:v libx264 -preset ultrafast -crf 0 -c:a copy -vf fps=25 out.avi"')
os.chdir(save_path)
os.system(fr'cmd /c "ffmpeg -ss 0 -i out.avi -t {total_duration} -c copy cut_out.avi"')