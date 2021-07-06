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


time_quantum_us = 1
master_clock = base.Clock(time_quantum_us)
exp_time = 1000000   # we want our experiment to last 1000000 us, or 1s

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
        "phy_react": {488: 1e-8,   # 1e-4
                      575: 1e-12},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 20e-9
dpxsz = 10e-9
bleach = False
p_ex = 1e-6
p_sted = 30e-3
pdt = np.ones(shroom.frame.shape) * 10e-6

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

# for now I will only code this acquisition loop in which the agent spams confocal acquisitions
selected_action = "confocal"
pixel_list = []
pixel_list_idx = 0
flash_step = 0
indices = {"flashes": flash_step}
confocal_acquisitions = []
dmaps_during_acqs = []
intensity = np.zeros(temporal_synapse_dmap.whole_datamap[temporal_synapse_dmap.roi].shape).astype(float)
for i in tqdm.trange(exp_time):
    master_clock.update_time()
    microscope.time_bank += master_clock.time_quantum_us * 1e-6   # add time to the time_bank in seconds

    # verify how many pixels we can image in the list
    if len(pixel_list) == 0:
        # refill the pixel list (with all pixels in a raster scan for now)
        pixel_list = utils.pixel_sampling(temporal_synapse_dmap.whole_datamap[temporal_synapse_dmap.roi])
        pixel_list_idx = 0
    next_pixel_time_to_img = pdt[tuple(pixel_list[pixel_list_idx])]
    if microscope.time_bank >= next_pixel_time_to_img:
        pixel_list_idx += 1
        microscope.time_bank -= next_pixel_time_to_img


    # if the nanodomains flashing are updated, do a partial acquisition
    if master_clock.current_time % temporal_synapse_dmap.time_usec_between_flash_updates == 0:

        # faire l'acquisition de la confocale jusqu'où on a assez de temps
        confocal_acq, _, intensity = microscope.get_signal_and_bleach(temporal_synapse_dmap,
                                                                      temporal_synapse_dmap.pixelsize,
                                                                      pdt, p_ex, 0.0, indices=indices,
                                                                      acquired_intensity=intensity,
                                                                      pixel_list=pixel_list[:pixel_list_idx + 1],
                                                                      bleach=False, update=False,
                                                                      filter_bypass=True)
        # remove the imaged pixels from the pixel_list and reset the idx
        pixel_list = pixel_list[pixel_list_idx + 1:]
        pixel_list_idx = 0

        # if this completed the acquisition, add the acquisition to the list, reset the intensity map
        if len(pixel_list) == 0:
            print("do I ever go in here?")
            confocal_acquisitions.append(np.copy(confocal_acq))
            dmaps_during_acqs.append(np.copy(temporal_synapse_dmap.whole_datamap[temporal_synapse_dmap.roi]))
            intensity = np.zeros(temporal_synapse_dmap.whole_datamap[temporal_synapse_dmap.roi].shape).astype(float)


        # update the flash
        flash_step += 1
        indices["flashes"] = flash_step
        temporal_synapse_dmap.update_whole_datamap(flash_step)
        temporal_synapse_dmap.update_dicts(indices)
        # plt.imshow(temporal_synapse_dmap.whole_datamap[temporal_synapse_dmap.roi])
        # plt.title(f"current_time = {master_clock.current_time}, flash_step = {flash_step}")
        # plt.show()

    elif pixel_list_idx == len(pixel_list) - 1:   # or else complete the acquisition if we are in measure of
        confocal_acq, _, intensity = microscope.get_signal_and_bleach(temporal_synapse_dmap,
                                                                      temporal_synapse_dmap.pixelsize, pdt, p_ex,
                                                                      0.0, indices=indices,
                                                                      acquired_intensity=intensity,
                                                                      pixel_list=pixel_list[:pixel_list_idx + 1],
                                                                      bleach=False, update=False,
                                                                      filter_bypass=True)
        # remove the imaged pixels from the pixel_list and reset the idx
        pixel_list = pixel_list[pixel_list_idx + 1:]
        pixel_list_idx = 0

        confocal_acquisitions.append(np.copy(confocal_acq))
        dmaps_during_acqs.append(np.copy(temporal_synapse_dmap.whole_datamap[temporal_synapse_dmap.roi]))
        intensity = np.zeros(temporal_synapse_dmap.whole_datamap[temporal_synapse_dmap.roi].shape).astype(float)

        if len(pixel_list) != 0:
            print("uh oh something's wrong")

dmaps_during_acqs = np.array(dmaps_during_acqs)
confocal_acquisitions = np.array(confocal_acquisitions)
print(f"dmaps_array.shape = {dmaps_during_acqs.shape}")
print(f"confocal_acquisitions.shape = {confocal_acquisitions.shape}")
save_path = os.path.join(os.path.expanduser('~'), "Documents", "research", "NeurIPS", "data_generation",
                         "confocals_spam_no_bleach")
np.save(save_path + "/test1_confocals", confocal_acquisitions)
np.save(save_path + "/test1_dmaps", dmaps_during_acqs)
# J'AI UN PROBLÈME OÙ J'AI PAS L'AIR D'ACQUÉRIR SUR LA BONNE DATAMAP :)))))))))))
# C'EST COMME SI JE FAISAIS L'ACQ JUSTE SUR LA BASE ET NON SUR LA BASE + FLASH
