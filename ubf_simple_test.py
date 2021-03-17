import numpy as np
import tqdm
from pysted import base, utils, raster
import os
import argparse
from matplotlib import pyplot as plt
import time


print("Setting up the datamap and its flashes ...")
# Get light curves stuff to generate the flashes later
event_file_path = "flash_files/stream1_events.txt"
video_file_path = "flash_files/stream1.tif"

# Generate a datamap
frame_shape = (64, 64)
ensemble_func, synapses_list = utils.generate_synaptic_fibers(frame_shape, (9, 55), (3, 10), (2, 5),
                                                              seed=27)

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
        "phy_react": {488: 1e-8,   # 1e-4
                      575: 1e-12},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 10e-9
confoc_pxsize = 30e-9   # confoc ground truths will be taken at a resolution 3 times lower than sted scans
dpxsz = 10e-9
bleach = True
p_ex = np.ones(frame_shape) * 1e-6
# p_ex = 1e-6
p_sted = 30e-3
min_pdt = 1e-6   # le min pdt est 1 us
pdt = 10e-6
# pdt = np.ones(frame_shape) * min_pdt
# higher_pdt_pixels = utils.pixel_sampling(poils_frame, mode="checkers")
# for row, col in higher_pdt_pixels:
#     pdt[row, col] = 10e-6
roi = 'max'
acquisition_time = 1
flash_prob = 0.05   # every iteration, all synapses will have a 5% to start flashing
flash_seed = None

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
temporal_datamap = base.TemporalDatamap(poils_frame, dpxsz, flat_synapses_list)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, bleach_func="default_bleach")
i_ex, _, _ = microscope.cache(temporal_datamap.pixelsize)
# temporal_datamap = base.TemporalDatamap(poils_frame, dpxsz, flat_synapses_list)
temporal_datamap.set_roi(i_ex, roi)
temporal_datamap.create_t_stack_dmap(acquisition_time, pdt, (10, 1.5), event_file_path, video_file_path, flash_prob,
                                     i_ex, roi)
idx_dict = {"flashes": 3}
# for i in range(temporal_datamap.flash_tstack.shape[0]):
#     plt.imshow(temporal_datamap.flash_tstack[i][temporal_datamap.roi])
#     plt.title(f"i = {i}")
#     plt.show()
# exit()

# va falloir que je fasse cette gestion là dans la loop d'acq
temporal_datamap.whole_datamap = temporal_datamap.base_datamap + temporal_datamap.flash_tstack[idx_dict["flashes"]]
temporal_datamap.sub_datamaps_idx_dict["flashes"] = idx_dict["flashes"]
temporal_datamap.sub_datamaps_dict["flashes"] = temporal_datamap.flash_tstack[temporal_datamap.sub_datamaps_idx_dict["flashes"]]

start_time = time.time()
photons, bleached, intensity = microscope.get_signal_and_bleach_fast_2(temporal_datamap, temporal_datamap.pixelsize,
                                                                       pdt, p_ex, p_sted, bleach=bleach,
                                                                       update=True, indices=idx_dict,
                                                                       raster_func=raster.raster_func_c_self_bleach_split)
run_time = time.time() - start_time

print(f"run time = {run_time}")
bleached_total = np.zeros(temporal_datamap.whole_datamap.shape)
for key in bleached:
    bleached_total += bleached[key]

fig, axes = plt.subplots(1, 3)
axes[0].imshow(temporal_datamap.whole_datamap[temporal_datamap.roi])
axes[1].imshow(bleached_total[temporal_datamap.roi])
axes[2].imshow(photons)
fig.suptitle(f"run time = {run_time}")
plt.show()

fig, axes = plt.subplots(1, 2)
axes[0].imshow(temporal_datamap.base_datamap[temporal_datamap.roi])
axes[1].imshow(temporal_datamap.flash_tstack[idx_dict["flashes"]][temporal_datamap.roi])
plt.show()

for i in range(temporal_datamap.flash_tstack.shape[0]):
    plt.imshow(temporal_datamap.flash_tstack[i][temporal_datamap.roi])
    plt.title(f"i = {i}")
    plt.show()
exit()
