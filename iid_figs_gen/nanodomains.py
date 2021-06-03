import numpy as np
import tqdm
from pysted import base, utils, temporal
import os
import argparse
from matplotlib import pyplot as plt


save_path = r"D:\SCHOOL\Maitrise\E2021\research\iid_pres\figs\simulated_data\nanodomains\3_cols_left_brighter"
if not os.path.exists(save_path):
    os.mkdir(save_path)

print("Setting up the datamap and its flashes ...")
# Get light curves stuff to generate the flashes later
# event_file_path = "flash_files/stream1_events.txt"
# video_file_path = "flash_files/stream1.tif"
curves_path = "flash_files/events_curves.npy"

# I want a datamap with an oval-ish thing in the center, within which will be 2 synapses objects
# to look like columns

# Generate a datamap
frame_shape = (64, 64)
fluo_gt = np.zeros(frame_shape)

# I want an oval ish shape centered
# base rectangle
fluo_gt[20:40, 22:42] += 5

# top hat
fluo_gt[19, 23:41] += 5
fluo_gt[18, 24:40] += 5
fluo_gt[17, 25:39] += 5
fluo_gt[16, 26:38] += 5
fluo_gt[15, 27:37] += 5
# bot hat
fluo_gt[40, 23:41] += 5
fluo_gt[41, 24:40] += 5
fluo_gt[42, 25:39] += 5
fluo_gt[43, 26:38] += 5
fluo_gt[44, 27:37] += 5
# left hat
fluo_gt[22:38, 21] += 5
# right hat
fluo_gt[22:38, 42] += 5

# for the first test, I want 2 nanodomains close enough to each other that confocal can't resolve them,
# but STED can. I will make columns for my nanodomains

left_column_center = 28
right_column_center = 35
center_column_center = 31

left_column = [(40, left_column_center - 1), (41, left_column_center - 1),
               (42, left_column_center - 1), (43, left_column_center - 1),
               (39, left_column_center), (40, left_column_center), (41, left_column_center),
               (42, left_column_center), (43, left_column_center), (44, left_column_center),
               (40, left_column_center + 1), (41, left_column_center + 1),
               (42, left_column_center + 1), (43, left_column_center + 1)]

right_column = [(40, right_column_center - 1), (41, right_column_center - 1),
                (42, right_column_center - 1), (43, right_column_center - 1),
                (39, right_column_center), (40, right_column_center), (41, right_column_center),
                (42, right_column_center), (43, right_column_center), (44, right_column_center),
                (40, right_column_center + 1), (41, right_column_center + 1),
                (42, right_column_center + 1), (43, right_column_center + 1)]

center_column = [(40, center_column_center - 1), (41, center_column_center - 1),
                 (42, center_column_center - 1), (43, center_column_center - 1),
                 (39, center_column_center), (40, center_column_center), (41, center_column_center),
                 (42, center_column_center), (43, center_column_center), (44, center_column_center),
                 (40, center_column_center + 1), (41, center_column_center + 1),
                 (42, center_column_center + 1), (43, center_column_center + 1)]

for pixel in left_column:
    fluo_gt[pixel] += 45
for pixel in right_column:
    fluo_gt[pixel] += 25
for pixel in center_column:
    fluo_gt[pixel] += 25

plt.imshow(fluo_gt)
plt.show()

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
p_ex = 1e-6
# p_sted = 30e-3
sted_powers = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.03, 0.08, 0.1, 0.3, 0.8]
pdt = 10e-6
roi = 'max'

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap = base.Datamap(fluo_gt, dpxsz)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
i_ex, i_sted, psf_det = microscope.cache(datamap.pixelsize, save_cache=True)
datamap.set_roi(i_ex, roi)

list_dmaps, list_acqs = [], []

for idx, p_sted in enumerate(sted_powers):

    print(f"Starting acquisition {idx + 1} with STED power {p_sted} ...")
    acq, bleached, _ = microscope.get_signal_and_bleach(datamap, dpxsz, pdt, p_ex, p_sted,
                                                        update=False)

    list_dmaps.append(bleached["base"][datamap.roi])
    list_acqs.append(acq)

dmaps_stack = np.stack(list_dmaps)
acqs_stack = np.stack(list_acqs)
np.save(save_path + f"/bleached.npy", dmaps_stack)
np.save(save_path + f"/acquisitions.npy", acqs_stack)

# fig, axes = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True)
#
# axes[0, 0].imshow(datamap.whole_datamap[datamap.roi])
# axes[0, 0].set_title(f"Datamap before acquisition")
#
# axes[0, 1].imshow(sted_bleached["base"][datamap.roi])
# axes[0, 1].set_title(f"Datamap after STED acq (bleach = {bleach})")
#
# axes[1, 0].imshow(confoc_acq)
# axes[1, 0].set_title(f"Confocal signal")
#
# axes[1, 1].imshow(sted_acq)
# axes[1, 1].set_title(f"STED signal")
#
# plt.show()
