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
parser.add_argument("--acq_time", type=int, default=1, help="Acquisition time (in seconds)")
args = parser.parse_args()


save_path = r"D:\SCHOOL\Maitrise\E2021\research\iid_pres\figs\simulated_data"
if not os.path.exists(save_path):
    os.mkdir(save_path)

print("Setting up the datamap and its flashes ...")
# Get light curves stuff to generate the flashes later
# event_file_path = "flash_files/stream1_events.txt"
# video_file_path = "flash_files/stream1.tif"
curves_path = "flash_files/events_curves.npy"

# Generate a datamap
frame_shape = (64, 64)
ensemble_func, synapses_list = utils.generate_synaptic_fibers(frame_shape, (9, 55), (3, 10), (2, 5),
                                                              seed=27)
# Build a dictionnary corresponding synapses to a bool saying if they are currently flashing or not
# They all start not flashing
flat_synapses_list = [item for sublist in synapses_list for item in sublist]

poils_frame = ensemble_func.return_frame().astype(int)

# plt.imshow(poils_frame)
# plt.show()
# exit()

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
bleach = args.bleach
p_ex = np.ones(frame_shape) * 1e-6
p_sted = 30e-3
min_pdt = 1e-6   # le min pdt est 1 us
# pdt = np.ones(frame_shape) * min_pdt
# higher_pdt_pixels = utils.pixel_sampling(poils_frame, mode="checkers")
# for row, col in higher_pdt_pixels:
#     pdt[row, col] = 10e-6
pdt = np.ones(frame_shape) * 10e-6
roi = 'max'
acquisition_time = 5
flash_prob = 0.05   # every iteration, all synapses will have a 5% to start flashing
flash_seed = args.flash_seed

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
temporal_datamap = base.TemporalDatamap(poils_frame, dpxsz, flat_synapses_list)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)
i_ex, _, _ = microscope.cache(temporal_datamap.pixelsize)
temporal_datamap = base.TemporalDatamap(poils_frame, dpxsz, flat_synapses_list)
temporal_datamap.set_roi(i_ex, roi)
temporal_datamap.create_t_stack_dmap(acquisition_time, min_pdt, (10, 1.5), curves_path, flash_prob)

vmax = np.max(temporal_datamap["base"]) + np.max(temporal_datamap.flash_tstack)

for i in range(temporal_datamap.flash_tstack.shape[0]):
    dmap_to_show = temporal_datamap["base"][temporal_datamap.roi] + temporal_datamap.flash_tstack[i][temporal_datamap.roi]
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(temporal_datamap["base"][temporal_datamap.roi] + temporal_datamap.flash_tstack[i][temporal_datamap.roi],
               cmap="inferno", vmax=vmax)
    fig.savefig(save_path + f"/flashing_seed_27/{i}.png")
    plt.close()
