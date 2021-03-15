import numpy as np
import tqdm
from pysted import base, utils
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
dpxsz = 10e-9
bleach = True
p_ex = 1e-6
p_sted = 30e-3
pdt = 10e-6
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
temporal_datamap.set_roi(i_ex, roi)
temporal_datamap.create_t_stack_dmap(acquisition_time, pdt, (10, 1.5), event_file_path, video_file_path, flash_prob,
                                     i_ex, roi)

# for flash in temporal_datamap.flash_tstack:
#     wdm = temporal_datamap.base_datamap + flash
#     with np.errstate(divide='ignore', invalid='ignore'):
#         base_ratio = temporal_datamap.base_datamap / wdm
#         flash_ratio = flash / wdm
#     base_ratio[np.isnan(base_ratio)] = 0
#     flash_ratio[np.isnan(flash_ratio)] = 0
#
#     reconstruction = base_ratio * wdm + flash_ratio * wdm
#
#     fig, axes = plt.subplots(1, 2)
#     axes[0].imshow(wdm[temporal_datamap.roi])
#     axes[1].imshow(reconstruction[temporal_datamap.roi])
#     fig.suptitle(f"mse = {utils.mse_calculator(wdm, reconstruction)}")
#     plt.show()
# exit()

# je vais me garder une liste des flash intacte pour comparer aux flashes que je bleach
flash_tsack_static = np.copy(temporal_datamap.flash_tstack)

# pour le premier test, je commence au temps 0
for flash_tstep_idx in range(temporal_datamap.flash_tstack.shape[0]):
    # print(temporal_datamap.whole_datamap.dtype)
    # print(temporal_datamap.base_datamap.dtype)
    # print(temporal_datamap.flash_tstack[flash_tstep_idx].dtype)
    temporal_datamap.whole_datamap = temporal_datamap.base_datamap + temporal_datamap.flash_tstack[flash_tstep_idx]

    # faire l'acq sur la datamap comme je ferais normalement
    # je m'attends à ce que t_d.whole_datamap bleach, mais que t_d.base et t_d.flash restent intactes

    signal, bleached, intensity = microscope.get_signal_and_bleach_fast(temporal_datamap, dpxsz, pdt, p_ex, p_sted,
                                                                        bleach=True, update=False)

    # calculer le ratio de molécules appartenant à la base et au flash pour ce t step
    with np.errstate(divide='ignore', invalid='ignore'):
        base_ratio = temporal_datamap.base_datamap / temporal_datamap.whole_datamap
        flash_ratio = temporal_datamap.flash_tstack[flash_tstep_idx] / temporal_datamap.whole_datamap
    base_ratio[np.isnan(base_ratio)] = 0
    flash_ratio[np.isnan(flash_ratio)] = 0

    # séparer le bleaching en bleaching à la base et bleaching au flash
    base_bleached = base_ratio * bleached
    flash_bleached = flash_ratio * bleached

    # pour t_d.base, elle devient simplement égale à base_bleached
    temporal_datamap.base_datamap = np.copy(base_bleached).astype(int)   # jfais une copie au cas ou dk if actually necessary

    # pour les datamaps de flash, je dois multiplier à partir de l'idx courant par le ratio de survivants
    # si mon premier flash est 0 partout, j'ai un survival de 0 que je multiplie à tout le reste -> tout devient 0
    if not np.array_equal(flash_ratio, np.zeros(flash_ratio.shape)):
        with np.errstate(divide='ignore', invalid='ignore'):
            flash_survival = flash_bleached / temporal_datamap.flash_tstack[flash_tstep_idx]
        flash_survival[np.isnan(flash_survival)] = 1
        temporal_datamap.flash_tstack[flash_tstep_idx:] = np.multiply(temporal_datamap.flash_tstack[flash_tstep_idx:],
                                                                      flash_survival)
        temporal_datamap.flash_tstack[flash_tstep_idx:] = np.rint(temporal_datamap.flash_tstack[flash_tstep_idx:])

    plt.imshow(temporal_datamap.whole_datamap[temporal_datamap.roi])
    plt.show()

    # for i in range(temporal_datamap.flash_tstack[flash_tstep_idx:].shape[0]):
    #     fig, axes = plt.subplots(1, 2)
    #
    #     axes[0].imshow(flash_tsack_static[flash_tstep_idx + i])
    #     axes[0].set_title(f"static flash for tstep = {flash_tstep_idx + i}")
    #
    #     axes[1].imshow(temporal_datamap.flash_tstack[flash_tstep_idx + i])
    #     axes[1].set_title(f"bleached flash for tstep = {flash_tstep_idx + i}")
    #
    #     plt.show()


mse = utils.mse_calculator(temporal_datamap.whole_datamap[temporal_datamap.roi], bleached[temporal_datamap.roi])

fig, axes = plt.subplots(1, 2)

axes[0].imshow(temporal_datamap.whole_datamap[temporal_datamap.roi])
axes[0].set_title(f"temporal_datamap")

axes[1].imshow(bleached[temporal_datamap.roi])
axes[1].set_title(f"bleached")

fig.suptitle(f"mse = {mse}")
plt.show()
