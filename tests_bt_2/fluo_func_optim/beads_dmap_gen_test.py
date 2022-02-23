import argparse
import numpy as np
from matplotlib import pyplot as plt
import tifffile
import os
import pandas as pd
from skimage.feature import peak_local_max
from gym_sted.utils import get_foreground
from pysted import base, utils


def line_step_pixel_list_builder(dmap, line_step=1):
    """
    Builds a pixel_list with line_repetitions
    :param dmap: The datamap that will be acquired on (Datamap object)
    :param line_sted: The number of line repetitions. If 1, a normal raster scan pixel_list will be returned
    :returns: The pixel_list with appropriate number of line repetitions
    """
    # might be a more efficient way to this without such loops but I do not care for now :)
    n_rows, n_cols = dmap.whole_datamap[dmap.roi].shape
    pixel_list = []
    for row in range(n_rows):
        row_pixels = []
        for col in range(n_cols):
            pixel_list.append((row, col))
            row_pixels.append((row, col))
        if line_step > 1:
            for i in range(line_step - 1):   # - 1 cause the row is already there once at this stage
                for pixel in row_pixels:
                    pixel_list.append(pixel)
    return pixel_list


parser = argparse.ArgumentParser()
parser.add_argument("--img", type=int, default=0)
args = parser.parse_args()

data_dir = os.path.expanduser(os.path.join("~", "Documents", "research", "h22_code", "albert_beads_exps",
                                           "bandit-optimization-experiments", "2021-09-14_grid_articacts",
                                           "four_params_five_reps_2timesDwell"))

results_table = pd.read_csv(data_dir + "/results_df.csv")

p_ex_max = 100 * 1.5135e-6
p_sted_max = 100 * 1.7681e-3

img_idx = args.img
conf1 = tifffile.imread(data_dir + f"/conf1/{img_idx}.tiff")
sted = tifffile.imread(data_dir + f"/sted/{img_idx}.tiff")
conf2 = tifffile.imread(data_dir + f"/conf2/{img_idx}.tiff")

images = np.dstack([conf1, sted, conf2])
conf1_fg_bool = get_foreground(conf1)
conf1_fg = conf1 * conf1_fg_bool
conf2_fg_bool = get_foreground(conf2)
conf2_fg = conf2 * conf2_fg_bool
sted_fg_bool = get_foreground(sted)
sted_fg = sted * sted_fg_bool

photobleaching_bt = 100 * np.sum(conf2_fg) / np.sum(conf1_fg)
photobleaching_alb = 100 * results_table.at[img_idx, 'Bleach']

imshow_opts = {
    "vmax": np.max(images),
    "cmap": "hot"
}
# fig, axes = plt.subplots(1, 3, figsize=(12, 8))
#
# conf1_imshow = axes[0].imshow(conf1, **imshow_opts)
# axes[0].set_title(f"Confocal 1")
# fig.colorbar(conf1_imshow, ax=axes[0], fraction=0.05, pad=0.05)
#
# sted_imshow = axes[1].imshow(sted, **imshow_opts)
# axes[1].set_title(f"STED")
# fig.colorbar(sted_imshow, ax=axes[1], fraction=0.05, pad=0.05)
#
# conf2_imshow = axes[2].imshow(conf2, **imshow_opts)
# axes[2].set_title(f"Confocal 2")
# fig.colorbar(conf2_imshow, ax=axes[2], fraction=0.05, pad=0.05)
#
# for ax in axes:
#     ax.set_axis_off()
#
# fig.suptitle(f"img size : {conf1.shape}, \n"
#              f"pdt = {results_table.at[img_idx, 'dwelltime']} s, \n"
#              f"p_ex = {0.01 * results_table.at[img_idx, 'p_ex'] * p_ex_max} W, \n"
#              f"p_sted = {0.01 * results_table.at[img_idx, 'p_sted'] * p_sted_max} W, \n"
#              f"line_step = {results_table.at[img_idx, 'line_step']}, \n"
#              f"photon count in STED = {np.sum(sted)} photons, \n"
#              f"photobleaching (bt) = {photobleaching_bt} % , \n"
#              f"photobleaching(albert) = {photobleaching_alb} %")
# plt.tight_layout()
# plt.show()
# exit()

# now let's try and build a datamap from the STED img ...

# tester ça sur le foreground de la STED à la place ?

# avec ces params là ça a l'air pas trop pire, semble mettre les billes aux bons endroits, y'en manque une couple tho :(
min_distance = 5
threshold_rel = 0.1
bead_positions_sted = peak_local_max(sted, min_distance=min_distance, threshold_rel=threshold_rel)
bead_positions_sted_fg = peak_local_max(sted_fg, min_distance=min_distance, threshold_rel=threshold_rel)

# sted_stack = np.dstack([sted, sted_fg])
# fig, axes = plt.subplots(1, 2)
#
# sted_imshow = axes[0].imshow(sted, cmap="hot", vmax=np.max(sted_stack))
# axes[0].scatter(bead_positions_sted[:, 1], bead_positions_sted[:, 0])
# axes[0].set_title(f"Normal STED")
# fig.colorbar(sted_imshow, ax=axes[0], fraction=0.05, pad=0.05)
# axes[0].set_axis_off()
#
# sted_fg_imshow = axes[1].imshow(sted_fg, cmap="hot", vmax=np.max(sted_stack))
# axes[1].scatter(bead_positions_sted_fg[:, 1], bead_positions_sted_fg[:, 0])
# axes[1].set_title(f"Foreground STED")
# fig.colorbar(sted_fg_imshow, ax=axes[1], fraction=0.05, pad=0.05)
# axes[1].set_axis_off()
#
# plt.tight_layout()
# plt.show()

# ok now let's create a microscope and shit and datamaps based on the bead position detection

egfp = {
#     "lambda": 535e-9,
    "lambda_": 635e-9, # TODO: verify ok to change like that...
    "qy": 0.6, # COPIED FROM BEFORE
    "sigma_abs": {
        635: 0.42e-22, #Table S3, Oracz et al., nature 2017   ALBERT USES 0.1e-21
        750: 3.5e-25,  # (1 photon exc abs) Table S3, Oracz et al., nature 2017
    },
    "sigma_ste": {
        750: 2.8e-22, #Table S3, Oracz et al., nature 2017   ALBERT USES 4.8e-22
    },
    "sigma_tri": 10.14e-21, # COPIED FROM BEFORE
#     "tau": 3e-09,
    "tau" : 3.5e-9, # @646nm, ATTO Fluorescent labels, ATTO-TEC GmbH catalog 2016/2018
    "tau_vib": 1.0e-12, #t_vib, Table S3, Oracz et al., nature 2017
    "tau_tri": 1.2e-6, # COPIED FROM BEFORE
    "phy_react": {
#         488: 0.008e-5,
#         575: 0.008e-8,
        635: 0.002e-7,   # ALBERT USES 0.008e-5
        750:  0.002e-10,   # ALBERT USES 0.008e-8
    },
    "k_isc": 0.48e+6 # COPIED FROM BEFORE
}

action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350.0e-3},
    "p_ex" : {"low" : 0., "high" : 250.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}

optim_params = {
    "pdt": 10e-6,
    "p_ex": 0.25e-3,
    "p_sted": 87.5e-3
}

pixelsize = 20e-9
laser_ex = base.GaussianBeam(635e-9)
laser_sted = base.DonutBeam(750e-9, zero_residual=0)
detector = base.Detector(noise=True, background=4)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
# pk faut jmette load_cache=True pour que ça marche ???
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
i_ex, i_sted, _ = microscope.cache(pixelsize, save_cache=True)

molecule_disposition = np.zeros(sted.shape)
for col, row in bead_positions_sted:
    molecule_disposition[row, col] += 150
molecule_disposition_fg = np.zeros(sted_fg.shape)
for col, row in bead_positions_sted_fg:
    molecule_disposition_fg[row, col] += 150

dmap = base.TemporalDatamap(molecule_disposition, pixelsize, None)
dmap.set_roi(i_ex, "max")

dmap_fg = base.TemporalDatamap(molecule_disposition_fg, pixelsize, None)
dmap_fg.set_roi(i_ex, "max")

# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(dmap.whole_datamap[dmap.roi])
# axes[1].imshow(dmap_fg.whole_datamap[dmap_fg.roi])
# plt.show()

alb_params = {
    "pdt": results_table.at[img_idx, 'dwelltime'],   # results_table.at[img_idx, 'Bleach']
    "p_ex": 0.01 * results_table.at[img_idx, 'p_ex'] * p_ex_max,
    "p_sted": 0.01 * results_table.at[img_idx, 'p_sted'] * p_sted_max
}

conf_params = {   # I should ask Albert what the params for conf1 / conf2 were
    "pdt": 10e-6,
    "p_ex": 0.4 * p_ex_max,
    "p_sted": 0.0
}

# on dirait jsuis pas sur si mon implem a actually fait de quoi ou pas ???
pixel_list = line_step_pixel_list_builder(dmap, line_step=results_table.at[img_idx, 'line_step'])

# pixel_list_ls1 = line_step_pixel_list_builder(dmap, line_step=1)   # results_table.at[img_idx, 'line_step']
# pixel_list_ls2 = line_step_pixel_list_builder(dmap, line_step=2)
# pixel_list_ls3 = line_step_pixel_list_builder(dmap, line_step=3)
#
# acq_ls1, bleached_ls1, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **alb_params,
#                                                             bleach=True, update=False,
#                                                             pixel_list=pixel_list_ls1, filter_bypass=True,
#                                                             seed=42)
# acq_ls2, bleached_ls2, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **alb_params,
#                                                             bleach=True, update=False,
#                                                             pixel_list=pixel_list_ls2, filter_bypass=True,
#                                                             seed=42)
# acq_ls3, bleached_ls3, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **alb_params,
#                                                             bleach=True, update=False,
#                                                             pixel_list=pixel_list_ls3, filter_bypass=True,
#                                                             seed=42)
#
# fig, axes = plt.subplots(3, 3, figsize=(15, 15))
#
# dmap_init1_imshow = axes[0, 0].imshow(dmap.whole_datamap[dmap.roi], vmax=np.max(dmap.whole_datamap))
# axes[0, 0].set_title(f"dmap before acq with line_step = 1")
# fig.colorbar(dmap_init1_imshow, ax=axes[0, 0], fraction=0.05, pad=0.05)
# acq_ls1_imshow = axes[0, 1].imshow(acq_ls1, cmap="hot")
# axes[0, 1].set_title(f"acq with ls = 1")
# fig.colorbar(acq_ls1_imshow, ax=axes[0, 1], fraction=0.05, pad=0.05)
# dmap_post1_imshow = axes[0, 2].imshow(bleached_ls1["base"][dmap.roi], vmax=np.max(dmap.whole_datamap))
# axes[0, 2].set_title(f"dmap after acq with line_step = 1")
# fig.colorbar(dmap_post1_imshow, ax=axes[0, 2], fraction=0.05, pad=0.05)
#
# dmap_init2_imshow = axes[1, 0].imshow(dmap.whole_datamap[dmap.roi], vmax=np.max(dmap.whole_datamap))
# axes[1, 0].set_title(f"dmap before acq with line_step = 2")
# fig.colorbar(dmap_init2_imshow, ax=axes[1, 0], fraction=0.05, pad=0.05)
# acq_ls2_imshow = axes[1, 1].imshow(acq_ls2, cmap="hot")
# axes[1, 1].set_title(f"acq with ls = 2")
# fig.colorbar(acq_ls2_imshow, ax=axes[1, 1], fraction=0.05, pad=0.05)
# dmap_post2_imshow = axes[1, 2].imshow(bleached_ls2["base"][dmap.roi], vmax=np.max(dmap.whole_datamap))
# axes[1, 2].set_title(f"dmap after acq with line_step = 2")
# fig.colorbar(dmap_post2_imshow, ax=axes[1, 2], fraction=0.05, pad=0.05)
#
# dmap_init3_imshow = axes[2, 0].imshow(dmap.whole_datamap[dmap.roi], vmax=np.max(dmap.whole_datamap))
# axes[2, 0].set_title(f"dmap before acq with line_step = 3")
# fig.colorbar(dmap_init3_imshow, ax=axes[2, 0], fraction=0.05, pad=0.05)
# acq_ls3_imshow = axes[2, 1].imshow(acq_ls3, cmap="hot")
# axes[2, 1].set_title(f"acq with ls = 3")
# fig.colorbar(acq_ls3_imshow, ax=axes[2, 1], fraction=0.05, pad=0.05)
# dmap_post3_imshow = axes[2, 2].imshow(bleached_ls3["base"][dmap.roi], vmax=np.max(dmap.whole_datamap))
# axes[2, 2].set_title(f"dmap after acq with line_step = 3")
# fig.colorbar(dmap_post3_imshow, ax=axes[2, 2], fraction=0.05, pad=0.05)
#
# plt.tight_layout()
# plt.show()
# plt.close(fig)
# exit()

# oker là jserais rendu à faire des acqs là dessus pi tester ça avant de pitcher ça dans une loop d'optim

simul_conf1, _, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                     bleach=False, update=False)
acq, bleached, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **alb_params,
                                                    bleach=True, update=True,
                                                    pixel_list=pixel_list, filter_bypass=True)
simul_conf2, _, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                      bleach=False, update=False)

simul_fg_conf1, _, _ = microscope.get_signal_and_bleach(dmap_fg, dmap_fg.pixelsize, **conf_params,
                                                        bleach=False, update=False)
acq_fg, bleached_fg, _ = microscope.get_signal_and_bleach(dmap_fg, dmap_fg.pixelsize, **alb_params,
                                                          bleach=True, update=True,
                                                          pixel_list=pixel_list, filter_bypass=True)
simul_fg_conf2, _, _ = microscope.get_signal_and_bleach(dmap_fg, dmap_fg.pixelsize, **conf_params,
                                                        bleach=False, update=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 15))

images = np.dstack([conf1, sted, conf2])
conf1_imshow = axes[0].imshow(conf1, cmap="hot", vmax=np.max(images))
axes[0].set_title(f"Albert confocal 1")
fig.colorbar(conf1_imshow, ax=axes[0], fraction=0.05, pad=0.05)

sted_imshow = axes[1].imshow(sted, cmap="hot", vmax=np.max(images))
axes[1].set_title(f"Albert acquisition")
fig.colorbar(sted_imshow, ax=axes[1], fraction=0.05, pad=0.05)

conf2_imshow = axes[2].imshow(conf2, cmap="hot", vmax=np.max(images))
axes[2].set_title(f"Albert confocal 2")
fig.colorbar(conf2_imshow, ax=axes[2], fraction=0.05, pad=0.05)

alb_params = {
    "pdt": results_table.at[img_idx, 'dwelltime'],   # results_table.at[img_idx, 'Bleach']
    "p_ex": 0.01 * results_table.at[img_idx, 'p_ex'] * p_ex_max,
    "p_sted": 0.01 * results_table.at[img_idx, 'p_sted'] * p_sted_max
}
fig.suptitle(f"Imaging parameters : \n"
             f"pdt : {alb_params['pdt']} s, \n"
             f"p_ex : {alb_params['p_ex']} W, \n"
             f"p_sted : {alb_params['p_sted']} W, \n"
             f"line_step = {results_table.at[img_idx, 'line_step']}")
plt.tight_layout()
plt.show()
plt.close(fig)
exit()

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

images = np.dstack([conf1, sted, conf2])
conf1_imshow = axes[0, 0].imshow(conf1, cmap="hot", vmax=np.max(images))
axes[0, 0].set_title(f"Albert confocal 1")
fig.colorbar(conf1_imshow, ax=axes[0, 0], fraction=0.05, pad=0.05)

sted_imshow = axes[0, 1].imshow(sted, cmap="hot", vmax=np.max(images))
axes[0, 1].set_title(f"Albert acquisition")
fig.colorbar(sted_imshow, ax=axes[0, 1], fraction=0.05, pad=0.05)

conf2_imshow = axes[0, 2].imshow(conf2, cmap="hot", vmax=np.max(images))
axes[0, 2].set_title(f"Albert confocal 2")
fig.colorbar(conf2_imshow, ax=axes[0, 2], fraction=0.05, pad=0.05)

simul_images = np.dstack([simul_conf1, acq, simul_conf2])
simul_conf1_imshow = axes[1, 0].imshow(simul_conf1, cmap="hot", vmax=np.max(simul_images))
axes[1, 0].set_title(f"Simulated confocal 1")
fig.colorbar(simul_conf1_imshow, ax=axes[1, 0], fraction=0.05, pad=0.05)

simul_acq_imshow = axes[1, 1].imshow(acq, cmap="hot", vmax=np.max(simul_images))
axes[1, 1].set_title(f"Simulated acquisition")
fig.colorbar(simul_acq_imshow, ax=axes[1, 1], fraction=0.05, pad=0.05)

simul_conf2_imshow = axes[1, 2].imshow(simul_conf2, cmap="hot", vmax=np.max(simul_images))
axes[1, 2].set_title(f"Simulated confocal 2")
fig.colorbar(simul_conf2_imshow, ax=axes[1, 2], fraction=0.05, pad=0.05)

simul_fg_images = np.dstack([simul_fg_conf1, acq_fg, simul_fg_conf2])
simul_fg_conf1_imshow = axes[2, 0].imshow(simul_fg_conf1, cmap="hot", vmax=np.max(simul_fg_images))
axes[2, 0].set_title(f"Simulated confocal 1 (using fg to construct dmap)")
fig.colorbar(simul_fg_conf1_imshow, ax=axes[2, 0], fraction=0.05, pad=0.05)

simul_fg_acq_imshow = axes[2, 1].imshow(acq_fg, cmap="hot", vmax=np.max(simul_fg_images))
axes[2, 1].set_title(f"Simulated acquisition (using fg to construct dmap)")
fig.colorbar(simul_fg_acq_imshow, ax=axes[2, 1], fraction=0.05, pad=0.05)

simul_fg_conf2_imshow = axes[2, 2].imshow(simul_fg_conf2, cmap="hot", vmax=np.max(simul_fg_images))
axes[2, 2].set_title(f"Simulated confocal 2 (using fg to construct dmap)")
fig.colorbar(simul_fg_conf2_imshow, ax=axes[2, 2], fraction=0.05, pad=0.05)

plt.tight_layout()
plt.show()
plt.close(fig)

# images = np.dstack([conf1, sted, conf2])
# conf1_fg_bool = get_foreground(conf1)
# conf1_fg = conf1 * conf1_fg_bool
# conf2_fg_bool = get_foreground(conf2)
# conf2_fg = conf2 * conf2_fg_bool
# sted_fg_bool = get_foreground(sted)
# sted_fg = sted * sted_fg_bool
# photobleaching_bt = 100 * np.sum(conf2_fg) / np.sum(conf1_fg)

# ok à partir d'ici je vais assumer qu'on veut reconstruire la datamap à partir de l'image de forground de la sted,
# ça fait plus de sens comme ça imo

# compute photobleaching and photon count
simul_conf1_fg_bool = get_foreground(simul_fg_conf1)
simul_conf1_fg = simul_fg_conf1 * simul_conf1_fg_bool
simul_conf2_fg_bool = get_foreground(simul_fg_conf2)
simul_conf2_fg = simul_fg_conf2 * simul_conf2_fg_bool
simul_sted_fg_bool = get_foreground(acq_fg)
simul_sted_fg = acq_fg * simul_sted_fg_bool

photobleaching_bt_simul = 100 * np.sum(simul_conf2_fg) / np.sum(simul_conf1_fg)
photon_count_real = np.sum(sted)
photon_count_bt_simul = np.sum(acq_fg)

print(f"photobleaching in Albert's real acquisitions : {photobleaching_bt} %")
print(f"photobleaching in bt's simulation of Albert's acq : {photobleaching_bt_simul} %")
print(f"------------------------------------------------")
print(f"photon count in Albert's real acquisition : {photon_count_real}")
print(f"photon count in bt's simulation of Albert's acq : {photon_count_bt_simul}")
