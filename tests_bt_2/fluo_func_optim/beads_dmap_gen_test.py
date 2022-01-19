import numpy as np
from matplotlib import pyplot as plt
import tifffile
import os
import pandas as pd
from skimage.feature import peak_local_max
from gym_sted.utils import get_foreground
from pysted import base, utils


data_dir = os.path.expanduser(os.path.join("~", "Documents", "research", "h22_code", "albert_beads_exps",
                                           "bandit-optimization-experiments", "2021-09-14_grid_articacts",
                                           "four_params_five_reps_2timesDwell"))

results_table = pd.read_csv(data_dir + "/results_df.csv")

p_ex_max = 100 * 1.5135e-6
p_sted_max = 100 * 1.7681e-3

img_idx = 99
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
        635: 0.1e-21, #Table S3, Oracz et al., nature 2017
        750: 3.5e-25,  # (1 photon exc abs) Table S3, Oracz et al., nature 2017
    },
    "sigma_ste": {
        750: 4.8e-22, #Table S3, Oracz et al., nature 2017
    },
    "sigma_tri": 10.14e-21, # COPIED FROM BEFORE
#     "tau": 3e-09,
    "tau" : 3.5e-9, # @646nm, ATTO Fluorescent labels, ATTO-TEC GmbH catalog 2016/2018
    "tau_vib": 1.0e-12, #t_vib, Table S3, Oracz et al., nature 2017
    "tau_tri": 1.2e-6, # COPIED FROM BEFORE
    "phy_react": {
#         488: 0.008e-5,
#         575: 0.008e-8,
        635: 0.008e-5, # COPIED FROM BEFORE
        750:  0.008e-8, # COPIED FROM BEFORE
    },
    "k_isc": 0.48e+6 # COPIED FROM BEFORE
}

action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350.0e-3},
    "p_ex" : {"low" : 0., "high" : 250.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}

pixelsize = 20e-9
laser_ex = base.GaussianBeam(635e-9)
laser_sted = base.DonutBeam(750e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
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

fig, axes = plt.subplots(1, 2)
axes[0].imshow(dmap.whole_datamap[dmap.roi])
axes[1].imshow(dmap_fg.whole_datamap[dmap_fg.roi])
plt.show()

# oker là jserais rendu à faire des acqs là dessus pi tester ça avant de pitcher ça dans une loop d'optim
