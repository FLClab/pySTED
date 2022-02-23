import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from pysted import base, utils
from pysted import exp_data_gen as dg

from gym_sted.rewards.objectives_timed import find_nanodomains, Signal_Ratio, Resolution
from gym_sted.utils import get_foreground
import metrics
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, default="f1")
parser.add_argument("--fluo", type=str, default="default")
args = parser.parse_args()


def rescale_func(val_to_scale, current_range_min, current_range_max, new_min, new_max):
    return ( (val_to_scale - current_range_min) / (current_range_max - current_range_min) ) * (new_max - new_min) + new_min


action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350.0e-3},
    "p_ex" : {"low" : 0., "high" : 250.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}
multipliers = np.array([0., 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.])
tick_labels = ["0.00 %", "0.78%", "1,56%", "3.12%", "6.25%", "12.5%", "25.0%", "50.0%", "100.%"]
pdt_values = multipliers * action_spaces["pdt"]["high"]
pdt_values = rescale_func(pdt_values, np.min(pdt_values), np.max(pdt_values),
                          action_spaces["pdt"]["low"], action_spaces["pdt"]["high"])
p_ex_values = multipliers * action_spaces["p_ex"]["high"]
p_sted_values = multipliers * action_spaces["p_sted"]["high"]

data_dir = f"./tests_bt_2/save_dir/fluo_{args.fluo}/"

f1_score_avg = np.load(data_dir + "f1_score_values_avg_baseline.npy")
n_molecules_avg = np.load(data_dir + "n_molecules_values_avg_baseline.npy")
n_photons_values_avg = np.load(data_dir + "n_photons_values_avg_baseline.npy")

# faut jrefasse mon cube pour qu'il soit (9, 9, 9)
molecules_left_cube = n_molecules_avg[:, :, :, 1] / n_molecules_avg[:, :, :, 0]
photobleaching_cube = (n_molecules_avg[:, :, :, 0] - n_molecules_avg[:, :, :, 1]) / n_molecules_avg[:, :, :, 0]

values_to_plot_dict = {
    "f1": f1_score_avg,
    "photons": n_photons_values_avg,
    "photobleaching": photobleaching_cube,
    "molecules": molecules_left_cube
}

# dimensions are [pdt, p_ex, p_sted]
# 1st square : max proj along pdt ==> rows are p_ex, cols are p_sted
pdt_max_proj = np.max(values_to_plot_dict[args.metric], axis=0)
# 2nd square : max proj along p_ex ==> rows are pdt, cols are p_sted
pex_max_proj = np.max(values_to_plot_dict[args.metric], axis=1)
# 3rd square : max proj along p_sted ==> rows are pdtm cols are p_sted
psted_max_proj = np.max(values_to_plot_dict[args.metric], axis=2)

if args.metric == "f1":
    vmax = 1
elif args.metric == "photobleaching":
    vmax = 1
elif args.metric == "molecules":
    vmax = 1
elif args.metric == "photons":
    vmax = np.max(values_to_plot_dict[args.metric])
else:
    # uh oh
    vmax = 0

imshow_kwargs = {"vmin": 0,
                 "vmax": vmax,
                 "cmap": "inferno"}

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

imshow_0 = axes[0].imshow(pdt_max_proj, **imshow_kwargs)
axes[0].set_ylabel("p_ex")
axes[0].set_yticks(np.arange(0, 9))
axes[0].set_yticklabels(tick_labels)
axes[0].set_xlabel("p_sted")
axes[0].set_xticks(np.arange(0, 9))
axes[0].set_xticklabels(tick_labels, rotation=45)
divider0 = make_axes_locatable(axes[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
fig.colorbar(imshow_0, cax=cax0)


imshow_1 = axes[1].imshow(pex_max_proj, **imshow_kwargs)
axes[1].set_ylabel("pdt")
axes[1].set_yticks(np.arange(0, 9))
axes[1].set_yticklabels(tick_labels)
axes[1].set_xlabel("p_sted")
axes[1].set_xticks(np.arange(0, 9))
axes[1].set_xticklabels(tick_labels, rotation=45)
divider1 = make_axes_locatable(axes[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
fig.colorbar(imshow_1, cax=cax1)

imshow_2 = axes[2].imshow(psted_max_proj, **imshow_kwargs)
axes[2].set_ylabel("pdt")
axes[2].set_yticks(np.arange(0, 9))
axes[2].set_yticklabels(tick_labels)
axes[2].set_xlabel("p_ex")
axes[2].set_xticks(np.arange(0, 9))
axes[2].set_xticklabels(tick_labels, rotation=45)
divider2 = make_axes_locatable(axes[2])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
fig.colorbar(imshow_2, cax=cax2)

if not os.path.exists(data_dir + "/results"):
    os.mkdir(data_dir + "/results")

if args.fluo == "default":
    fluo_params = "default"
elif args.fluo == "1":
    fluo_params = "x10"
elif args.fluo == "2":
    fluo_params = "x100"
else:
    fluo_params = "OOPS"

plot_cube = np.array([pdt_max_proj, pex_max_proj, psted_max_proj])
print(plot_cube[0, -1, -3])
fig.suptitle(f"fluo_params : {fluo_params}, \n"
             f"metric : {args.metric}, \n"
             f"pdt range : [{action_spaces['pdt']['low']}, {action_spaces['pdt']['high']}] s, \n"
             f"p_ex range : [{action_spaces['p_ex']['low']}, {action_spaces['p_ex']['high']}] W, \n"
             f"p_sted range : [{action_spaces['p_sted']['low']}, {action_spaces['p_sted']['high']}] W")
plt.tight_layout()
plt.savefig(data_dir + f"results/{args.metric}.pdf")
plt.savefig(data_dir + f"results/{args.metric}.png")
plt.show()
