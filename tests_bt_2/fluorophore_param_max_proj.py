import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from pysted import base, utils
from pysted import exp_data_gen as dg

from gym_sted.rewards.objectives_timed import find_nanodomains, Signal_Ratio, Resolution
from gym_sted.utils import get_foreground
import metrics


def rescale_func(val_to_scale, current_range_min, current_range_max, new_min, new_max):
    return ( (val_to_scale - current_range_min) / (current_range_max - current_range_min) ) * (new_max - new_min) + new_min


action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350.0e-3},
    "p_ex" : {"low" : 0., "high" : 250.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}
multipliers = np.array([0., 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.])
pdt_values = multipliers * action_spaces["pdt"]["high"]
pdt_values = rescale_func(pdt_values, np.min(pdt_values), np.max(pdt_values),
                          action_spaces["pdt"]["low"], action_spaces["pdt"]["high"])
p_ex_values = multipliers * action_spaces["p_ex"]["high"]
p_sted_values = multipliers * action_spaces["p_sted"]["high"]

data_dir = "./tests_bt_2/save_dir/fluo_default/"

f1_score_avg = np.load(data_dir + "f1_score_values_avg_baseline.npy")
n_molecules_avg = np.load(data_dir + "n_molecules_values_avg_baseline.npy")
n_photons_values_avg = np.load(data_dir + "n_photons_values_avg_baseline.npy")

# faut jrefasse mon cube pour qu'il soit (9, 9, 9)
molecules_left_cube = n_molecules_avg[:, :, :, 1] / n_molecules_avg[:, :, :, 0]
photobleaching_cube = (n_molecules_avg[:, :, :, 0] - n_molecules_avg[:, :, :, 1]) / n_molecules_avg[:, :, :, 0]

values_to_plot_dict = {
    "f1_score": f1_score_avg,
    "n_photons": n_photons_values_avg,
    "photobleaching": photobleaching_cube,
    "molecules_left": molecules_left_cube
}

# metric_to_obs = "f1_score"
# metric_to_obs = "n_photons"
# metric_to_obs = "photobleaching"
metric_to_obs = "molecules_left"

# dimensions are [pdt, p_ex, p_sted]
# 1st square : max proj along pdt ==> rows are p_ex, cols are p_sted
pdt_max_proj = np.max(values_to_plot_dict[metric_to_obs], axis=0)
# 2nd square : max proj along p_ex ==> rows are pdt, cols are p_sted
pex_max_proj = np.max(values_to_plot_dict[metric_to_obs], axis=1)
# 3rd square : max proj along p_sted ==> rows are pdtm cols are p_sted
psted_max_proj = np.max(values_to_plot_dict[metric_to_obs], axis=2)

if metric_to_obs == "f1_score":
    vmax = 1
elif metric_to_obs == "photobleaching":
    vmax = 1
elif metric_to_obs == "molecules_left":
    vmax = 1
elif metric_to_obs == "n_photons":
    vmax = np.max(values_to_plot_dict[metric_to_obs])
else:
    # uh oh
    vmax = 0

imshow_kwargs = {"vmin": 0,
                 "vmax": vmax,
                 "cmap": "inferno"}

fig, axes = plt.subplots(1, 3)

axes[0].imshow(pdt_max_proj, **imshow_kwargs)
axes[0].set_ylabel("p_ex")
axes[0].set_yticks(np.arange(0, 9))
# axes[0].set_yticklabels(p_ex_values)
axes[0].set_xlabel("p_sted")
axes[0].set_xticks(np.arange(0, 9))
# axes[0].set_xticklabels(p_sted_values)

axes[1].imshow(pex_max_proj, **imshow_kwargs)
axes[1].set_ylabel("pdt")
axes[1].set_yticks(np.arange(0, 9))
# axes[1].set_yticklabels(pdt_values)
axes[1].set_xlabel("p_sted")
axes[1].set_xticks(np.arange(0, 9))
# axes[1].set_xticklabels(p_sted_values)

axes[2].imshow(psted_max_proj, **imshow_kwargs)
axes[2].set_ylabel("pdt")
axes[2].set_yticks(np.arange(0, 9))
# axes[2].set_yticklabels(pdt_values)
axes[2].set_xlabel("p_ex")
axes[2].set_xticks(np.arange(0, 9))
# axes[2].set_xticklabels(p_ex_values)

plot_cube = np.array([pdt_max_proj, pex_max_proj, psted_max_proj])
fig.suptitle(f"metric = {metric_to_obs}, \n"
             f"min = {np.min(plot_cube)}, max = {np.max(plot_cube)}")
plt.tight_layout()
plt.savefig(data_dir + f"results/{metric_to_obs}.pdf")
plt.show()
