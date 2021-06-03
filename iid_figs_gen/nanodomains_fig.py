import numpy as np
from matplotlib import pyplot as plt


arrays_path = r"D:\SCHOOL\Maitrise\E2021\research\iid_pres\figs\simulated_data\nanodomains\3_cols_left_brighter"

bleached_dmaps = np.load(arrays_path + f"/bleached.npy")
acquisitions = np.load(arrays_path + f"/acquisitions.npy")
max_photons = np.max(acquisitions)

sted_powers = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.03, 0.08, 0.1, 0.3, 0.8]

for i in range(bleached_dmaps.shape[0]):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

    axes[0].imshow(bleached_dmaps[i], cmap="inferno")
    axes[0].set_title(f"Datamap after bleaching")
    axes[0].set_axis_off()

    axes[1].imshow(acquisitions[i], cmap="inferno")
    axes[1].plot([0, 63], [42, 42], color="red")
    axes[1].set_title(f"Acquired photons")
    axes[1].set_axis_off()

    axes[2].plot(acquisitions[i][42] / np.max(acquisitions[i][42]))
    axes[2].set_title(f"Line profile")
    axes[2].set_ylabel(f"Normalized photon count [-]")
    axes[2].set_xlabel(f"Position [pixel]")
    axes[2].set_ylim(bottom=0, top=1.2)

    fig.suptitle(f"STED power = {sted_powers[i]} W")
    plt.savefig(arrays_path + f"/{i}_p_sted_{sted_powers[i]}.jpg", )
    plt.close()
