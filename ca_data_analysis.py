from matplotlib import pyplot as plt
import numpy as np
import tifffile
import tqdm
from pysted import utils

path_video = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"
path_events = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
theresa_data = tifffile.imread(path_video)
list_of_events = utils.event_reader(path_events)
event_oi = list_of_events[0]

start = 450
end = 490
len_sequence = theresa_data.shape[0] - start
col_start, col_end = 140, 160
row_start, row_end = 100, 125

# je veux que mon affichage soit normalisé pendant le vidéo, donc je trouve le max de tout et j'ajoute une bordure
# à mes frames or something :)
event_data = theresa_data[event_oi["start frame"]: event_oi["end frame"],
                          event_oi["start row"]: event_oi["end row"],
                          event_oi["start col"]: event_oi["end col"]]
max_val = np.max(event_data)

plt.ion()
fig, axes = plt.subplots(1, 1)
for i in tqdm.trange(start, end):
    theresa_data[i, row_start, col_start] = max_val
    axes.clear()
    axes.set_title(f"frame {i + 1}")
    # axes.imshow(theresa_data[i], cmap="inferno")   # whole
    flash_imshow = axes.imshow(theresa_data[i, row_start: row_end, col_start: col_end], cmap="inferno")   # crop
    plt.pause(0.01)
plt.show()

# plotter l'intensité moyenne par frame

mean_photons = np.mean(event_data, axis=(1, 2))
max_photons = np.max(event_data, axis=(1, 2))
mult_fact = 3
# plotter en fct du temps à place? c'est 100 ms par frame je crois?
# FRAME RATE DE 100ms, LIVE TROP CAVE POUR DÉTERMINER ÇA VEUT DIRE QUOI
# 1 CHOSE À FOIS

# un frame rate de 100ms ça a pas de sens right? frame rate devrait être en nb_frames / s
times = np.arange(0, event_oi["end frame"] - event_oi["start frame"]) * 100   # 100 ms par frame
frames = np.arange(event_oi["start frame"], event_oi["end frame"])

# plt.plot(frames, mean_photons, label="Avg photons")
# plt.plot(frames, max_photons, label="Max photons")
# plt.plot(frames, mean_photons * mult_fact, label=f"{mult_fact} * Avg photons")
# plt.legend()
# plt.xlabel("Frame")
# plt.ylabel("Photons")
# plt.show()

# plt.plot(times, mean_photons, label="Avg photons")
# plt.plot(times, max_photons, label="Max photons")
# plt.plot(times, mean_photons * mult_fact, label=f"{mult_fact} * Avg photons")
# plt.legend()
# plt.xlabel("Time [ms]")
# plt.ylabel("Photons")
# plt.show()
