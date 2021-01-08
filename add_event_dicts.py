# this little script will only be used to add events to files, so I don't have to do this
# in the test script file, so I don't write the same event 1000 times in one file

# juste faire une fonction dans utils pour ça?
from pysted import utils
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import tifffile

# ---------------------------------- FIND EVENTS IN VIDEOS -------------------------------------------------------------
"""
txtfile_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
video_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"

video_stream = tifffile.imread(video_path)
print(video_stream.shape)

frame_start, frame_end = 0, video_stream.shape[0]
# frame_start, frame_end = 5, 45
col_start, col_end = 0, video_stream.shape[2]
# col_start, col_end = 310, 345
row_start, row_end = 0, video_stream.shape[1]
# row_start, row_end = 340, 370

min_count = np.min(video_stream[frame_start: frame_end, row_start: row_end, col_start: col_end])
max_count = np.max(video_stream[frame_start: frame_end, row_start: row_end, col_start: col_end])

plt.ion()
fig, axes = plt.subplots(1, 1)
for i in tqdm.trange(frame_start, frame_end):
    axes.clear()
    axes.set_title(f"frame {i + 1}")
    # axes.imshow(theresa_data[i], cmap="inferno")   # whole
    flash_imshow = axes.imshow(video_stream[i, row_start: row_end, col_start: col_end], cmap="inferno",
                               vmin=min_count, vmax=max_count)   # crop
    plt.pause(0.001)
plt.show()

event = {"start frame": frame_start,
         "end frame": frame_end,
         "start col": col_start,
         "end col": col_end,
         "start row": row_start,
         "end row": row_end}
light_curve = utils.get_light_curve(video_path, event)
print(len(light_curve))
frames = np.arange(frame_start, frame_end)
plt.plot(frames, light_curve)
plt.show()
"""
# ---------------------------------- ADD EVENT DICTS TO THE TXT FILE ---------------------------------------------------
"""
txtfile_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
video_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"

frames_list = [(450, 490), (40, 80), (45, 85), (35, 75), (5, 45),
          (40, 80), (135, 175), (115, 155), (100, 140), (110, 150)]
rows_list = [(100, 125), (250, 300), (84, 120), (130, 160), (185, 198),
        (330, 350), (405, 420), (140, 160), (290, 310), (340, 370)]
cols_list = [(140, 160), (485, 520), (0, 30), (30, 70), (465, 490),
        (170, 195), (495, 515), (170, 190), (560, 590), (310, 345)]

for frames, rows, cols in zip(frames_list, rows_list, cols_list):
    utils.add_event(txtfile_path, frames[0], frames[1], rows[0], rows[1], cols[0], cols[1])
"""
# ---------------------------------- LIGHT CURVE STUFF -----------------------------------------------------------------
"""
# plan est de lire les events du fichier et de les plotter
txtfile_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
video_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"
events = utils.event_reader(txtfile_path)

curves, rescaled_curves = [], []
x_axis = np.arange(0, 40)
counter = 1
for event in events:
    light_curve = utils.get_light_curve(video_path, event)
    curves.append(light_curve)
    counter += 1

for curve_idx, curve in enumerate(curves):
    rescaled_curve = utils.rescale_data(curve, to_int=False, divider=1)
    rescaled_curves.append(rescaled_curve)
    # plt.plot(x_axis, curve, label=f"Event {curve_idx + 1}")
    # plt.plot(x_axis, rescaled_curve, label=f"Event {curve_idx + 1} (normalized)")
    # plt.xlabel(f"Frame")
    # plt.ylabel(f"Average photon count")
    # plt.legend()
    # plt.savefig(f"D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_light_curves_normalized/event_{curve_idx + 1}_normalized")
    # plt.close()

# for curve_idx, curve in enumerate(rescaled_curves):
#     plt.plot(x_axis, curve, label=f"Event {curve_idx + 1} (normalized)")
#
# plt.xlabel(f"Frame")
# plt.ylabel(f"Average photon count")
# plt.legend()
# plt.show()
"""
#------------------------------ SHIFTING THE CURVES --------------------------------------------------------------------
"""
# # l'idée ici est que le max de chaque courbe soit au même endroit :)
# rescaled_curves.pop(6)
# rescaled_curves.pop(4)

unshifted_peaks, shifted_curves = [], []
for curve_idx, curve in enumerate(rescaled_curves):
    print(curve_idx)
    if curve_idx == 4 or curve_idx == 6:
        unshifted_peaks.append(np.nan)
        continue
    peak_arg = np.argmax(curve)
    unshifted_peaks.append(peak_arg)
    shifted_curve = rescaled_curves[curve_idx][unshifted_peaks[curve_idx] - 5:]
    while len(shifted_curve) != 40:
        shifted_curve = np.append(shifted_curve, shifted_curve[-1])
    shifted_curves.append(shifted_curve)

# for curve_idx, curve in enumerate(shifted_curves):
#     plt.plot(x_axis, curve, label=f"Event {curve_idx} (normalized, shifted)")
#
# plt.xlabel(f"Frame")
# plt.ylabel(f"Average photon count")
# plt.legend()
# plt.show()

avg_shifted_curves = np.mean(shifted_curves, axis=0)
std_shifted_curves = np.std(shifted_curves, axis=0)
plt.plot(x_axis, avg_shifted_curves)
plt.fill_between(x_axis, avg_shifted_curves + std_shifted_curves, avg_shifted_curves - std_shifted_curves, alpha=0.4)
plt.xlabel(f"Frame")
plt.ylabel(f"Average photon count")
plt.show()
"""
#--------------------------------- TESTING LIGHT CURVE FUNC ------------------------------------------------------------

# get the light curves
txtfile_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
video_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"
events = utils.event_reader(txtfile_path)

curves = []
x_axis = np.arange(0, 40)
counter = 1
for event in events:
    light_curve = utils.get_light_curve(video_path, event)
    curves.append(light_curve)
    counter += 1

# remove the weirder events
curves.pop(6)
curves.pop(4)
avg_curve, std_curve = utils.get_avg_lightcurve(curves)
plt.plot(x_axis, avg_curve)
plt.fill_between(x_axis, avg_curve + std_curve, avg_curve - std_curve, alpha=0.4)
plt.xlabel(f"Frame")
plt.ylabel(f"Average photon count")
plt.show()
