from matplotlib import pyplot as plt
import numpy as np
import tifffile
import tqdm
from pysted import base, utils
import time

# ------------------------------------------- GET LIGHT CURVE TESTS ----------------------------------------------------
"""
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
"""
# -------------------------------------------- ACQUIRE AND SIMULATE A FLASH --------------------------------------------
"""
# le plan c'est de faire une acq par frame d'un evt (40 pour l'evt 1) sur la datamap de poils classique, et faire
# augmenter le nombre de molécules dans une région selon la courbe de flash que j'ai extrait

# get l'image de poils and crop a 100 by 100 region
poils = tifffile.imread("examples/data/fibres.tif")
poils = poils[325: 425, 370: 470]
poils = (poils / np.max(poils) * 3).astype(int)

# génère mon microscope et toutes les choses qui vont avec :)
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
        "phy_react": {488: 1e-4,   # 1e-4
                      575: 1e-8},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 10e-9
dpxsz = 10e-9
bleach = False
p_ex = 1e-6
p_sted = 30e-3
pdt = 10e-6
size = 64 + (2 * 22 + 1)
roi = 'max'
seed = True

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
# zero_residual controls how much of the donut beam "bleeds" into the the donut hole
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
# noise allows noise on the detector, background adds an average photon count for the empty pixels
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap = base.Datamap(poils, dpxsz)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, bleach_func="default_bleach")
i_ex, _, _ = microscope.cache(datamap.pixelsize)
datamap.set_roi(i_ex, roi)

# obtenir la light curve pour pouvoir l'utiliser pendant l'acquisition
path_video = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"
path_events = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
list_of_events = utils.event_reader(path_events)
event_oi = list_of_events[0]
light_curve = utils.get_light_curve(path_video, event_oi)
# normaliser la light curve entre 1 et X? comme ça la région commencerait à la bonne val et monterait aux bonnes vals
normalized_light_curve = utils.rescale_data(light_curve, to_int=True, divider=3)


# faire 40 acquisitions back to back, afficher les 40 après
# une fois que ce sera fait, il restera à sélectionner une petite région que je flasherai :)
# flash_roi = {"row start": 375,   # good for uncropped img, not for cropped
#              "row end": 382,
#              "col start": 420,
#              "col end": 432}
flash_roi = {"row start": 52,   # good for cropped
             "row end": 56,
             "col start": 54,
             "col end": 60}
nb_frames = event_oi["end frame"] - event_oi["start frame"]
save_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Analysis/Ca2+/fibres_flash/Confocal_norm_v2/"
frozen_datamap = np.copy(datamap.whole_datamap[datamap.roi])
acquisitions = []
# jdevrais le refaire et les normaliser
for i in range(nb_frames):
    # multiply the region by the normalized light curve
    print(f"acq on frame {i + 1}")
    datamap.whole_datamap[datamap.roi][flash_roi["row start"]: flash_roi["row end"],
                                       flash_roi["col start"]: flash_roi["col end"]] *= normalized_light_curve[i]
    time_start = time.time()
    acq, bleached = microscope.get_signal_and_bleach_fast(datamap, datamap.pixelsize, pdt, p_ex, 0.0,
                                                          pixel_list=None, bleach=bleach, update=False)
    acq_time = time.time() - time_start
    print(f"acq took {acq_time}")
    # plt.imsave(save_path+str(i)+".png", acq)
    acquisitions.append(acq)

    # reset the datamap so as not to "stack" the multiplications
    datamap.whole_datamap[datamap.roi] = frozen_datamap
    # plt.imshow(datamap.whole_datamap[datamap.roi])
    # plt.show()

all_max, all_min = np.max(acquisitions), np.min(acquisitions)
for i, acq_i in enumerate(acquisitions):
    # acq_i[0, 0] = all_max
    plt.imsave(save_path+str(i)+".png", acq_i, vmin=all_min, vmax=all_max)
"""
#------------------------------------------- GENERATE RANDOM FLASHES ---------------------------------------------------

# step 1 is to see if simply sampling each of the 40 pts from a normal with mean and std given by my func returns good
# looking light curves. My fear is that it might look too jaggedy? maybe I can smooth it out after?

# get the light curves
txtfile_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
video_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"
events = utils.event_reader(txtfile_path)

curves = []
counter = 1
for event in events:
    light_curve = utils.get_light_curve(video_path, event)
    curves.append(light_curve)
    counter += 1

# remove the weirder events
curves.pop(6)
curves.pop(4)
avg_curve, std_curve = utils.get_avg_lightcurve(curves)

# Comparing Avg ± std to sampled to sampled+smoothed
# sampled_curve = np.random.normal(avg_curve, std_curve)
# smoothed_sampled = utils.savitzky_golay(sampled_curve, 5, 2)
#
# x_axis = np.arange(0, 40)
# plt.plot(x_axis, avg_curve, label="Average light curve")
# plt.plot(x_axis, sampled_curve, label="Sampled light curve")
# plt.plot(x_axis, smoothed_sampled, label="Smoothed sampled curved")
# plt.fill_between(x_axis, avg_curve + std_curve, avg_curve - std_curve, alpha=0.4)
# plt.xlabel(f"Frame")
# plt.ylabel(f"Average photon count")
# plt.legend()
# plt.show()

# Comparing multiple sampled+smoothed curve on 1 graph
# n_tries = 5
# x_axis = np.arange(0, 40)
# for i in range(n_tries):
#     sampled_curve = np.random.normal(avg_curve, std_curve)
#     sampled_curve = np.where(sampled_curve >= 1, sampled_curve, 1)
#     smoothed_sampled = utils.savitzky_golay(sampled_curve, 5, 2)
#     plt.plot(x_axis, smoothed_sampled)
#
# plt.xlabel(f"Frame")
# plt.ylabel(f"Average photon count")
# plt.legend()
# plt.show()

# Trying to add flashes together with delays
# I will test this using the average curve and my utils.shift_data.
# I will first need to verify if my shift function works when I wish to shift to the right

# shifted_curve = utils.shift_data(avg_curve, peak_idx=15)
# multi_pulse = 0.7 * avg_curve + 0.3 * shifted_curve
#
# x_axis = np.arange(0, 40)
# plt.plot(x_axis, multi_pulse, label="Avg curve")
# plt.xlabel(f"Frame")
# plt.ylabel(f"Average photon count")
# plt.legend()
# plt.show()

sampled_curve = np.random.normal(avg_curve, std_curve)
sampled_curve = np.where(sampled_curve >= 1, sampled_curve, 1)
smoothed_sampled = utils.savitzky_golay(sampled_curve, 5, 2)
shifted_smoothed_sampeld = utils.shift_data(smoothed_sampled, peak_idx=15)

multi_pulse = 0.7 * smoothed_sampled + 0.3 * shifted_smoothed_sampeld

x_axis = np.arange(0, 40)
plt.plot(x_axis, multi_pulse, label="Avg curve")
plt.xlabel(f"Frame")
plt.ylabel(f"Average photon count")
plt.legend()
plt.show()

#------------------------------------------- RANDOM TESTS --------------------------------------------------------------
"""
path_video = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"
path_events = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
list_of_events = utils.event_reader(path_events)
event_oi = list_of_events[0]

frames = np.arange(event_oi["start frame"], event_oi["end frame"])
light_curve = utils.get_light_curve(path_video, event_oi)
normalized_light_curve = utils.rescale_data(light_curve, divider=5)
print(normalized_light_curve.astype(int))
plt.plot(frames, light_curve, label="Light curve")
plt.plot(frames, normalized_light_curve.astype(int), label="Normalize + inted")
plt.legend()
plt.show()
"""