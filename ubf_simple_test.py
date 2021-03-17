"""
c koi lplan
le plan ici c'est de tester mes différentes c func pour comparer leur vitesse, et voir s'il y en aurait pt certaines
que je peux deleter, si leur gain de vitesse est minime

- faire une datamap de taille X par X, valeur constante de 10 partout
- Acq avec les 5 fonctions, compare la vitesse
- faire une datamap de taille 2X par 2X, valeur constante de 10 partout
- Acq avec les 5 fonctions, compare la vitesse

Dans le cas ici, pas de sub_datamaps, donc je m'attends à avoir une vitesse comporable entre
split et celle qui prend des arrays

"""

import numpy as np
from pysted import base, utils, raster
import time
from matplotlib import pyplot as plt


frame_shape = (64, 64)

molecules_disposition = np.ones(frame_shape) * 10

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
pdt = 10e-6
p_ex = 1e-6
p_ex_array = np.ones(frame_shape) * p_ex
p_sted = 30e-3
p_sted_array = np.ones(frame_shape) * p_sted
roi = 'max'

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap = base.Datamap(molecules_disposition, pixelsize)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, bleach_func="default_bleach")
i_ex, _, _ = microscope.cache(datamap.pixelsize)
datamap.set_roi(i_ex, roi)

# quand j'utilise raster.raster_func_c_self_bleach_split, utiliser ça
# acq, bleached_dict, intensity_map = microscope.get_signal_and_bleach_fast_2(datamap, datamap.pixelsize, pdt, p_ex_array,
#                                                                             p_sted_array, bleach=True, update=False,
#                                                                             filter_bypass=False,
#                                                                             raster_func=raster.raster_func_c_self_bleach_split)
#
# fig, axes = plt.subplots(1, 3)
#
# axes[0].imshow(datamap.whole_datamap[datamap.roi])
# axes[1].imshow(bleached_dict["base"][datamap.roi])
# axes[2].imshow(acq)
#
# plt.show()

# quand j'utilise les autres c_func, utiliser ça
# acq, bleached, intensity_map = microscope.get_signal_and_bleach_fast_3(datamap, datamap.pixelsize, pdt, p_ex,
#                                                                             p_sted, bleach=True, update=False,
#                                                                             filter_bypass=False,
#                                                                             raster_func=raster.raster_func_c)
#
# fig, axes = plt.subplots(1, 3)
#
# axes[0].imshow(datamap.whole_datamap[datamap.roi])
# axes[1].imshow(bleached[datamap.roi])
# axes[2].imshow(acq)
#
# plt.show()

"""
Voici l'idée : 
définir des tailles de datamap
itérer sur les tailles de datamap, créer l'objet et set sa ROI et tout et tout
itérer sur les 5 fonctions de bleach, calculer le run time pour chacune, ajouter à la liste
plotter les run times en fonction du size d'un coté
"""

frame_shapes_list = [(64, 64), (128, 128), (256, 256)]
c_funcs_list = [raster.raster_func_wbleach_c, raster.raster_func_c, raster.raster_func_c_self_bleach,
                raster.raster_func_c_self_bleach_split, raster.raster_func_c_self]
wbleach_c_times, c_times, self_bleach_times, self_times, self_split_times = [], [], [], [], []

for frame_shape in frame_shapes_list:
    print(f"executing loop for frame_shape = {frame_shape}")
    molecules_disposition = np.ones(frame_shape) * 10
    datamap = base.Datamap(molecules_disposition, pixelsize)
    datamap.set_roi(i_ex, roi)
    p_ex = 1e-6
    p_ex_array = np.ones(frame_shape) * p_ex
    p_sted = 30e-3
    p_sted_array = np.ones(frame_shape) * p_sted
    for c_func in c_funcs_list:
        print(c_func)
        time_start = time.time()
        if c_func == raster.raster_func_c_self_bleach_split:
            acq, bleached_dict, intensity_map = microscope.get_signal_and_bleach_fast_2(datamap, datamap.pixelsize, pdt,
                                                                                        p_ex_array, p_sted_array,
                                                                                        bleach=True, update=False,
                                                                                        filter_bypass=False,
                                                                                        raster_func=c_func)
        elif c_func == raster.raster_func_c_self or c_func == raster.raster_func_c_self_bleach:
            acq, bleached, intensity_map = microscope.get_signal_and_bleach_fast_3(datamap, datamap.pixelsize, pdt,
                                                                                   p_ex_array, p_sted_array,
                                                                                   bleach=True, update=False,
                                                                                   filter_bypass=False,
                                                                                   raster_func=c_func)
        elif c_func == raster.raster_func_c or c_func == raster.raster_func_wbleach_c:
            acq, bleached, intensity_map = microscope.get_signal_and_bleach_fast_3(datamap, datamap.pixelsize, pdt,
                                                                                   p_ex, p_sted,
                                                                                   bleach=True, update=False,
                                                                                   filter_bypass=False,
                                                                                   raster_func=c_func)
        else:
            print("uh oh")
        time_run = time.time() - time_start
        if c_func == raster.raster_func_c_self_bleach_split:
            self_split_times.append(time_run)
        elif c_func == raster.raster_func_wbleach_c:
            wbleach_c_times.append(time_run)
        elif c_func == raster.raster_func_c:
            c_times.append(time_run)
        elif c_func == raster.raster_func_c_self_bleach:
            self_bleach_times.append(time_run)
        elif c_func == raster.raster_func_c_self:
            self_times.append(time_run)
        else:
            print("uh oh")

x_axis = [64, 128, 256]
plt.plot(x_axis, self_split_times, label="bleach split c func, var powers")
plt.plot(x_axis, self_bleach_times, label="bleach c func, var powers")
plt.plot(x_axis, self_times, label="c func, var powers")
plt.plot(x_axis, wbleach_c_times, label="bleach c func, static powers")
plt.plot(x_axis, c_times, label="c func, static powers")
plt.legend()
plt.xlabel("Frame size")
plt.ylabel("Time (s)")
plt.show()

