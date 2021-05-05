
import numpy
import time
import random

from pysted import base, utils, raster, bleach_funcs
from dymin import DyMINMicroscope
from matplotlib import pyplot

numpy.random.seed(42)
random.seed(42)

START = time.time()

delta = 1
molecules_disposition = numpy.zeros((50, 50))
# molecules_disposition[
#     molecules_disposition.shape[0]//2 - delta : molecules_disposition.shape[0]//2+delta,
#     molecules_disposition.shape[1]//2 - delta : molecules_disposition.shape[1]//2+delta] = 8
num_mol = 2
for j in range(1,4):
    for i in range(1,4):
#         molecules_disposition[
#             i * molecules_disposition.shape[0]//4,
#             j * molecules_disposition.shape[1]//4] = num_mol
        molecules_disposition[
            j * molecules_disposition.shape[0]//4 - delta : j * molecules_disposition.shape[0]//4 + delta + 1,
            i * molecules_disposition.shape[1]//4 - delta : i * molecules_disposition.shape[1]//4 + delta + 1] = num_mol

print("Setting up the microscope ...")
# Microscope stuff
# egfp = {"lambda_": 535e-9,
#         "qy": 0.6,
#         "sigma_abs": {488: 1.15e-20,
#                       575: 6e-21},
#         "sigma_ste": {560: 1.2e-20,
#                       575: 6.0e-21,
#                       580: 5.0e-21},
#         "sigma_tri": 1e-21,
#         "tau": 3e-09,
#         "tau_vib": 1.0e-12,
#         "tau_tri": 5e-6,
#         "phy_react": {488: 1e-7,   # 1e-4
#                       575: 1e-11},   # 1e-8
#         "k_isc": 0.26e6}
egfp = {"lambda_": 535e-9,
        "qy": 0.6,
        "sigma_abs": {488: 3e-20,
                      575: 6e-21},
        "sigma_ste": {560: 1.2e-20,
                      575: 6.0e-21,
                      580: 5.0e-21},
        "sigma_tri": 1e-21,
        "tau": 3e-09,
        "tau_vib": 1.0e-12,
        "tau_tri": 5e-6,
        "phy_react": {488: 0.25e-7,   # 1e-4
                      575: 0.25e-11},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 10e-9
bleach = True
p_ex = 2e-6
p_ex_array = numpy.ones(molecules_disposition.shape) * p_ex
p_sted = 2.5e-3
p_sted = 0.
p_sted_array = numpy.ones(molecules_disposition.shape) * p_sted
pdt = 100e-6
pdt_array = numpy.ones(molecules_disposition.shape) * pdt
roi = 'max'

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=False, background=0, pcef=0.1)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap = base.Datamap(molecules_disposition, pixelsize)

opts = {
    "scale_power" : [0., 0.25, 1.],
    "decision_time" : [10e-6, 10e-6, -1],
    "threshold_count" : [10, 5, 0]
}
microscope = DyMINMicroscope(laser_ex, laser_sted, detector, objective, fluo, opts=opts)
# microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)
start = time.time()
i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)
datamap.set_roi(i_ex, roi)
print("Setup done...")

time_start = time.time()
acquisition, bleached, scaled_power = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
                                                                    bleach=bleach, update=False, seed=42)
# laser_received, sampled = microscope.laser_dans_face(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
#                                                                     bleach=bleach, update=False)
# laser_received = laser_received[datamap.roi]
# sampled = sampled[datamap.roi]
# fig, axes = pyplot.subplots(1,2)
# axes[0].imshow(laser_received)
# axes[0].set_title("Laser received")
# axes[1].imshow(sampled)
# axes[1].set_title("Sampled")
# print(sampled[molecules_disposition > 0])
# print(laser_received[molecules_disposition > 0])

print(f"ran in {time.time() - time_start} s")


fig, axes = pyplot.subplots(1, 4, figsize=(10,3), sharey=True, sharex=True)

axes[0].imshow(datamap.whole_datamap[datamap.roi])
axes[0].set_title(f"Datamap roi")

axes[1].imshow(bleached["base"][datamap.roi], vmin=0, vmax=num_mol)
axes[1].set_title(f"Bleached datamap")

axes[2].imshow(acquisition)
axes[2].set_title(f"Acquired signal (photons)")

axes[3].imshow(scaled_power)
axes[3].set_title(f"Scaled power")

print("Average molecules left : ", bleached["base"][datamap.roi][molecules_disposition != 0].mean(axis=-1))

print("Total run time : {}".format(time.time() - START))

pyplot.show()
