
import numpy
import time
import random

from pysted import base, utils, raster, bleach_funcs
from matplotlib import pyplot

numpy.random.seed(42)
random.seed(42)

# molecules_disposition = (numpy.random.rand(40, 40) > 0.9).astype(int) * 10
delta = 2
molecules_disposition = numpy.zeros((50, 50))
molecules_disposition[
    molecules_disposition.shape[0]//2 - delta : molecules_disposition.shape[0]//2+delta,
    molecules_disposition.shape[1]//2 - delta : molecules_disposition.shape[1]//2+delta] = 10

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
        "phy_react": {488: 1e-7,   # 1e-4
                      575: 1e-11},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 10e-9
bleach = False
p_ex = 1e-6
p_ex_array = numpy.ones(molecules_disposition.shape) * p_ex
p_sted = 30e-3
p_sted_array = numpy.ones(molecules_disposition.shape) * p_sted
pdt = 100e-6
pdt_array = numpy.ones(molecules_disposition.shape) * pdt
roi = 'max'

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap = base.Datamap(molecules_disposition, pixelsize)
microscope = DyMINMicroscope(laser_ex, laser_sted, detector, objective, fluo)
start = time.time()
i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)
datamap.set_roi(i_ex, roi)

print(f'starting acq with phy_react = {egfp["phy_react"]}')
time_start = time.time()
acquisition, bleached, scaled_power = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
                                                                    bleach=bleach, update=False)
print(f"ran in {time.time() - time_start} s")


fig, axes = pyplot.subplots(1, 4, figsize=(10,3), sharey=True)

axes[0].imshow(datamap.whole_datamap[datamap.roi])
axes[0].set_title(f"Datamap roi")

axes[1].imshow(bleached["base"][datamap.roi])
axes[1].set_title(f"Bleached datamap")

axes[2].imshow(acquisition)
axes[2].set_title(f"Acquired signal (photons)")

axes[3].imshow(scaled_power)
axes[3].set_title(f"Scaled power")

pyplot.show()
