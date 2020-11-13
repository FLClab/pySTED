# Import packages
from matplotlib import pyplot
import numpy
import time

from pysted import base, utils
import raster

print(f"Starting test to see if I can pass self to raster c func :)")

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
        "phy_react": {488: 1e-6,
                      575: 1e-10},
        "k_isc": 0.26e6}

datamap_pixelsize = 20e-9
p_ex = 1e-6
p_sted = 30e-3
pdt = 10e-6
bleach = True
seed = False
random_state = None
# ROI max is fine for bleach testing, since I'm interested in the bleaching inside the ROI :)
roi = 'max'
datamap_size = 200
colorbar_params = {'fraction': 0.04, 'pad': 0.05}

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9)
detector = base.Detector(noise=True)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, bleach_func="default_bleach")
i_ex, _, _ = microscope.cache(datamap_pixelsize)

if seed:
    random_state = 420
molecule_disposition = utils.datamap_generator(datamap_size, int((datamap_size ** 2 / 2)), 5,
                                               random_state=random_state)

datamap = base.Datamap(molecule_disposition, datamap_pixelsize)
datamap.set_roi(i_ex, roi)

p_ex_array = numpy.linspace(0, 5*p_ex, num=datamap.whole_datamap[datamap.roi].shape[0] *
                            datamap.whole_datamap[datamap.roi].shape[1])
p_ex_array = numpy.reshape(p_ex_array, datamap.whole_datamap[datamap.roi].shape)

p_sted_array = numpy.linspace(0, 5 * p_sted, num=datamap.whole_datamap[datamap.roi].shape[0] *
                              datamap.whole_datamap[datamap.roi].shape[1])
p_sted_array = numpy.reshape(p_sted_array, datamap.whole_datamap[datamap.roi].shape)

pdt_array = numpy.linspace(0, 5 * pdt, num=datamap.whole_datamap[datamap.roi].shape[0] *
                           datamap.whole_datamap[datamap.roi].shape[1])
pdt_array = numpy.reshape(pdt_array, datamap.whole_datamap[datamap.roi].shape)

print(f"Starting SELF acq...")
self_start = time.time()
acq_self, bleached_self = microscope.get_signal_and_bleach_test(datamap, datamap.pixelsize, pdt_array, p_ex_array,
                                                                p_sted_array, pixel_list=None, bleach=True,
                                                                raster_func=raster.raster_func_c_self, update=False,
                                                                seed=420)
self_acq_time = time.time() - self_start

print(f"Starting FAST acq...")
fast_start = time.time()
acq_fast, bleached_fast = microscope.get_signal_and_bleach_fast(datamap, datamap.pixelsize, pdt_array, p_ex_array,
                                                                p_sted_array, pixel_list=None, bleach=True,
                                                                raster_func=raster.raster_func_wbleach_c, update=False,
                                                                seed=420)
fast_acq_time = time.time() - fast_start

print(f"Starting OG acq...")
og_start = time.time()
acq_og, bleached_og = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt_array, p_ex_array, p_sted_array,
                                                       pixel_list=None, bleach=True, update=False)
og_acq_time = time.time() - og_start

acq_mse = utils.mse_calculator(acq_fast, acq_self)
bleach_mse = utils.mse_calculator(bleached_fast, bleached_self)

fig, axes = pyplot.subplots(3, 3)

self_datamap_imshow = axes[0, 0].imshow(datamap.whole_datamap)
axes[0, 0].set_title(f"Datamap, \n"
                     f"SELF acq took {self_acq_time} s")
fig.colorbar(self_datamap_imshow, ax=axes[0, 0], **colorbar_params)

self_acq_imshow = axes[0, 1].imshow(acq_self)
axes[0, 1].set_title(f"Acquisition (self)")
fig.colorbar(self_acq_imshow, ax=axes[0, 1], **colorbar_params)

self_bleached_imshow = axes[0, 2].imshow(bleached_self)
axes[0, 2].set_title(f"Bleached datamap (self)")
fig.colorbar(self_bleached_imshow, ax=axes[0, 2], **colorbar_params)

fast_datamap_imshow = axes[1, 0].imshow(datamap.whole_datamap)
axes[1, 0].set_title(f"Datamap, \n"
                     f"FAST acq took {fast_acq_time} s")
fig.colorbar(fast_datamap_imshow, ax=axes[1, 0], **colorbar_params)

fast_acq_imshow = axes[1, 1].imshow(acq_fast)
axes[1, 1].set_title(f"Acquisition (fast), \n"
                     f"MSE = {acq_mse}")
fig.colorbar(fast_acq_imshow, ax=axes[1, 1], **colorbar_params)

fast_bleached_imshow = axes[1, 2].imshow(bleached_fast)
axes[1, 2].set_title(f"Bleached datamap (fast), \n"
                     f"MSE = {bleach_mse}")
fig.colorbar(fast_bleached_imshow, ax=axes[1, 2], **colorbar_params)

og_datamap_imshow = axes[2, 0].imshow(datamap.whole_datamap)
axes[2, 0].set_title(f"Datamap, \n"
                     f"OG acq took {og_acq_time} s")
fig.colorbar(og_datamap_imshow, ax=axes[2, 0], **colorbar_params)

og_acq_imshow = axes[2, 1].imshow(acq_og)
axes[2, 1].set_title(f"Acquisition (OG)")
fig.colorbar(og_acq_imshow, ax=axes[2, 1], **colorbar_params)

og_bleached_imshow = axes[2, 2].imshow(bleached_og)
axes[2, 2].set_title(f"Bleached datamap (OG)")
fig.colorbar(og_bleached_imshow, ax=axes[2, 2], **colorbar_params)

pyplot.suptitle(f"SELF acquisition took {self_acq_time} s, \n"
                f"FAST acquisition took {fast_acq_time} s, \n"
                f"OG acquisition took {og_acq_time} s")
pyplot.show()
