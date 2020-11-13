# Import packages
from matplotlib import pyplot, image
import numpy
import tifffile

from pysted import base, utils
from datamap_implem_vars import *
import time

print("Starting tests...")

molecule_disposition = tifffile.imread('examples/data/fibres.tif')
molecule_disposition = (molecule_disposition / numpy.amax(molecule_disposition) * 5).astype(int)

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9)
detector = base.Detector(noise=True)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap = base.Datamap(molecule_disposition, datamap_pixelsize)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, bleach_func="default_bleach")
i_ex, _, _ = microscope.cache(datamap.pixelsize)
datamap.set_roi(i_ex, 'max')

pdt_array = numpy.reshape(numpy.linspace(0, 5 * pdt,
                                         datamap.whole_datamap[datamap.roi].shape[0] *
                                         datamap.whole_datamap[datamap.roi].shape[1]),
                          datamap.whole_datamap[datamap.roi].shape)

p_ex_array = numpy.reshape(numpy.linspace(0, 5 * p_ex,
                                          datamap.whole_datamap[datamap.roi].shape[0] *
                                          datamap.whole_datamap[datamap.roi].shape[1]),
                           datamap.whole_datamap[datamap.roi].shape)

p_sted_array = numpy.reshape(numpy.linspace(0, 5 * p_sted,
                                            datamap.whole_datamap[datamap.roi].shape[0] *
                                            datamap.whole_datamap[datamap.roi].shape[1]),
                             datamap.whole_datamap[datamap.roi].shape)

pixel_list = utils.pixel_sampling(datamap.whole_datamap[datamap.roi], mode="all")

acq_start = time.time()
# numpy.random.seed(1)
intensity, bleached = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt_array, p_ex_array, p_sted_array,
                                                       pixel_list=pixel_list, bleach=bleach, update=False)
acq_end = time.time()
print(f"took {acq_end - acq_start} s to acquire the image")

fig, axes = pyplot.subplots(1, 3)

og_datamap_imshow = axes[0].imshow(datamap.whole_datamap)
axes[0].set_title(f"Datamap before acquisition, shape = {datamap.whole_datamap.shape}")
fig.colorbar(og_datamap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

acquired_signal_imshow = axes[1].imshow(intensity)
axes[1].set_title(f"Acquired signal, shape = {intensity.shape}")
fig.colorbar(acquired_signal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

bleached_imshow = axes[2].imshow(bleached)
axes[2].plot(numpy.linspace(datamap.roi_corners['tl'][1], datamap.roi_corners['tr'][1]),
             numpy.linspace(datamap.roi_corners['tl'][0], datamap.roi_corners['tr'][0]), color='r')   # top line
axes[2].plot(numpy.linspace(datamap.roi_corners['bl'][1], datamap.roi_corners['br'][1]),
             numpy.linspace(datamap.roi_corners['bl'][0], datamap.roi_corners['br'][0]), color='r')   # bottom line
axes[2].plot(numpy.linspace(datamap.roi_corners['tl'][1], datamap.roi_corners['bl'][1]),
             numpy.linspace(datamap.roi_corners['tl'][0], datamap.roi_corners['bl'][0]), color='r')   # left line
axes[2].plot(numpy.linspace(datamap.roi_corners['tr'][1], datamap.roi_corners['br'][1]),
             numpy.linspace(datamap.roi_corners['tr'][0], datamap.roi_corners['br'][0]), color='r')   # right line
axes[2].set_title(f"Bleached datamap, shape = {bleached.shape}")
fig.colorbar(bleached_imshow, ax=axes[2], fraction=0.04, pad=0.05)

pyplot.suptitle(f"Bleaching = {bleach}, \n"
                f"datamap.pixelsize = {datamap.pixelsize}, laser.shape = {i_ex.shape}, \n"
                f"took {round(acq_end - acq_start, 5)} s to run")
pyplot.show()
