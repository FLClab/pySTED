# Import packages
from matplotlib import pyplot, image
import numpy
import tifffile
import os, datetime
from tkinter.filedialog import askopenfilename

from pysted import base, utils
from datamap_implem_vars import *
import time

print("Starting tests...")

# molecule_disposition = utils.datamap_generator(100, 200, 5)
molecule_disposition = numpy.ones((20, 20)) * 5
# roi = {'rows': [22, 77], 'cols': [22, 77]}   # oops
# molecule_disposition = tifffile.imread('examples/data/fibres.tif')
# molecule_disposition = (molecule_disposition / numpy.amax(molecule_disposition) * 5).astype(int)

# pdt_array_linspace = numpy.linspace(0, pixeldwelltime * 5,
#                                     num=molecule_disposition.shape[0] * molecule_disposition.shape[1])
# pdt_array = numpy.reshape(pdt_array_linspace, molecule_disposition.shape)
pdt_array = numpy.ones(molecule_disposition.shape) * pixeldwelltime

# p_ex_array_linspace = numpy.linspace(0, p_ex * 5,
#                                      num=molecule_disposition.shape[0] * molecule_disposition.shape[1])
# p_ex_array = numpy.reshape(p_ex_array_linspace, molecule_disposition.shape)
p_ex_array = numpy.ones(molecule_disposition.shape) * p_ex

# p_sted_array_linspace = numpy.linspace(0, p_sted * 5,
#                                        num=molecule_disposition.shape[0] * molecule_disposition.shape[1])
# p_sted_array = numpy.reshape(p_sted_array_linspace, molecule_disposition.shape)
p_sted_array = numpy.ones(molecule_disposition.shape) * p_sted

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9)
detector = base.Detector(noise=True)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap = base.Datamap(molecule_disposition, datamap_pixelsize, roi=roi, pdt=pdt_array,
                       p_ex=p_ex_array, p_sted=p_sted_array)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, datamap, bleach_func="fifty_fifty")


og_whole_datamap = numpy.copy(datamap.whole_datamap)

pixel_list = utils.pixel_sampling(datamap.whole_datamap[datamap.roi], mode="all")

acq_start = time.time()
# numpy.random.seed(1)
intensity = microscope.get_signal_and_bleach_functions(pixelsize, p_ex, p_sted, bleach=bleach)
acq_end = time.time()
print(f"took {acq_end - acq_start} s to acquire the image")

fig, axes = pyplot.subplots(1, 3)

og_datamap_imshow = axes[0].imshow(og_whole_datamap)
axes[0].set_title(f"Datamap before acquisition, shape = {og_whole_datamap.shape}")
fig.colorbar(og_datamap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

acquired_signal_imshow = axes[1].imshow(intensity)
axes[1].set_title(f"Acquired signal, shape = {intensity.shape}")
fig.colorbar(acquired_signal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

bleached_imshow = axes[2].imshow(datamap.whole_datamap)
axes[2].plot(numpy.linspace(datamap.roi_corners['tl'][1], datamap.roi_corners['tr'][1]),
             numpy.linspace(datamap.roi_corners['tl'][0], datamap.roi_corners['tr'][0]), color='r')   # top line
axes[2].plot(numpy.linspace(datamap.roi_corners['bl'][1], datamap.roi_corners['br'][1]),
             numpy.linspace(datamap.roi_corners['bl'][0], datamap.roi_corners['br'][0]), color='r')   # bottom line
axes[2].plot(numpy.linspace(datamap.roi_corners['tl'][1], datamap.roi_corners['bl'][1]),
             numpy.linspace(datamap.roi_corners['tl'][0], datamap.roi_corners['bl'][0]), color='r')   # left line
axes[2].plot(numpy.linspace(datamap.roi_corners['tr'][1], datamap.roi_corners['br'][1]),
             numpy.linspace(datamap.roi_corners['tr'][0], datamap.roi_corners['br'][0]), color='r')   # right line
axes[2].set_title(f"Bleached datamap, shape = {datamap.whole_datamap.shape}")
fig.colorbar(bleached_imshow, ax=axes[2], fraction=0.04, pad=0.05)

pyplot.suptitle(f"Whole datamap shape = {datamap.whole_shape}, \n"
                f"ROI shape = {datamap.whole_datamap[datamap.roi].shape}, \n"
                f"Bleaching = {bleach}")
pyplot.show()
