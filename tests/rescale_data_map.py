'''Generate images of fiber structures. A cache folder is required.
'''

import IPython

import pickle
import re
import sys
import yaml

from matplotlib import cm, pyplot
import numpy

import cv2

from pysted import io, simulator, utils

# handle scientific numbers
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

laser_ex = simulator.GaussianBeam(488e-9)
laser_sted = simulator.DonutBeam(575e-9, distortion=0.04)
fluo = simulator.Fluorescence(535e-9)
detector = simulator.Detector(pdef=0.02)

microscope = simulator.Microscope(laser_ex, laser_sted, detector, fluo)

data_map = io.read_data_map(sys.argv[1], 1)
data_resolution = eval(sys.argv[2])
pixelsize = eval(sys.argv[3]) # nm
new_data_map = utils.rescale(data_map, int(pixelsize / data_resolution))

#results_path = "results/" + sys.argv[4]

pyplot.figure()
pyplot.title("Data map")
pyplot.imshow(data_map, interpolation="nearest")
pyplot.colorbar()
#pyplot.savefig(results_path + "data_map.png", borderpad=.1)

pyplot.figure()
pyplot.title("Data map rescaled")
pyplot.imshow(new_data_map, interpolation="nearest")
pyplot.colorbar()
#pyplot.savefig(results_path + "data_map_rescaled.png", borderpad=.1)

# imaging parameters
pdt = 10e-6
p_ex = 1e-6
p_sted = 0
#p_sted = 30e-3

signal, _ = microscope.get_signal(data_map, pixelsize*1e-9, pdt, p_ex, p_sted, cache=False)

pyplot.figure()
pyplot.title("Signal (on data map rescaled)")
pyplot.imshow(signal, interpolation="nearest")
pyplot.colorbar()
#pyplot.savefig(results_path + "signal.png", borderpad=.1)

pyplot.show()

