
import argparse

from matplotlib import cm, pyplot
import numpy
import tifffile

from pysted import base, utils


parser = argparse.ArgumentParser(description="Simulate the acquisition of a single 10nm particule.")
parser.add_argument("pixelsize", type=float,
                    help="pixelsize (in m) of the acquired image")
parser.add_argument("--pdt", type=float, default=10e-6,
                    help="pixel dwell time (in s)")
parser.add_argument("--exc", type=float, default=1e-6,
                    help="excitation power (in W)")
parser.add_argument("--sted", type=float, default=30e-3,
                    help="STED power (in W)")
args = parser.parse_args()

datamap = tifffile.imread("data/singledot.tif")
# normalize datamap, 3 molecules/pixel at most
datamap = datamap / numpy.max(datamap) * 3

laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, distortion=0.04)
detector = base.Detector(pdef=0.02)
objective = base.Objective()
fluo = base.Fluorescence(535e-9)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)

datamap = utils.rescale(datamap, int(args.pixelsize / 10e-9))

pyplot.figure("Data map")
pyplot.imshow(datamap, interpolation="nearest")
#pyplot.colorbar()

signal = microscope.get_signal(datamap, args.pixelsize, args.pdt, args.exc, args.sted)

pyplot.figure("Signal")
pyplot.imshow(signal, interpolation="nearest")
#pyplot.colorbar()

pyplot.show()

