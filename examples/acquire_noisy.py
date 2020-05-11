
import argparse

from matplotlib import cm, pyplot
import numpy
import tifffile

from pysted import base, utils


parser = argparse.ArgumentParser(description="""Simulate the acquisition of an
                                                image serie with bleaching.""")
parser.add_argument("pixelsize", type=float,
                    help="pixelsize (in m) of the acquired images")
parser.add_argument("--pdt", type=float, default=10e-6,
                    help="pixel dwell time (in s)")
parser.add_argument("--exc", type=float, default=1e-6,
                    help="excitation power (in W)")
parser.add_argument("--sted", type=float, default=30e-3,
                    help="STED power (in W)")
args = parser.parse_args()

datamap = tifffile.imread("data/fibres.tif")
# normalize datamap, 3 molecules/pixel at most
datamap = datamap / numpy.max(datamap) * 3
datamap = datamap[:150, :150]

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
        "phy_react": {488: 1e-4,
                      575: 1e-8},
        "k_isc": 0.26e6}

laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, distortion=0.04)
detector = base.Detector(pdef=0.02)
detector_noisy = base.Detector(pdef=0.02, noise=True, background=100)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)

microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)
microscope_noisy = base.Microscope(laser_ex, laser_sted, detector_noisy, objective, fluo)

# the pixel size of the datamap is 5nm
datamap = utils.rescale(datamap, int(args.pixelsize / 5e-9))

pyplot.figure("Data map")
pyplot.imshow(datamap, interpolation="nearest")
pyplot.colorbar()

signal = microscope.get_signal(datamap, args.pixelsize, args.pdt, args.exc, args.sted)
signal_noisy = microscope_noisy.get_signal(datamap, args.pixelsize, args.pdt, args.exc, args.sted)

vmax = max(numpy.max(signal), numpy.max(signal_noisy))

pyplot.figure("Signal")
pyplot.imshow(signal, interpolation="nearest", vmin=0, vmax=vmax)
pyplot.colorbar()

pyplot.figure("Signal noisy")
pyplot.imshow(signal_noisy, interpolation="nearest", vmin=0, vmax=vmax)
pyplot.colorbar()

pyplot.show()

