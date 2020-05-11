
import argparse

from matplotlib import cm, pyplot
import numpy
import tifffile

from pysted import base, utils


parser = argparse.ArgumentParser(description="""Simulate the acquisition of an
                                                image serie with bleaching.""")
parser.add_argument("pixelsize", type=float,
                    help="pixelsize (in m) of the acquired images")
parser.add_argument("stacksize", type=int,
                    help="number of images to acquire")
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
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)

datamap = utils.rescale(datamap, int(args.pixelsize / 10e-9))

pyplot.figure("Data map")
pyplot.imshow(datamap, interpolation="nearest")
pyplot.colorbar()

for i in range(args.stacksize):
    signal = microscope.get_signal(datamap, args.pixelsize, args.pdt, args.exc, args.sted)

    pyplot.figure("Signal ("+str(i+1)+")")
    pyplot.imshow(signal, interpolation="nearest")
    pyplot.colorbar()
    
    pdtmap = numpy.full(signal.shape, args.pdt)
    
    datamap = microscope.bleach(datamap, args.pixelsize, pdtmap, args.exc, args.sted)

pyplot.show()

