import argparse

from matplotlib import cm, pyplot
import numpy
import tifffile

import time

from pysted import base, utils


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Test de script d'acquisition par BT :)")
# param obligatoire du pixel size, je pense que par défault on veut 10e-9
parser.add_argument("--pixelsize", type=float, default=10e-9, help="pixel size (in m) of the acquired image")
# params avec valeurs par défaut
parser.add_argument("--pdt", type=float, default=10e-6, help="pixel dwell time (in s)")
parser.add_argument("--exc", type=float, default=1e-6,  help="excitation power (in W)")
parser.add_argument("--sted", type=float, default=30e-3, help="STED power (in W)")
parser.add_argument("--dpxsz", type=float, default=10e-9, help="Pixel size of raw data")
parser.add_argument("--seed", type=str2bool, default=False, help="If True, every acquisition is seeded at 1 :)")
parser.add_argument("--bleach", type=str2bool, default=False, help="If True, new get_signal function will apply bleach"
                                                                  "at each iteration (how it should be, this is mostyl"
                                                                  "for testing purposes).")
args = parser.parse_args()

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
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector()   # noise=True, background=10e6
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)

# datamap tirée d'une image
datamap = tifffile.imread("examples/data/fibres.tif")
# normalize the datamap, 3 molecules/pixel at most
datamap = datamap / numpy.amax(datamap) * 3

# créer une datamap freestyle, juste se faire un array vide et ajouter des pts où tu veux
# datamap = numpy.zeros((100, 100))

# liste de pixels sur lesquels itérer
pixel_list = utils.pixel_sampling(datamap, mode="all")

if args.seed is True:
    numpy.random.seed(1)
signal, datamap_bleached = microscope.get_signal_bleach_mod(datamap, args.pixelsize, args.pdt, args.exc,
                                                                args.sted, args.dpxsz, pixel_list, bleach=args.bleach)

fig, axes = pyplot.subplots(1, 3)

datamap_imshow = axes[0].imshow(datamap, interpolation="nearest")
axes[0].set_title(f"Datamap, shape = {datamap.shape},")
fig.colorbar(datamap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

signal_imshow = axes[1].imshow(signal, interpolation="nearest")
axes[1].set_title(f"Aquired signal")
fig.colorbar(signal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

bleached_imshow = axes[2].imshow(datamap_bleached, interpolation="nearest")
axes[2].set_title(f"Bleached datamap, bleaching is {args.bleach}")
fig.colorbar(bleached_imshow, ax=axes[2], fraction=0.04, pad=0.05)

fig.suptitle(f"pixelsize = {args.pixelsize} m , datamap_pixelsize = {args.dpxsz} m, \n"
             f"Excitation beam power = {args.exc} W, STED beam power = {args.sted}, \n"
             f"Seed is {args.seed}")
figManager = pyplot.get_current_fig_manager()
figManager.window.showMaximized()
# pyplot.tight_layout()
pyplot.show()