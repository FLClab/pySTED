"""
Dans ce fichier je veux faire tous les tests relatif à ma génération de laser, qui devrait produire les lasers de façon
symmétrique, mais ne le fait pas.
"""

# Import packages
import argparse

from matplotlib import pyplot, image
import numpy
import tifffile
import os, datetime
from tkinter.filedialog import askopenfilename
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


parser = argparse.ArgumentParser(description="Exemple de scripte d'acquisition")
parser.add_argument("--exc", type=float, default=1e-6,  help="excitation power (in W)")
parser.add_argument("--sted", type=float, default=30e-3, help="STED power (in W)")
parser.add_argument("--zero_residual", type=float, default=0, help="Fraction of the doughnut beam that bleeds into"
                                                                   "the centre (between 0 and 1)")
parser.add_argument("--pixelsize", type=float, default=10e-9, help="Displacement of laser between pulses. Must be a "
                                                                   "multiple of the datamap_pixelsize, 10 nm. (m)")
parser.add_argument("--dpxsz", type=float, default=10e-9, help="Size of a pixel in the datamap. (m)")
parser.add_argument("--symmetry", type=str2bool, default=False, help="Determines if lasers are ratchet-made symmetric")
args = parser.parse_args()

# fluorescence parameters
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
laser_sted = base.DonutBeam(575e-9, zero_residual=args.zero_residual)
detector = base.Detector(noise=True)   # background=10e6
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)

print("Starting tests :)")

print("Intermediate test ???")
"""
            f, n, na = self.objective.f, self.objective.n, self.objective.na
            
            transmission = self.objective.get_transmission(self.excitation.lambda_)
            i_ex = self.excitation.get_intensity(1, f, n, na,
                                                 transmission, data_pixelsize)
"""
f, n, na = objective.f, objective.n, objective.na
transmission = objective.get_transmission(laser_ex.lambda_)
i_ex = laser_ex.get_intensity(1, f, n, na, transmission, args.dpxsz)
i_sted = laser_sted.get_intensity(1, f, n, na, transmission, args.dpxsz)
psf = fluo.get_psf(na, args.dpxsz)
psf_det = detector.get_detection_psf(fluo.lambda_, psf, na, transmission, args.dpxsz)
print(f"i_ex.shape = {i_ex.shape}, i_sted.shape = {i_sted.shape}, psf_det.shape = {psf_det.shape}")

# LASER GENERATION
start = time.time()
if args.symmetry:
    print("Using symmetrical lasers")
    i_ex, i_sted, psf_det = microscope.cache_verif(args.pixelsize, data_pixelsize=args.dpxsz)
else:
    print(f"Using default (non-symmetrical) lasers")
    i_ex, i_sted, psf_det = microscope.cache(args.pixelsize, data_pixelsize=args.dpxsz)
end = time.time()

ex_sym_vert = utils.symmetry_verifier(i_ex, direction="vertical")
ex_sym_horiz = utils.symmetry_verifier(i_ex, direction="horizontal")
sted_sym_vert = utils.symmetry_verifier(i_sted, direction="vertical")
sted_sym_horiz = utils.symmetry_verifier(i_sted, direction="horizontal")
psf_sym_vert = utils.symmetry_verifier(psf_det, direction="vertical")
psf_sym_horiz = utils.symmetry_verifier(psf_det, direction="horizontal")

fig, axes = pyplot.subplots(3, 3)

ex_imshow = axes[0, 0].imshow(i_ex)
axes[0, 0].set_title(f"Excitation beam")
fig.colorbar(ex_imshow, ax=axes[0, 0], fraction=0.04, pad=0.05)

ex_vert_imshow = axes[1, 0].imshow(ex_sym_vert)
axes[1, 0].set_title(f"Excitation beam vertical symmetry")
fig.colorbar(ex_vert_imshow, ax=axes[1, 0], fraction=0.04, pad=0.05)

ex_horiz_imshow = axes[2, 0].imshow(ex_sym_horiz)
axes[2, 0].set_title(f"Excitation beam horizontal symmetry")
fig.colorbar(ex_horiz_imshow, ax=axes[2, 0], fraction=0.04, pad=0.05)

sted_imshow = axes[0, 1].imshow(i_sted)
axes[0, 1].set_title(f"STED beam")
fig.colorbar(sted_imshow, ax=axes[0, 1], fraction=0.04, pad=0.05)

sted_vert_imshow = axes[1, 1].imshow(sted_sym_vert)
axes[1, 1].set_title(f"STED beam vertical symmetry")
fig.colorbar(sted_vert_imshow, ax=axes[1, 1], fraction=0.04, pad=0.05)

sted_horiz_imshow = axes[2, 1].imshow(sted_sym_horiz)
axes[2, 1].set_title(f"STED beam horizontal symmetry")
fig.colorbar(sted_horiz_imshow, ax=axes[2, 1], fraction=0.04, pad=0.05)

psf_imshow = axes[0, 2].imshow(psf_det)
axes[0, 2].set_title(f"PSF")
fig.colorbar(psf_imshow, ax=axes[0, 2], fraction=0.04, pad=0.05)

psf_vert_imshow = axes[1, 2].imshow(psf_sym_vert)
axes[1, 2].set_title(f"PSF vertical symmetry")
fig.colorbar(psf_vert_imshow, ax=axes[1, 2], fraction=0.04, pad=0.05)

psf_horiz_imshow = axes[2, 2].imshow(psf_sym_horiz)
axes[2, 2].set_title(f"PSF horizontal symmetry")
fig.colorbar(psf_horiz_imshow, ax=axes[2, 2], fraction=0.04, pad=0.05)

fig.suptitle(f"datamap_pixelsize = {args.dpxsz}, laser shape = {i_ex.shape}\n"
             f"took {round(end - start, 5)} s to generate lasers")
figManager = pyplot.get_current_fig_manager()
figManager.window.showMaximized()
pyplot.show()
