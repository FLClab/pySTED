
import numpy
import time
import random

from pysted import base, utils, raster, bleach_funcs
# from dymin import DyMINMicroscope
from matplotlib import pyplot
from collections import defaultdict
from tqdm import tqdm, trange

START = time.time()

# molecules_disposition = (numpy.random.rand(40, 40) > 0.9).astype(int) * 10
delta = 1
molecules_disposition = numpy.zeros((50, 50))
# molecules_disposition[
#     molecules_disposition.shape[0]//2 - delta : molecules_disposition.shape[0]//2+delta,
#     molecules_disposition.shape[1]//2 - delta : molecules_disposition.shape[1]//2+delta] = 8
num_mol = 10
for i in range(1, 4):
    for j in range(1, 4):
        molecules_disposition[
            i * molecules_disposition.shape[0]//4,
            j * molecules_disposition.shape[1]//4] = num_mol

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
pixelsize = 20e-9
bleach = True
p_ex = 2e-6
p_ex_array = numpy.ones(molecules_disposition.shape) * p_ex
p_sted = 2.5e-3
p_sted_array = numpy.ones(molecules_disposition.shape) * p_sted
pdt = 100e-6
pdt_array = numpy.ones(molecules_disposition.shape) * pdt
roi = 'max'

out = defaultdict(list)
argwhere = numpy.argwhere(molecules_disposition)
for i in trange(10, desc="Reps"):

    numpy.random.seed(42 + i)
    random.seed(42 + i)

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(488e-9)
    laser_sted = base.DonutBeam(575e-9, zero_residual=0)
    detector = base.Detector(noise=False, background=0)
    objective = base.Objective()
    fluo = base.Fluorescence(**egfp)
    datamap = base.Datamap(molecules_disposition, pixelsize)
    # microscope = DyMINMicroscope(laser_ex, laser_sted, detector, objective, fluo)
    microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)
    start = time.time()
    i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)
    datamap.set_roi(i_ex, roi)

    time_start = time.time()
    values = []
    for _ in trange(2, desc="Bleach", leave=False):

        left_mol = datamap.sub_datamaps_dict["base"][datamap.roi][argwhere[:, 0], argwhere[:, 1]]
        values.append(left_mol)

        acquisition, bleached, scaled_power = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
                                                                            bleach=bleach, update=True)
    for arg, value in zip(argwhere, numpy.array(values).T):
        # for arg, val in zip(argwhere, value):
        out["{}".format(arg//12)].append(value)

fig, ax = pyplot.subplots(figsize=(3, 3))
cmap = pyplot.cm.get_cmap("tab10")
for i, (key, values) in enumerate(out.items()):
    mean, std = numpy.mean(values, axis=0), numpy.std(values, axis=0)
    x = numpy.arange(len(mean))
    ax.plot(x, mean, label=key, color=cmap(i))
    # ax.fill_between(x, mean - std, mean + std, color=cmap(i), alpha=0.3)
ax.set(
    ylabel="Molecules left", xlabel="Num. scans"
)
ax.legend()
fig.savefig("./panels/bleach.pdf", transparent=True, bbox_inches="tight")
pyplot.show()
