import numpy as np
from matplotlib import pyplot as plt
import tqdm
from pysted import base, utils, temporal


"""
Le but de se script est de faire le framework pour une acquisition avec prise de décision.
Prise de décision ici signifie dire au microscope quels pixels imager selon l'information qu'on a.
L'information qu'on a correspondera à un scan confocal à plus basse résolution que les scans steds.
Après chaque décision et action, on enregistre l'état de l'acquisition 
Version 1 (simple):
    La prise de décision est simplement de faire un raster scan complet.
    Donc dans la loop, on va : 
        faire une acq confoc à basse résolution
        prendre une décision (juste faire un raster scan for now)
        Faire l'action
        enregistrer la dernière image sted, datamap, ...
        
FUCK LA VERSION 1, si je veux que ça marche je pense que j'ai pas le choix de faire des iterations
dans le time step du microscope. Par contre ça va me faire une loop de genre 100000000000000000000
itérations, alors comment est-ce que je veux gérer ça?
"""

# Get light curves stuff to generate the flashes later
event_file_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"
video_file_path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"

# Generate a datamap
frame_shape = (64, 64)
ensemble_func, synapses_list = utils.generate_synaptic_fibers(frame_shape, (9, 55), (3, 10), (2, 5))

poils_frame = ensemble_func.return_frame().astype(int)

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
        "phy_react": {488: 1e-4,   # 1e-4
                      575: 1e-8},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 10e-9
confoc_pxsize = 30e-9   # confoc ground truths will be taken at a resolution 3 times lower than sted scans
dpxsz = 10e-9
bleach = False
p_ex = 1e-6
p_sted = 30e-3
pdt = 10e-6   # pour (10, 1.5) ça me donne 15k pixels par iter
# pdt = 0.3   # pour (10, 1.5) ça me donne 0.5 pixels par iter
size = 64 + (2 * 22 + 1)
roi = 'max'
seed = True

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
# zero_residual controls how much of the donut beam "bleeds" into the the donut hole
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
# noise allows noise on the detector, background adds an average photon count for the empty pixels
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap = base.Datamap(poils_frame, dpxsz)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, bleach_func="default_bleach")
i_ex, _, _ = microscope.cache(datamap.pixelsize)
datamap.set_roi(i_ex, roi)

# Build a dictionnary corresponding synapses to a bool saying if they are currently flashing or not
# They all start not flashing
flat_synapses_list = [item for sublist in synapses_list for item in sublist]

synpase_flashing_dict, synapse_flash_idx_dict, synapse_flash_curve_dict, isolated_synapses_frames = \
    utils.generate_synapse_flash_dicts(flat_synapses_list, frame_shape)

# start acquisition loop
save_path = "D:/SCHOOL/Maitrise/H2021/Recherche/data_generation/time_integration/test_refactoring/"
flash_prob = 0.05   # every iteration, all synapses will have a 5% to start flashing
frozen_datamap = np.copy(datamap.whole_datamap[datamap.roi])
list_datamaps, list_confocals, list_steds = [], [], []
n_pixels_per_tstep, n_time_steps = utils.compute_time_correspondances((10, 1.5), 20, pdt)
print(f"pixels = {n_pixels_per_tstep}")
print(f"time steps = {n_time_steps}")