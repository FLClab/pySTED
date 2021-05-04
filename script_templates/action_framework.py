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
# find the shape of the lower resolution confocal ground truths
# this will be useful for verification purposes I think
# là live j'assume que mon acq sted sera toujours fait à la même résolution que ma datamap
ratio = utils.pxsize_ratio(confoc_pxsize, datamap.pixelsize)
confoc_n_rows, confoc_n_cols = int(np.ceil(frame_shape[0] / ratio)), int(np.ceil(frame_shape[1] / ratio))
# set starting pixel, for raster action I need to set them both to the origin [0, 0]
action_starting_pixel, confocal_starting_pixel = [0, 0], [0, 0]
confoc_valid_pixels_list = utils.pxsize_grid(confoc_pxsize, datamap.pixelsize, datamap.whole_datamap[datamap.roi])
confoc_intensity = np.zeros((confoc_n_rows, confoc_n_cols)).astype(float)
sted_intensity = np.zeros(frozen_datamap.shape).astype(float)
confoc_taken, action_taken = False, False
for i in tqdm.trange(n_time_steps):
    datamap.whole_datamap[datamap.roi] = np.copy(frozen_datamap)  # essayer np.copy?

    # loop through all synapses, make some start to flash, randomly, maybe
    for idx_syn in range(len(flat_synapses_list)):
        if np.random.binomial(1, flash_prob) and synpase_flashing_dict[idx_syn] is False:
            # can start the flash
            synpase_flashing_dict[idx_syn] = True
            synapse_flash_idx_dict[idx_syn] = 1
            sampled_curve = utils.flash_generator_old(event_file_path, video_file_path)
            synapse_flash_curve_dict[idx_syn] = utils.rescale_data(sampled_curve, to_int=True, divider=3)

        if synpase_flashing_dict[idx_syn]:
            datamap.whole_datamap[datamap.roi] -= isolated_synapses_frames[idx_syn]
            datamap.whole_datamap[datamap.roi] += isolated_synapses_frames[idx_syn] * \
                                                  synapse_flash_curve_dict[idx_syn][synapse_flash_idx_dict[idx_syn]]
            synapse_flash_idx_dict[idx_syn] += 1
            if synapse_flash_idx_dict[idx_syn] >= 40:
                synapse_flash_idx_dict[idx_syn] = 0
                synpase_flashing_dict[idx_syn] = False

    n_pixels_this_iter = int(n_pixels_per_tstep) + microscope.take_from_pixel_bank()

    if not confoc_taken and not action_taken:
        # cas 1 où je dois faire l'acquisition confocale au complet et ensuite faire ma décision + agir

        # construire une liste de pixels (ET LA FILTRER) pour le raster en confoc à plus basse résolution
        low_res_confoc_plist = utils.generate_raster_pixel_list(n_pixels_this_iter, confocal_starting_pixel,
                                                                frozen_datamap)
        low_res_confoc_plist = utils.pixel_list_filter(frozen_datamap, low_res_confoc_plist, confoc_pxsize,
                                                       datamap.pixelsize, output_empty=True)

        if len(low_res_confoc_plist) < 1:
            # need to break free
            # print("la liste est de la bonne longueure!")
            microscope.add_to_pixel_bank(n_pixels_per_tstep)
            continue

        # faire le scan confocal sur la pixel_list
        # DOIT TROUVER COMMENT TESTER LE CAS OÙ ÇA LUI PREND PLUS QU'UNE LOOP ICI POUR FINIR L'ACQ
        confoc_acq, _, confoc_intensity = microscope.get_signal_and_bleach_fast(datamap, confoc_pxsize, pdt, p_ex, 0.0,
                                                                                acquired_intensity=confoc_intensity,
                                                                                pixel_list=low_res_confoc_plist,
                                                                                bleach=bleach, update=False,
                                                                                filter_bypass=True)

        # vérifier si le dernier pixel était le pixel en bas à droite,
        # vérifier combien de pixels je peux faire encore, idk
        if low_res_confoc_plist[-1] == confoc_valid_pixels_list[-1]:
            # mettre confoc_taken à True, faire commencer l'action
            confoc_taken = True

            # calculer combien de pixels il me reste à imager dans cette iter de la loop for

        else:
            # set le confoc starting pixel comme il faut? not sure what to do here
            pass

        # faire les vérifs avec n_pixels_this_iter pour vérifier combien d'iter de loops j'aurai besoin de passer ici
        exit()

    elif confoc_taken and not action_taken:
        # cas 2 où je suis rendu à faire l'acquisition STED
        pass
    elif confoc_taken and action_taken:
        # cas 3 où je suis rendu à reset
        pass
    else:   # (if not confoc_taken and action_taken)
        # cas 4 qui devrait être impossible, right?
        pass

    # clair que je vais devoir faire ça mais pt que c'est pas ici la bonne place
    # print("BROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    # microscope.add_to_pixel_bank(n_pixels_per_tstep)
