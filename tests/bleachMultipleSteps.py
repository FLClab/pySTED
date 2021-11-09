
import numpy
import time
import random

from pysted import base, utils, raster, bleach_funcs
from matplotlib import pyplot
from collections import defaultdict
from tqdm import tqdm

"""
This is a test in which I bleach in multiple steps. The complete pixel dwelltime
is the same for all tests, i.e. 100 microseconds.
In the multiple steps implementation, I will use 10 steps of 10 microseconds.

I compare the bleaching of the datamap if the bleaching is performed at every steps
OR if it is a combination of all the steps at the current pixel.

get_signal_and_bleach
    Implements the default bleach without steps

get_signal_and_bleach_test1
    Implements the bleach at every steps of each pixel

get_signal_and_bleach_test2
    Implements the bleach after steps completion at each pixel

In all tests, the STED power is set at 0.
"""

class DebugMicroscope(base.Microscope):
    def __init__(self, excitation, sted, detector, objective, fluo):
        super(DebugMicroscope, self).__init__(excitation, sted, detector, objective, fluo)

    def get_signal_and_bleach_test1(self, datamap, pixelsize, pdt, p_ex, p_sted, indices=None, acquired_intensity=None,
                                  pixel_list=None, bleach=True, update=True, seed=None, filter_bypass=False,
                                  bleach_func=bleach_funcs.default_bleach):

        datamap_roi = datamap.whole_datamap[datamap.roi]
        pdt = utils.float_to_array_verifier(pdt, datamap_roi.shape)
        p_ex = utils.float_to_array_verifier(p_ex, datamap_roi.shape)
        p_sted = utils.float_to_array_verifier(p_sted, datamap_roi.shape)

        datamap_pixelsize = datamap.pixelsize
        i_ex, i_sted, psf_det = self.cache(datamap_pixelsize)
        if datamap.roi is None:
            datamap.set_roi(i_ex)

        datamap_roi = datamap.whole_datamap[datamap.roi]
        pixel_list = utils.pixel_list_filter(datamap_roi, pixel_list, pixelsize, datamap_pixelsize)

        ratio = utils.pxsize_ratio(pixelsize, datamap_pixelsize)
        rows_pad, cols_pad = datamap.roi_corners['tl'][0], datamap.roi_corners['tl'][1]
        laser_pad = i_ex.shape[0] // 2

        prob_ex = numpy.ones(datamap.whole_datamap.shape)
        prob_sted = numpy.ones(datamap.whole_datamap.shape)
        bleached_datamap = numpy.copy(datamap.whole_datamap)
        returned_photons = numpy.zeros(datamap.whole_datamap[datamap.roi].shape)
        scaled_power = numpy.zeros(datamap.whole_datamap[datamap.roi].shape)

        bleached_sub_datamaps_dict = {}
        if isinstance(indices, type(None)):
            indices = 0   # VÉRIF À QUOI INDICES SERT?
        for key in datamap.sub_datamaps_dict:
            bleached_sub_datamaps_dict[key] = numpy.copy(datamap.sub_datamaps_dict[key])

        for (row, col) in tqdm(pixel_list, desc="Pixels", leave=False):
            row_slice = slice(row + rows_pad - laser_pad, row + rows_pad + laser_pad + 1)
            col_slice = slice(col + cols_pad - laser_pad, col + cols_pad + laser_pad + 1)

            scale_power, threshold_count, steps = 0., 0., 10
            for _ in range(0, int(pdt[row, col] * 1e+6), steps):

                decision_time = steps * 1e-6

                effective = self.get_effective(datamap_pixelsize, p_ex[row, col], scale_power * p_sted[row, col])
                h, w = effective.shape

                # Uses the bleached datamap
                bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int32)
                for key in bleached_sub_datamaps_dict:
                    bleached_datamap += bleached_sub_datamaps_dict[key]

                pixel_intensity = numpy.sum(effective * bleached_datamap[row_slice, col_slice])
                pixel_photons = self.detector.get_signal(self.fluo.get_photons(pixel_intensity), decision_time)

                if bleach:
                    bleach_func(self, i_ex, i_sted, p_ex[row, col], scale_power * p_sted[row, col],
                                decision_time, bleached_sub_datamaps_dict,
                                row, col, h, w, prob_ex, prob_sted)

                # If signal is less than threshold count then skip pixel
                scaled_power[row, col] = scale_power
                if pixel_photons < threshold_count:
                    break

                # Update the photon counts only on the last pixel power scale
                if scale_power > 0.:
                    returned_photons[row, col] += pixel_photons

        if update and bleach:
            datamap.sub_datamaps_dict = bleached_sub_datamaps_dict
            datamap.base_datamap = datamap.sub_datamaps_dict["base"]
            datamap.whole_datamap = numpy.copy(datamap.base_datamap)

        return returned_photons, bleached_sub_datamaps_dict, scaled_power

    def get_signal_and_bleach_test2(self, datamap, pixelsize, pdt, p_ex, p_sted, indices=None, acquired_intensity=None,
                                  pixel_list=None, bleach=True, update=True, seed=None, filter_bypass=False,
                                  bleach_func=bleach_funcs.default_bleach_multisteps):

        datamap_roi = datamap.whole_datamap[datamap.roi]
        pdt = utils.float_to_array_verifier(pdt, datamap_roi.shape)
        p_ex = utils.float_to_array_verifier(p_ex, datamap_roi.shape)
        p_sted = utils.float_to_array_verifier(p_sted, datamap_roi.shape)

        datamap_pixelsize = datamap.pixelsize
        i_ex, i_sted, psf_det = self.cache(datamap_pixelsize)
        if datamap.roi is None:
            datamap.set_roi(i_ex)

        datamap_roi = datamap.whole_datamap[datamap.roi]
        pixel_list = utils.pixel_list_filter(datamap_roi, pixel_list, pixelsize, datamap_pixelsize)

        ratio = utils.pxsize_ratio(pixelsize, datamap_pixelsize)
        rows_pad, cols_pad = datamap.roi_corners['tl'][0], datamap.roi_corners['tl'][1]
        laser_pad = i_ex.shape[0] // 2

        prob_ex = numpy.ones(datamap.whole_datamap.shape)
        prob_sted = numpy.ones(datamap.whole_datamap.shape)
        bleached_datamap = numpy.copy(datamap.whole_datamap)
        returned_photons = numpy.zeros(datamap.whole_datamap[datamap.roi].shape)
        scaled_power = numpy.zeros(datamap.whole_datamap[datamap.roi].shape)

        bleached_sub_datamaps_dict = {}
        if isinstance(indices, type(None)):
            indices = 0   # VÉRIF À QUOI INDICES SERT?
        for key in datamap.sub_datamaps_dict:
            bleached_sub_datamaps_dict[key] = numpy.copy(datamap.sub_datamaps_dict[key])

        for (row, col) in tqdm(pixel_list, desc="Pixels", leave=False):
            row_slice = slice(row + rows_pad - laser_pad, row + rows_pad + laser_pad + 1)
            col_slice = slice(col + cols_pad - laser_pad, col + cols_pad + laser_pad + 1)

            scale_power, threshold_count, steps = 0., 0., 10
            pdts, p_exs, p_steds = [], [], []
            for _ in range(0, int(pdt[row, col] * 1e+6), steps):

                decision_time = steps * 1e-6

                effective = self.get_effective(datamap_pixelsize, p_ex[row, col], scale_power * p_sted[row, col])
                h, w = effective.shape

                # Uses the bleached datamap
                bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int32)
                for key in bleached_sub_datamaps_dict:
                    bleached_datamap += bleached_sub_datamaps_dict[key]

                pixel_intensity = numpy.sum(effective * bleached_datamap[row_slice, col_slice])
                pixel_photons = self.detector.get_signal(self.fluo.get_photons(pixel_intensity), decision_time)

                # If signal is less than threshold count then skip pixel
                scaled_power[row, col] = scale_power
                if pixel_photons < threshold_count:
                    break

                # Update the photon counts only on the last pixel power scale
                if scale_power > 0.:
                    returned_photons[row, col] += pixel_photons

                pdts.append(decision_time)
                p_exs.append(p_ex[row, col])
                p_steds.append(scale_power * p_sted[row, col])

            pdts, p_exs, p_steds = map(numpy.array, (pdts, p_exs, p_steds))
            if bleach:
                bleach_func(self, i_ex, i_sted, p_exs, p_steds,
                            pdts, bleached_sub_datamaps_dict,
                            row, col, h, w, prob_ex, prob_sted)
        if update and bleach:
            datamap.sub_datamaps_dict = bleached_sub_datamaps_dict
            datamap.base_datamap = datamap.sub_datamaps_dict["base"]
            datamap.whole_datamap = numpy.copy(datamap.base_datamap)

        return returned_photons, bleached_sub_datamaps_dict, scaled_power

START = time.time()

out = defaultdict(list)

delta = 1
molecules_disposition = numpy.zeros((50, 50))
num_mol = 2
for j in range(1,4):
    for i in range(1,4):
        molecules_disposition[
            j * molecules_disposition.shape[0]//4 - delta : j * molecules_disposition.shape[0]//4 + delta + 1,
            i * molecules_disposition.shape[1]//4 - delta : i * molecules_disposition.shape[1]//4 + delta + 1] = num_mol

egfp = {"lambda_": 535e-9,
        "qy": 0.6,
        "sigma_abs": {488: 3e-20,
                      575: 6e-21},
        "sigma_ste": {560: 1.2e-20,
                      575: 6.0e-21,
                      580: 5.0e-21},
        "sigma_tri": 1e-21,
        "tau": 3e-09,
        "tau_vib": 1.0e-12,
        "tau_tri": 5e-6,
        "phy_react": {488: 0.25e-7,   # 1e-4
                      575: 0.25e-11},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 20e-9
bleach = True
p_ex = 2e-6
p_ex_array = numpy.ones(molecules_disposition.shape) * p_ex
p_sted = 2.5e-3
p_sted = 0.
p_sted_array = numpy.ones(molecules_disposition.shape) * p_sted
pdt = 100e-6
pdt_array = numpy.ones(molecules_disposition.shape) * pdt
roi = 'max'

print_msg = """
Imaging parameters
==================
Bleach : {bleach}
p_ex : {p_ex}
p_sted : {p_sted}
pdt : {pdt}
==================
""".format(bleach=bleach, p_ex=p_ex, p_sted=p_sted, pdt=pdt)
print(print_msg)

seed, reps = 42, 5
for i in range(reps):
    seed = seed + i

    numpy.random.seed(seed)
    random.seed(seed)

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(488e-9)
    laser_sted = base.DonutBeam(575e-9, zero_residual=0)
    detector = base.Detector(noise=False, background=0, pcef=0.1)
    objective = base.Objective()
    fluo = base.Fluorescence(**egfp)
    datamap = base.Datamap(molecules_disposition, pixelsize)
    microscope = DebugMicroscope(laser_ex, laser_sted, detector, objective, fluo)
    start = time.time()
    i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)
    datamap.set_roi(i_ex, roi)

    time_start = time.time()
    acquisition, bleached, scaled_power = microscope.get_signal_and_bleach_test1(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
                                                                        bleach=bleach, update=False, seed=seed)
    print()
    print("bleach every steps of the pixel (get_signal_and_bleach_test1)")
    print(f"ran in {time.time() - time_start} s")
    print("Average molecules left : ", bleached["base"][datamap.roi][molecules_disposition != 0].mean(axis=-1))

    out["get_signal_and_bleach_test1"].append(bleached["base"][datamap.roi][molecules_disposition != 0].mean(axis=-1) / num_mol)

seed, reps = 42, 5
for i in range(reps):
    seed = seed + i

    numpy.random.seed(seed)
    random.seed(seed)

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(488e-9)
    laser_sted = base.DonutBeam(575e-9, zero_residual=0)
    detector = base.Detector(noise=False, background=0, pcef=0.1)
    objective = base.Objective()
    fluo = base.Fluorescence(**egfp)
    datamap = base.Datamap(molecules_disposition, pixelsize)
    microscope = DebugMicroscope(laser_ex, laser_sted, detector, objective, fluo)
    start = time.time()
    i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)
    datamap.set_roi(i_ex, roi)

    time_start = time.time()
    acquisition, bleached, scaled_power = microscope.get_signal_and_bleach_test2(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
                                                                        bleach=bleach, update=False, seed=seed)

    print()
    print("bleach every pixel with array (get_signal_and_bleach_test2)")
    print(f"ran in {time.time() - time_start} s")
    print("Average molecules left : ", bleached["base"][datamap.roi][molecules_disposition != 0].mean(axis=-1))

    out["get_signal_and_bleach_test2"].append(bleached["base"][datamap.roi][molecules_disposition != 0].mean(axis=-1) / num_mol)

seed, reps = 42, 5
for i in range(reps):
    seed = seed + i

    numpy.random.seed(seed)
    random.seed(seed)

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(488e-9)
    laser_sted = base.DonutBeam(575e-9, zero_residual=0)
    detector = base.Detector(noise=False, background=0, pcef=0.1)
    objective = base.Objective()
    fluo = base.Fluorescence(**egfp)
    datamap = base.Datamap(molecules_disposition, pixelsize)
    microscope = DebugMicroscope(laser_ex, laser_sted, detector, objective, fluo)
    start = time.time()
    i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)
    datamap.set_roi(i_ex, roi)

    time_start = time.time()
    acquisition, bleached, scaled_power = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
                                                                        bleach=bleach, update=False, seed=seed)

    print()
    print("Normal bleach (get_signal_and_bleach)")
    print(f"ran in {time.time() - time_start} s")
    print("Average molecules left : ", bleached["base"][datamap.roi][molecules_disposition != 0].mean(axis=-1))

    out["get_signal_and_bleach"].append(bleached["base"][datamap.roi][molecules_disposition != 0].mean(axis=-1) / num_mol)


fig, ax = pyplot.subplots()
keys = list(sorted(out.keys()))
for i, key in enumerate(keys):
    data = out[key]
    mean, std = numpy.mean(data), numpy.std(data)
    ax.bar(i, mean, yerr=std, label=key)
ax.legend(loc="lower right")
ax.set(
    ylabel="Average Mol. Left (ratio)"
)
fig.savefig(f"./panels/bleachMultipleSteps_{num_mol}.pdf", bbox_inches="tight", transparent=True)
pyplot.show()
