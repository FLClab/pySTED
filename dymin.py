
import numpy
import time
import random

from pysted import base, utils, raster, bleach_funcs
from tqdm import tqdm

class DyMINMicroscope(base.Microscope):
    def __init__(self, excitation, sted, detector, objective, fluo, opts=None):
        super(DyMINMicroscope, self).__init__(excitation, sted, detector, objective, fluo)

        if isinstance(opts, type(None)):
            opts = {
                "scale_power" : [0., 0.25, 1.],
                "decision_time" : [10e-6, 10e-6, -1],
                "threshold_count" : [10, 5, 0]
            }
        required_keys = ["scale_power", "decision_time", "threshold_count"]
        assert all(k in opts for k in required_keys), "Missing keys in opts. {}".format(required_keys)
        self.opts = opts

    def get_signal_and_bleach(self, datamap, pixelsize, pdt, p_ex, p_sted, indices=None, acquired_intensity=None,
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

        for (row, col) in pixel_list:
            row_slice = slice(row + rows_pad - laser_pad, row + rows_pad + laser_pad + 1)
            col_slice = slice(col + cols_pad - laser_pad, col + cols_pad + laser_pad + 1)

            pdts, p_exs, p_steds = numpy.zeros(len(self.opts["scale_power"])), numpy.zeros(len(self.opts["scale_power"])), numpy.zeros(len(self.opts["scale_power"]))
            for i, (scale_power, decision_time, threshold_count) in enumerate(zip(self.opts["scale_power"], self.opts["decision_time"], self.opts["threshold_count"])):
                if decision_time < 0.:
                    decision_time = pdt[row, col]

                effective = self.get_effective(datamap_pixelsize, p_ex[row, col], scale_power * p_sted[row, col])
                h, w = effective.shape

                # Uses the bleached datamap
                bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int32)
                for key in bleached_sub_datamaps_dict:
                    bleached_datamap += bleached_sub_datamaps_dict[key]

                pixel_intensity = numpy.sum(effective * bleached_datamap[row_slice, col_slice])
                pixel_photons = self.detector.get_signal(self.fluo.get_photons(pixel_intensity), decision_time)

                # Stores the action taken for futures bleaching
                pdts[i] = decision_time
                p_exs[i] = p_ex[row, col]
                p_steds[i] = scale_power * p_sted[row, col]

                # If signal is less than threshold count then skip pixel
                scaled_power[row, col] = scale_power
                if pixel_photons < threshold_count:
                    break

                # Update the photon counts only on the last pixel power scale
                if scale_power > 0.:
                    returned_photons[row, col] += pixel_photons

            if bleach:
                bleach_func(self, i_ex, i_sted, p_exs, p_steds,
                            pdts, bleached_sub_datamaps_dict,
                            row, col, h, w, prob_ex, prob_sted)

        if update and bleach:
            datamap.sub_datamaps_dict = bleached_sub_datamaps_dict
            datamap.base_datamap = datamap.sub_datamaps_dict["base"]
            datamap.whole_datamap = numpy.copy(datamap.base_datamap)

        return returned_photons, bleached_sub_datamaps_dict, scaled_power

    # def get_signal_and_bleach(self, datamap, pixelsize, pdt, p_ex, p_sted, indices=None, acquired_intensity=None,
    #                               pixel_list=None, bleach=True, update=True, seed=None, filter_bypass=False,
    #                               bleach_func=bleach_funcs.default_bleach_multisteps):
    #
    #     if seed is not None:
    #         numpy.random.seed(seed)
    #     datamap_pixelsize = datamap.pixelsize
    #     i_ex, i_sted, psf_det = self.cache(datamap_pixelsize)
    #
    #     # maybe I should just throw an error here instead
    #     if datamap.roi is None:
    #         # demander au dude de setter une roi
    #         datamap.set_roi(i_ex)
    #
    #     datamap_roi = datamap.whole_datamap[datamap.roi]
    #
    #     # convert scalar values to arrays if they aren't already arrays
    #     # C funcs need pre defined types, so in order to only have 1 general case C func, I convert scalars to arrays
    #     pdt = utils.float_to_array_verifier(pdt, datamap_roi.shape)
    #     p_ex = utils.float_to_array_verifier(p_ex, datamap_roi.shape)
    #     p_sted = utils.float_to_array_verifier(p_sted, datamap_roi.shape)
    #
    #     if not filter_bypass:
    #         pixel_list = utils.pixel_list_filter(datamap_roi, pixel_list, pixelsize, datamap_pixelsize)
    #
    #     # *** VÉRIFIER SI CE TO DO LÀ EST FAIT ***
    #     # TODO: make sure I handle passing an acq matrix correctly / verifying its shape and shit
    #     ratio = utils.pxsize_ratio(pixelsize, datamap_pixelsize)
    #     if acquired_intensity is None:
    #         acquired_intensity = numpy.zeros((int(numpy.ceil(datamap_roi.shape[0] / ratio)),
    #                                           int(numpy.ceil(datamap_roi.shape[1] / ratio))))
    #     else:
    #         # verify the shape and shit
    #         pass
    #     scaled_power = numpy.zeros_like(acquired_intensity)
    #     rows_pad, cols_pad = datamap.roi_corners['tl'][0], datamap.roi_corners['tl'][1]
    #     laser_pad = i_ex.shape[0] // 2
    #
    #
    #     prob_ex = numpy.ones(datamap.whole_datamap.shape)
    #     prob_sted = numpy.ones(datamap.whole_datamap.shape)
    #     # bleached_sub_datamaps_dict = copy.copy(datamap.sub_datamaps_dict)
    #     bleached_sub_datamaps_dict = {}
    #     if isinstance(indices, type(None)):
    #         indices = 0   # VÉRIF À QUOI INDICES SERT?
    #     for key in datamap.sub_datamaps_dict:
    #         bleached_sub_datamaps_dict[key] = numpy.copy(datamap.sub_datamaps_dict[key])
    #
    #     if seed is None:
    #         seed = 0
    #
    #     raster_func = raster.raster_func_c_self_bleach_dymin
    #     raster_func(self, datamap, acquired_intensity, scaled_power, numpy.array(pixel_list).astype(numpy.int32), ratio, rows_pad,
    #                 cols_pad, laser_pad, prob_ex, prob_sted, pdt, p_ex, p_sted, bleach, bleached_sub_datamaps_dict,
    #                 seed, bleach_func)
    #
    #     # Bleaching is done, the rest is for intensity calculation
    #     photons = self.fluo.get_photons(acquired_intensity)
    #
    #     if photons.shape == pdt.shape:
    #         returned_acquired_photons = self.detector.get_signal(photons, pdt)
    #     else:
    #         pixeldwelltime_reshaped = numpy.zeros((int(numpy.ceil(pdt.shape[0] / ratio)),
    #                                                int(numpy.ceil(pdt.shape[1] / ratio))))
    #         new_pdt_plist = utils.pixel_sampling(pixeldwelltime_reshaped, mode='all')
    #         for (row, col) in new_pdt_plist:
    #             pixeldwelltime_reshaped[row, col] = pdt[row * ratio, col * ratio]
    #         returned_acquired_photons = self.detector.get_signal(photons, pixeldwelltime_reshaped)
    #
    #     if update and bleach:
    #         datamap.sub_datamaps_dict = bleached_sub_datamaps_dict
    #         datamap.base_datamap = datamap.sub_datamaps_dict["base"]
    #         datamap.whole_datamap = numpy.copy(datamap.base_datamap)
    #         # BLEACHER LES FLASHS FUTURS
    #         # pt que je dois ajouter un if indices < flash_tstack.shape[0] aussi
    #         if datamap.contains_sub_datamaps["flashes"] and indices["flashes"] < datamap.flash_tstack.shape[0]:
    #             datamap.bleach_future(indices, bleached_sub_datamaps_dict)
    #
    #     return returned_acquired_photons, bleached_sub_datamaps_dict, scaled_power
