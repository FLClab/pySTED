
import numpy
import time
import random

from pysted import base, utils, raster, bleach_funcs

class DyMINMicroscope(base.Microscope):
    def __init__(self, excitation, sted, detector, objective, fluo, load_cache=False, opts=None):
        super(DyMINMicroscope, self).__init__(excitation, sted, detector, objective, fluo, load_cache=load_cache)

        if isinstance(opts, type(None)):
            opts = {
                "scale_power" : [0., 0.25, 1.],
                "decision_time" : [10.0e-6, 10.0e-6, -1],
                "threshold_count" : [8, 8, 0]
            }
        required_keys = ["scale_power", "decision_time", "threshold_count"]
        assert all(k in opts for k in required_keys), "Missing keys in opts. {}".format(required_keys)
        self.opts = opts

    def get_signal_and_bleach(self, datamap, pixelsize, pdt, p_ex, p_sted, indices=None, acquired_intensity=None,
                                  pixel_list=None, bleach=True, update=True, seed=None, filter_bypass=False,
                                  bleach_func=bleach_funcs.default_update_survival_probabilities,
                                  sample_func=bleach_funcs.sample_molecules):

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
        acquired_intensity = numpy.zeros(datamap.whole_datamap[datamap.roi].shape)

        bleached_sub_datamaps_dict = {}
        if isinstance(indices, type(None)):
            indices = 0   # VÉRIF À QUOI INDICES SERT?
        for key in datamap.sub_datamaps_dict:
            bleached_sub_datamaps_dict[key] = numpy.copy(datamap.sub_datamaps_dict[key].astype(numpy.int64))

        if isinstance(seed, type(None)):
            seed = 0
        for key, value in self.opts.items():
            if isinstance(value, list):
                self.opts[key] = numpy.array(value)
            self.opts[key] = self.opts[key].astype(float)

        raster_func = raster.raster_func_dymin
        raster_func(self, datamap, acquired_intensity, numpy.array(pixel_list).astype(numpy.int32), ratio, rows_pad,
                    cols_pad, laser_pad, prob_ex, prob_sted, returned_photons, scaled_power, pdt, p_ex, p_sted,
                    bleach, bleached_sub_datamaps_dict, seed, bleach_func, sample_func, [])

        # uniform_ex = numpy.all(p_ex == p_ex[0, 0])
        # uniform_sted = numpy.all(p_sted == p_sted[0, 0])
        # uniform_pdt = numpy.all(pdt == pdt[0, 0])
        # is_uniform = uniform_sted and uniform_ex and uniform_pdt
        # if is_uniform:
        #     effectives = []
        #     for i in range(len(self.opts["scale_power"])):
        #         effective = self.get_effective(datamap.pixelsize, p_ex[0, 0], self.opts["scale_power"][i] * p_sted[0, 0])
        #         effectives.append(effective)
        #
        #     photons_ex = self.fluo.get_photons(i_ex * p_ex[0, 0], self.excitation.lambda_)
        #     duty_cycle = self.sted.tau * self.sted.rate
        #     photons_sted = self.fluo.get_photons(i_sted * p_sted[0, 0] * duty_cycle, self.sted.lambda_)
        #
        #     k_steds, k_exs = [], []
        #     for i, (scale_power, decision_time) in enumerate(zip(self.opts["scale_power"], self.opts["decision_time"])):
        #         if decision_time < 0.:
        #             decision_time = pdt[0, 0]
        #
        #         photons_ex = self.fluo.get_photons(i_ex * p_ex[0, 0], self.excitation.lambda_)
        #         photons_sted = self.fluo.get_photons(i_sted * scale_power * p_sted[0, 0] * duty_cycle, self.sted.lambda_)
        #
        #         k_steds.append(self.fluo.get_k_bleach(self.excitation.lambda_, self.sted.lambda_, photons_ex, photons_sted, self.sted.tau, 1/self.sted.rate, decision_time, ))
        #         k_exs.append(k_steds[-1] * 0.)
        # else:
        #     k_sted, k_ex = None, None
        #
        # for (row, col) in pixel_list:
        #     row_slice = slice(row + rows_pad - laser_pad, row + rows_pad + laser_pad + 1)
        #     col_slice = slice(col + cols_pad - laser_pad, col + cols_pad + laser_pad + 1)
        #
        #     pdts, p_exs, p_steds = numpy.zeros(len(self.opts["scale_power"])), numpy.zeros(len(self.opts["scale_power"])), numpy.zeros(len(self.opts["scale_power"]))
        #     for i, (scale_power, decision_time, threshold_count) in enumerate(zip(self.opts["scale_power"], self.opts["decision_time"], self.opts["threshold_count"])):
        #         if decision_time < 0.:
        #             decision_time = pdt[row, col]
        #
        #         if not is_uniform:
        #             effective = self.get_effective(datamap_pixelsize, p_ex[row, col], scale_power * p_sted[row, col])
        #         else:
        #             effective = effectives[i]
        #
        #         h, w = effective.shape
        #
        #         # Uses the bleached datamap
        #         bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int32)
        #         for key in bleached_sub_datamaps_dict:
        #             bleached_datamap += bleached_sub_datamaps_dict[key]
        #
        #         pixel_intensity = numpy.sum(effective * bleached_datamap[row_slice, col_slice])
        #         pixel_photons = self.detector.get_signal(self.fluo.get_photons(pixel_intensity), decision_time, self.sted.rate)
        #
        #         # Stores the action taken for futures bleaching
        #         pdts[i] = decision_time
        #         p_exs[i] = p_ex[row, col]
        #         p_steds[i] = scale_power * p_sted[row, col]
        #
        #         # If signal is less than threshold count then skip pixel
        #         scaled_power[row, col] = scale_power
        #         if pixel_photons < threshold_count:
        #             break
        #
        #         # Update the photon counts only on the last pixel power scale
        #         if scale_power > 0.:
        #             returned_photons[row, col] += pixel_photons
        #
        #     # We add row_slice.start and col_slice.start to recenter the slice
        #     mask = (numpy.argwhere(bleached_datamap[row_slice, col_slice]) + numpy.array([[row_slice.start, col_slice.start]])).tolist()
        #     if bleach and (len(mask) > 0):
        #         for _p_ex, _p_sted, _pdt in zip(p_exs, p_steds, pdts):
        #             if _pdt > 0:
        #                 if is_uniform:
        #                     k_sted, k_ex = k_steds[i], k_exs[i]
        #                 bleach_func(self, i_ex, i_sted, _p_ex, _p_sted,
        #                             _pdt, bleached_sub_datamaps_dict,
        #                             row, col, h, w, mask, prob_ex, prob_sted, k_ex, k_sted)
        #         sample_func(self, bleached_sub_datamaps_dict, row, col, h, w, mask, prob_ex, prob_sted)

        if update and bleach:
            datamap.sub_datamaps_dict = bleached_sub_datamaps_dict
            datamap.base_datamap = datamap.sub_datamaps_dict["base"]
            datamap.whole_datamap = numpy.copy(datamap.base_datamap)

        return returned_photons, bleached_sub_datamaps_dict, scaled_power

class DyMINRESCueMicroscope(base.Microscope):
    def __init__(self, excitation, sted, detector, objective, fluo, load_cache=False, opts=None):
        super(DyMINRESCueMicroscope, self).__init__(excitation, sted, detector, objective, fluo, load_cache=load_cache)

        if isinstance(opts, type(None)):
            opts = {
                "scale_power" : [0., 0.1, 1.],
                "decision_time" : [10.0e-6, 10.0e-6, 10.0e-6],
                "threshold_count" : [8, 8, 3]
            }
        required_keys = ["scale_power", "decision_time", "threshold_count"]
        assert all(k in opts for k in required_keys), "Missing keys in opts. {}".format(required_keys)
        self.opts = opts

    def get_signal_and_bleach(self, datamap, pixelsize, pdt, p_ex, p_sted, indices=None, acquired_intensity=None,
                                  pixel_list=None, bleach=True, update=True, seed=None, filter_bypass=False,
                                  bleach_func=bleach_funcs.default_update_survival_probabilities,
                                  sample_func=bleach_funcs.sample_molecules):

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
            bleached_sub_datamaps_dict[key] = numpy.copy(datamap.sub_datamaps_dict[key].astype(numpy.int64))

        uniform_ex = numpy.all(p_ex == p_ex[0, 0])
        uniform_sted = numpy.all(p_sted == p_sted[0, 0])
        uniform_pdt = numpy.all(pdt == pdt[0, 0])
        is_uniform = uniform_sted and uniform_ex and uniform_pdt
        if is_uniform:
            effectives = []
            for i in range(len(self.opts["scale_power"])):
                effective = self.get_effective(datamap.pixelsize, p_ex[0, 0], self.opts["scale_power"][i] * p_sted[0, 0])
                effectives.append(effective)

        for (row, col) in pixel_list:
            row_slice = slice(row + rows_pad - laser_pad, row + rows_pad + laser_pad + 1)
            col_slice = slice(col + cols_pad - laser_pad, col + cols_pad + laser_pad + 1)

            pdts, p_exs, p_steds = numpy.zeros(len(self.opts["scale_power"])), numpy.zeros(len(self.opts["scale_power"])), numpy.zeros(len(self.opts["scale_power"]))
            for i, (scale_power, decision_time, threshold_count) in enumerate(zip(self.opts["scale_power"], self.opts["decision_time"], self.opts["threshold_count"])):

                if not is_uniform:
                    effective = self.get_effective(datamap_pixelsize, p_ex[row, col], scale_power * p_sted[row, col])
                else:
                    effective = effectives[i]
                h, w = effective.shape

                # Uses the bleached datamap
                bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int32)
                for key in bleached_sub_datamaps_dict:
                    bleached_datamap += bleached_sub_datamaps_dict[key]

                pixel_intensity = numpy.sum(effective * bleached_datamap[row_slice, col_slice])
                pixel_photons = self.detector.get_signal(self.fluo.get_photons(pixel_intensity), decision_time, self.sted.rate)

                # Stores the action taken for futures bleaching
                pdts[i] = decision_time
                p_exs[i] = p_ex[row, col]
                p_steds[i] = scale_power * p_sted[row, col]
                if i == len(self.opts["scale_power"]) - 1:
                    if (threshold_count > 0) and (pixel_photons > threshold_count):
                        scaled_power[row, col] = 4
                        returned_photons[row, col] += (pixel_photons * pdt[row, col] / decision_time).astype(int)
                        break
                    else:
                        scaled_power[row, col] = 3
                        returned_photons[row, col] += pixel_photons

                        # Acquire another time as in RESCue
                        pdts[i] += pdt[row, col]
                        pixel_photons = self.detector.get_signal(self.fluo.get_photons(pixel_intensity), pdt[row, col], self.sted.rate)
                        returned_photons[row, col] += pixel_photons
                else:
                    # If signal is less than threshold count then skip pixel
                    scaled_power[row, col] = scale_power
                    if pixel_photons < threshold_count:
                        break

                    # # Update the photon counts only on the last pixel power scale
                    # if scale_power > 0.:
                    #     returned_photons[row, col] += pixel_photons

            # We add row_slice.start and col_slice.start to recenter the slice
            mask = (numpy.argwhere(bleached_datamap[row_slice, col_slice] > 0) + numpy.array([[row_slice.start, col_slice.start]])).tolist()
            if bleach and (len(mask) > 0):
                for _p_ex, _p_sted, _pdt in zip(p_exs, p_steds, pdts):
                    if _pdt > 0:
                        bleach_func(self, i_ex, i_sted, _p_ex, _p_sted,
                                    _pdt, bleached_sub_datamaps_dict,
                                    row, col, h, w, mask, prob_ex, prob_sted, None, None)
                sample_func(self, bleached_sub_datamaps_dict, row, col, h, w, mask, prob_ex, prob_sted)

        if update and bleach:
            datamap.sub_datamaps_dict = bleached_sub_datamaps_dict
            datamap.base_datamap = datamap.sub_datamaps_dict["base"]
            datamap.whole_datamap = numpy.copy(datamap.base_datamap)

        return returned_photons, bleached_sub_datamaps_dict, scaled_power

class RESCueMicroscope(base.Microscope):
    def __init__(self, excitation, sted, detector, objective, fluo, load_cache=False, opts=None):
        super(RESCueMicroscope, self).__init__(excitation, sted, detector, objective, fluo, load_cache=load_cache)

        if isinstance(opts, type(None)):
            opts = {
                "lower_threshold" : [2, -1.],
                "upper_threshold" : [4, -1.],
                "decision_time" : [10.0e-6, -1.]
            }
        required_keys = ["lower_threshold", "upper_threshold", "decision_time"]
        assert all(k in opts for k in required_keys), "Missing keys in opts. {}".format(required_keys)
        self.opts = opts

    def get_signal_and_bleach(self, datamap, pixelsize, pdt, p_ex, p_sted, indices=None, acquired_intensity=None,
                                  pixel_list=None, bleach=True, update=True, seed=None, filter_bypass=False,
                                  bleach_func=bleach_funcs.default_update_survival_probabilities,
                                  sample_func=bleach_funcs.sample_molecules):

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
        thresholds = numpy.zeros(datamap.whole_datamap[datamap.roi].shape)
        acquired_intensity = numpy.zeros(datamap.whole_datamap[datamap.roi].shape)

        bleached_sub_datamaps_dict = {}
        if isinstance(indices, type(None)):
            indices = 0   # VÉRIF À QUOI INDICES SERT?
        for key in datamap.sub_datamaps_dict:
            bleached_sub_datamaps_dict[key] = numpy.copy(datamap.sub_datamaps_dict[key].astype(numpy.int64))

        uniform_ex = numpy.all(p_ex == p_ex[0, 0])
        uniform_sted = numpy.all(p_sted == p_sted[0, 0])
        uniform_pdt = numpy.all(pdt == pdt[0, 0])
        is_uniform = uniform_sted and uniform_ex and uniform_pdt
        if is_uniform:
            effectives = []
            for i in range(len(self.opts["decision_time"])):
                effective = self.get_effective(datamap.pixelsize, p_ex[0, 0], p_sted[0, 0])
                effectives.append(effective)

        if isinstance(seed, type(None)):
            seed = 0
        for key, value in self.opts.items():
            if isinstance(value, list):
                self.opts[key] = numpy.array(value)
            self.opts[key] = self.opts[key].astype(float)
        raster_func = raster.raster_func_rescue
        raster_func(self, datamap, acquired_intensity, numpy.array(pixel_list).astype(numpy.int32), ratio, rows_pad,
                    cols_pad, laser_pad, prob_ex, prob_sted, returned_photons, thresholds, pdt, p_ex, p_sted,
                    bleach, bleached_sub_datamaps_dict, seed, bleach_func, sample_func, [])

        # for (row, col) in pixel_list:
        #     row_slice = slice(row + rows_pad - laser_pad, row + rows_pad + laser_pad + 1)
        #     col_slice = slice(col + cols_pad - laser_pad, col + cols_pad + laser_pad + 1)
        #
        #     pdts, p_exs, p_steds = numpy.zeros(len(self.opts["decision_time"])), numpy.zeros(len(self.opts["decision_time"])), numpy.zeros(len(self.opts["decision_time"]))
        #     for i, (lower_threshold, upper_threshold, decision_time) in enumerate(zip(self.opts["lower_threshold"], self.opts["upper_threshold"], self.opts["decision_time"])):
        #         # If last decision, we aquire for the remaining time period
        #         if decision_time < 0.:
        #             decision_time = pdt[row, col]# - numpy.array(self.opts["decision_time"])[:-1].sum()
        #
        #         if not is_uniform:
        #             effective = self.get_effective(datamap_pixelsize, p_ex[row, col], scale_power * p_sted[row, col])
        #         else:
        #             effective = effectives[i]
        #         h, w = effective.shape
        #
        #         # Uses the bleached datamap
        #         bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int32)
        #         for key in bleached_sub_datamaps_dict:
        #             bleached_datamap += bleached_sub_datamaps_dict[key]
        #
        #         # Calculates the number of acquired photons
        #         pixel_intensity = numpy.sum(effective * bleached_datamap[row_slice, col_slice])
        #         pixel_photons = self.detector.get_signal(self.fluo.get_photons(pixel_intensity), decision_time, self.sted.rate)
        #
        #         # Stores the action taken for futures bleaching
        #         pdts[i] = decision_time
        #         p_exs[i] = p_ex[row, col]
        #         p_steds[i] = p_sted[row, col]
        #
        #         # STEPS
        #         # if number of photons is less than lower_threshold
        #         # we skip
        #         # if number of photons is higher than upper_threshold
        #         # we stop acquisition and assign number of count as total_time/decision_time
        #         # if number of photons is between
        #         # We continue to the next step
        #         # At the final step we assign the number of acquired photons
        #
        #         if (lower_threshold > 0) and (pixel_photons < lower_threshold):
        #             thresholds[row, col] = 0
        #             # returned_photons[row, col] += (pixel_photons * pdt[row, col] / decision_time).astype(int)
        #             break
        #         elif (upper_threshold > 0) and (pixel_photons > upper_threshold):
        #             thresholds[row, col] = 2
        #             returned_photons[row, col] += (pixel_photons * pdt[row, col] / decision_time).astype(int)
        #             break
        #         else:
        #             thresholds[row, col] = 1
        #             returned_photons[row, col] += pixel_photons
        #
        #     # We add row_slice.start and col_slice.start to recenter the slice
        #     mask = (numpy.argwhere(bleached_datamap[row_slice, col_slice] > 0) + numpy.array([[row_slice.start, col_slice.start]])).tolist()
        #     if bleach and (len(mask) > 0):
        #         for _p_ex, _p_sted, _pdt in zip(p_exs, p_steds, pdts):
        #             if _pdt > 0:
        #                 bleach_func(self, i_ex, i_sted, _p_ex, _p_sted,
        #                             _pdt, bleached_sub_datamaps_dict,
        #                             row, col, h, w, mask, prob_ex, prob_sted, None, None)
        #         sample_func(self, bleached_sub_datamaps_dict, row, col, h, w, mask, prob_ex, prob_sted)

        if update and bleach:
            datamap.sub_datamaps_dict = bleached_sub_datamaps_dict
            datamap.base_datamap = datamap.sub_datamaps_dict["base"]
            datamap.whole_datamap = numpy.copy(datamap.base_datamap)

        return returned_photons, bleached_sub_datamaps_dict, thresholds
