
'''This modules implements different configuration of a ``base.Microscope``.

Currently implemented microscopes are 

* ``DyMINMicroscope``
* ``DyMINRESCueMicroscope``
* ``RESCueMicroscope``

Implementing their own microscope requires to reimplement the ``get_signal_and_bleach`` 
method of a ``base.Microscope``.

.. rubric:: References

.. [Heine2017] Heine, J. et al. Adaptive-illumination STED nanoscopy. PNAS 114, 9797–9802 (2017).

.. [Staudt2011] Staudt, T. et al. Far-field optical nanoscopy with reduced number of state transition cycles. Opt. Express, OE 19, 5644–5657 (2011).

'''

import numpy
import time
import random

from pysted import base, utils, raster, bleach_funcs

class DyMINMicroscope(base.Microscope):
    '''Implements a ``DyMINMicroscope``.

    Refer to [Heine2017]_ for details about DyMIN microscopy.

    The DyMIN acquisition parameters are controlled with the `opts` variable. Other number 
    of DyMIN steps can be implemented by simply changing the length of each parameters.

    .. code-block:: python

        opts = {
            "scale_power" : [0, 0.25, 1.0], # Percentage of STED power 
            "decision_time" : [10e-6, 10e-6, -1], # Time to acquire photons
            "threshold_count" : [8, 8, 0] # Minimal number of photons for next step
        }
    '''
    def __init__(self, excitation, sted, detector, objective, fluo, load_cache=False, opts=None, verbose=False):
        """
        Instantiates the ``DyMINMicroscope``

        :param excitation: A :class:`~pysted.base.GaussianBeam` object
                        representing the excitation laser beam.
        :param sted: A :class:`~pysted.base.DonutBeam` object representing the
                    STED laser beam.
        :param detector: A :class:`~pysted.base.Detector` object describing the
                        microscope detector.
        :param objective: A :class:`~pysted.base.Objective` object describing the
                        microscope objective.
        :param fluo: A :class:`~pysted.base.Fluorescence` object describing the
                    fluorescence molecules to be used.
        :param load_cache: A bool which determines whether or not the microscope's lasers will be generated from scratch
                        (load_cache=False) or if they will be loaded from the previous save (load_cache=True). Generating
                        the lasers from scratch can take a long time (takes longer as the pixel_size decreases), so
                        loading the cache can save time when doing multiple experiments using the same pixel_size.    
        """
        super(DyMINMicroscope, self).__init__(excitation, sted, detector, objective, fluo, load_cache=load_cache, verbose=verbose)

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
        """
        This function acquires the signal and bleaches simultaneously.
        
        It makes a call to compiled C code for speed, so make sure the raster.pyx file is compiled!

        :param datamap: The datamap on which the acquisition is done, either a Datamap object or TemporalDatamap
        :param pixelsize: The pixelsize of the acquisition. (m)
        :param pdt: The pixel dwelltime. Can be either a single float value or an array of the same size as the ROI
                    being imaged. (s)
        :param p_ex: The excitation beam power. Can be either a single float value or an array of the same size as the
                     ROI being imaged. (W)
        :param p_sted: The depletion beam power. Can be either a single float value or an array of the same size as the
                       ROI being imaged. (W)
        :param indices: A dictionary containing the indices of the subdatamaps used. This is used to apply bleaching to
                        the future subdatamaps. If acquiring on a static Datamap, leave as None.
        :param acquired_intensity: The result of the last incomplete acquisition. This is useful in a time routine where
                                   flashes can occur mid acquisition. Leave as None if it is not the case. (array)
        :param pixel_list: The list of pixels to be iterated on. If none, a pixel_list of a raster scan will be
                           generated. (list of tuples (row, col))
        :param bleach: Determines whether bleaching is active or not. (Bool)
        :param update: Determines whether the datamap is updated in place. If set to false, the datamap can still be
                       updated later with the returned bleached datamap. (Bool)
        :param seed: Sets a seed for the random number generator.
        :param filter_bypass: Whether or not to filter the pixel list.
                              This is useful if you know your pixel list is adequate and ordered differently from a
                              raster scan (i.e. a left to right, row by row scan), as filtering the list return it
                              in raster order.
                              If pixel_list is none, this must be True then.
        :param bleach_func: The bleaching function to be applied.
        :param steps: list containing the pixeldwelltimes for the sub steps of an acquisition. Is none by default.
                      Should be used if trying to implement a DyMin type acquisition, where decisions are made
                      after some time on whether or not to continue the acq.

        :return: returned_acquired_photons, the acquired photon for the acquisition.
                 bleached_sub_datamaps_dict, a dict containing the results of bleaching on the subdatamaps
                 acquired_intensity, the intensity of the acquisition, used for interrupted acquisitions
        """
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
            indices = 0 
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

        if update and bleach:
            datamap.sub_datamaps_dict = bleached_sub_datamaps_dict
            datamap.base_datamap = datamap.sub_datamaps_dict["base"]
            datamap.whole_datamap = numpy.copy(datamap.base_datamap)

        return returned_photons, bleached_sub_datamaps_dict, scaled_power

class DyMINRESCueMicroscope(base.Microscope):
    """
    Implements a `DyMINRESCueMicroscope`.

    Refer to [Heine2017]_ for details about DyMIN microscopy. In this particular case, the 
    last step of the DyMIN acquisition is a RESCue acquisition.
    This microscope was not implemented in cython.

    The DyMINRESCUe acquisition parameters are controlled with the `opts` variable. Other number 
    of DyMIN steps can be implemented by simply changing the length of each parameters.

    .. code-block:: python

        opts = {
            "scale_power" : [0, 0.25, 1.0], # Percentage of STED power 
            "decision_time" : [10e-6, 10e-6, -1], # Time to acquire photons
            "threshold_count" : [8, 8, 0] # Minimal number of photons for next step
        }
    """    
    def __init__(self, excitation, sted, detector, objective, fluo, load_cache=False, opts=None, verbose=False):
        """
        Instantiates the `DyMINRESCueMicroscope`

        :param excitation: A :class:`~pysted.base.GaussianBeam` object
                        representing the excitation laser beam.
        :param sted: A :class:`~pysted.base.DonutBeam` object representing the
                    STED laser beam.
        :param detector: A :class:`~pysted.base.Detector` object describing the
                        microscope detector.
        :param objective: A :class:`~pysted.base.Objective` object describing the
                        microscope objective.
        :param fluo: A :class:`~pysted.base.Fluorescence` object describing the
                    fluorescence molecules to be used.
        :param load_cache: A bool which determines whether or not the microscope's lasers will be generated from scratch
                        (load_cache=False) or if they will be loaded from the previous save (load_cache=True). Generating
                        the lasers from scratch can take a long time (takes longer as the pixel_size decreases), so
                        loading the cache can save time when doing multiple experiments using the same pixel_size.    
        """        
        super(DyMINRESCueMicroscope, self).__init__(excitation, sted, detector, objective, fluo, load_cache=load_cache, verbose=verbose)

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
        """
        This function acquires the signal and bleaches simultaneously.

        :param datamap: The datamap on which the acquisition is done, either a Datamap object or TemporalDatamap
        :param pixelsize: The pixelsize of the acquisition. (m)
        :param pdt: The pixel dwelltime. Can be either a single float value or an array of the same size as the ROI
                    being imaged. (s)
        :param p_ex: The excitation beam power. Can be either a single float value or an array of the same size as the
                     ROI being imaged. (W)
        :param p_sted: The depletion beam power. Can be either a single float value or an array of the same size as the
                       ROI being imaged. (W)
        :param indices: A dictionary containing the indices of the subdatamaps used. This is used to apply bleaching to
                        the future subdatamaps. If acquiring on a static Datamap, leave as None.
        :param acquired_intensity: The result of the last incomplete acquisition. This is useful in a time routine where
                                   flashes can occur mid acquisition. Leave as None if it is not the case. (array)
        :param pixel_list: The list of pixels to be iterated on. If none, a pixel_list of a raster scan will be
                           generated. (list of tuples (row, col))
        :param bleach: Determines whether bleaching is active or not. (Bool)
        :param update: Determines whether the datamap is updated in place. If set to false, the datamap can still be
                       updated later with the returned bleached datamap. (Bool)
        :param seed: Sets a seed for the random number generator.
        :param filter_bypass: Whether or not to filter the pixel list.
                              This is useful if you know your pixel list is adequate and ordered differently from a
                              raster scan (i.e. a left to right, row by row scan), as filtering the list return it
                              in raster order.
                              If pixel_list is none, this must be True then.
        :param bleach_func: The bleaching function to be applied.
        :param steps: list containing the pixeldwelltimes for the sub steps of an acquisition. Is none by default.
                      Should be used if trying to implement a DyMin type acquisition, where decisions are made
                      after some time on whether or not to continue the acq.

        :return: returned_acquired_photons, the acquired photon for the acquisition.
                 bleached_sub_datamaps_dict, a dict containing the results of bleaching on the subdatamaps
                 acquired_intensity, the intensity of the acquisition, used for interrupted acquisitions
        """
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
            indices = 0
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
    """
    Implements a `RESCueMicroscope`.

    Refer to [Staudt2011]_ for details about RESCue microscopy.

    The RESCue acquisition parameters are controlled with the `opts` variable. Other number 
    of steps can be implemented by simply changing the length of each parameters.

    .. code-block:: python

        opts = {
            "lower_treshold" : [2, -1], # Lower threshold on the number of photons
            "upper_threshold" : [4, -1], # Upper threshold on the maximum number of photons
            "decision_time" : [10e-6, -1] # Time spent for the decision
        }
    """       
    def __init__(self, excitation, sted, detector, objective, fluo, load_cache=False, opts=None, verbose=False):
        """
        Instantiates the `RESCueMicroscope`

        :param excitation: A :class:`~pysted.base.GaussianBeam` object
                        representing the excitation laser beam.
        :param sted: A :class:`~pysted.base.DonutBeam` object representing the
                    STED laser beam.
        :param detector: A :class:`~pysted.base.Detector` object describing the
                        microscope detector.
        :param objective: A :class:`~pysted.base.Objective` object describing the
                        microscope objective.
        :param fluo: A :class:`~pysted.base.Fluorescence` object describing the
                    fluorescence molecules to be used.
        :param load_cache: A bool which determines whether or not the microscope's lasers will be generated from scratch
                        (load_cache=False) or if they will be loaded from the previous save (load_cache=True). Generating
                        the lasers from scratch can take a long time (takes longer as the pixel_size decreases), so
                        loading the cache can save time when doing multiple experiments using the same pixel_size.    
        """              
        super(RESCueMicroscope, self).__init__(excitation, sted, detector, objective, fluo, load_cache=load_cache, verbose=verbose)

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
        """
        This function acquires the signal and bleaches simultaneously. 
        
        It makes a call to compiled C code for speed, so make sure the raster.pyx file is compiled!

        :param datamap: The datamap on which the acquisition is done, either a Datamap object or TemporalDatamap
        :param pixelsize: The pixelsize of the acquisition. (m)
        :param pdt: The pixel dwelltime. Can be either a single float value or an array of the same size as the ROI
                    being imaged. (s)
        :param p_ex: The excitation beam power. Can be either a single float value or an array of the same size as the
                     ROI being imaged. (W)
        :param p_sted: The depletion beam power. Can be either a single float value or an array of the same size as the
                       ROI being imaged. (W)
        :param indices: A dictionary containing the indices of the subdatamaps used. This is used to apply bleaching to
                        the future subdatamaps. If acquiring on a static Datamap, leave as None.
        :param acquired_intensity: The result of the last incomplete acquisition. This is useful in a time routine where
                                   flashes can occur mid acquisition. Leave as None if it is not the case. (array)
        :param pixel_list: The list of pixels to be iterated on. If none, a pixel_list of a raster scan will be
                           generated. (list of tuples (row, col))
        :param bleach: Determines whether bleaching is active or not. (Bool)
        :param update: Determines whether the datamap is updated in place. If set to false, the datamap can still be
                       updated later with the returned bleached datamap. (Bool)
        :param seed: Sets a seed for the random number generator.
        :param filter_bypass: Whether or not to filter the pixel list.
                              This is useful if you know your pixel list is adequate and ordered differently from a
                              raster scan (i.e. a left to right, row by row scan), as filtering the list return it
                              in raster order.
                              If pixel_list is none, this must be True then.
        :param bleach_func: The bleaching function to be applied.
        :param steps: list containing the pixeldwelltimes for the sub steps of an acquisition. Is none by default.
                      Should be used if trying to implement a DyMin type acquisition, where decisions are made
                      after some time on whether or not to continue the acq.

        :return: returned_acquired_photons, the acquired photon for the acquisition.
                 bleached_sub_datamaps_dict, a dict containing the results of bleaching on the subdatamaps
                 acquired_intensity, the intensity of the acquisition, used for interrupted acquisitions
        """
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
            indices = 0
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

        if update and bleach:
            datamap.sub_datamaps_dict = bleached_sub_datamaps_dict
            datamap.base_datamap = datamap.sub_datamaps_dict["base"]
            datamap.whole_datamap = numpy.copy(datamap.base_datamap)

        return returned_photons, bleached_sub_datamaps_dict, thresholds
