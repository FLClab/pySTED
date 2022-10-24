
import time
import numpy
from matplotlib import pyplot as plt
import bleach_funcs
cimport numpy
import scipy
cimport cython

from libc.math cimport exp
from libc.stdlib cimport rand, srand, RAND_MAX

INTDTYPE = numpy.int32
INT64DTYPE = numpy.int64
FLOATDTYPE = numpy.float64

ctypedef numpy.int32_t INTDTYPE_t
ctypedef numpy.int64_t INT64DTYPE_t
ctypedef numpy.float64_t FLOATDTYPE_t

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def reset_prob(
    list mask,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
    numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted
):
    cdef int s, t
    for (s, t) in mask:
        prob_ex[s, t] = 1.0
        prob_sted[s, t] = 1.0


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def raster_func_c_self_bleach_split_g(
        object self,
        object datamap,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] acquired_intensity,
        numpy.ndarray[INTDTYPE_t, ndim=2] pixel_list,
        int ratio,
        int rows_pad,
        int cols_pad,
        int laser_pad,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] pdt_roi,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] p_ex_roi,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] p_sted_roi,
        bint bleach,   # bint is a bool
        dict bleached_sub_datamaps_dict,
        int seed,
        object bleach_func,   # uncertain of the type for a cfunc, but this seems to be working so ???
        object sample_func,
        list steps
):
    cdef int row, col
    cdef int sprime, tprime
    cdef int h, w
    cdef int current
    cdef int max_len = len(pixel_list)
    cdef FLOATDTYPE_t value
    cdef int sampled_value
    cdef int prob
    cdef int rsamp
    cdef FLOATDTYPE_t pdt, p_ex, p_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] pre_effective, effective
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex, k_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex, i_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
    cdef numpy.ndarray[INT64DTYPE_t, ndim=2] bleached_datamap
    cdef numpy.ndarray[INT64DTYPE_t, ndim=2] current_datamap
    cdef FLOATDTYPE_t duty_cycle
    cdef FLOATDTYPE_t step
    cdef list mask
    cdef bint uniform_sted, uniform_ex, uniform_pdt, is_uniform

    """
    raster_func_c_self_bleach executes the simultaneous acquisition and bleaching routine for the case where the
    excitation power (p_ex) AND/OR sted power (p_sted) vary through the sample. This function thus requires these
    parameters to be passed as arrays of floats the same size as the ROI being imaged.

    Additionally, this function seperately bleaches the different parts composing the datamap (i.e. the base and flash
    components of the datamap are bleached separately).
    """

    if seed == 0:
        # if no seed is passed, calculates a 'pseudo-random' seed form the time in ns
        srand(int(str(time.time_ns())[-5:-1]))
    else:
        srand(seed)

    i_ex, i_sted, _ = self.cache(datamap.pixelsize)

    # Calculate the bleaching rate once if the scanning powers and dwelltimes do not vary to
    # increase speed
    uniform_ex = numpy.all(p_ex_roi == p_ex_roi[0, 0])
    uniform_sted = numpy.all(p_sted_roi == p_sted_roi[0, 0])
    uniform_pdt = numpy.all(pdt_roi == pdt_roi[0, 0])
    if uniform_sted and uniform_ex and uniform_pdt:
        p_ex = p_ex_roi[0, 0]
        p_sted = p_sted_roi[0, 0]
        pdt = pdt_roi[0, 0]
        photons_ex = self.fluo.get_photons(i_ex * p_ex, self.excitation.lambda_)
        duty_cycle = self.sted.tau * self.sted.rate
        photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle, self.sted.lambda_)
        k_sted = self.fluo.get_k_bleach(self.excitation.lambda_, self.sted.lambda_, photons_ex, photons_sted, self.sted.tau, 1/self.sted.rate, pdt,)
        k_ex = k_sted * 0.
    else:
        k_sted = None
        k_ex = None

        effective = self.get_effective(datamap.pixelsize, p_ex, p_sted)

    pre_effective = self.get_effective(datamap.pixelsize, p_ex_roi[0, 0], p_sted_roi[0, 0])
    h, w = pre_effective.shape[0], pre_effective.shape[1]

    is_uniform = uniform_sted and uniform_ex and uniform_pdt
    effective = self.get_effective(datamap.pixelsize, p_ex, p_sted)

    for (row, col) in pixel_list:
        if not is_uniform:
            pdt = pdt_roi[row, col]
            p_ex = p_ex_roi[row, col]
            p_sted = p_sted_roi[row, col]
            effective = self.get_effective(datamap.pixelsize, p_ex, p_sted)

        mask = []

        # Combines the datamaps into a single datamap
        bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int64)
        for key in bleached_sub_datamaps_dict:
            current_datamap = bleached_sub_datamaps_dict[key]
            for s in range(row, row + h):
                for t in range(col, col + w):
                    bleached_datamap[s, t] += current_datamap[s, t]

        # Calculates the acquired intensity and keeps track of the position
        # of the emitters
        value = 0.0
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                if bleach and (bleached_datamap[s, t] > 0):
                    mask.append((s, t))
                value += effective[sprime, tprime] * bleached_datamap[s, t]
                tprime += 1
            sprime += 1
        acquired_intensity[int(row / ratio), int(col / ratio)] += value

        # Bleaches the sample
        if bleach:
            bleach_func(self, i_ex, i_sted, p_ex, p_sted, pdt, bleached_sub_datamaps_dict, row, col, h, w, mask, prob_ex,
                        prob_sted, k_ex, k_sted)
            sample_func(self, bleached_sub_datamaps_dict, row, col, h, w, mask, prob_ex, prob_sted)

            # We reset the survival probabilty
            prob_ex = numpy.ones_like(prob_ex)
            prob_sted = numpy.ones_like(prob_sted)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def raster_func_dymin(
        object self,
        object datamap,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] acquired_intensity,
        numpy.ndarray[INTDTYPE_t, ndim=2] pixel_list,
        int ratio,
        int rows_pad,
        int cols_pad,
        int laser_pad,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] returned_photons,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] scaled_power,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] pdt_roi,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] p_ex_roi,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] p_sted_roi,
        bint bleach,   # bint is a bool
        dict bleached_sub_datamaps_dict,
        int seed,
        object bleach_func,   # uncertain of the type for a cfunc, but this seems to be working so ???
        object sample_func,
        list steps
):
    cdef int row, col, i
    cdef int sprime, tprime
    cdef int h, w
    cdef int current
    cdef int max_len = len(pixel_list)
    cdef FLOATDTYPE_t value
    cdef int sampled_value
    cdef int prob
    cdef int rsamp
    cdef FLOATDTYPE_t pdt, p_ex, p_sted, _pdt, _p_ex, _p_sted
    cdef float scale_power, decision_time, threshold_count
    cdef int pixel_photons
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] pre_effective, effective
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=3] effectives, k_exs, k_steds
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex, k_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex, i_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=1] pdts, p_exs, p_steds
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=1] SCALE_POWER, DECISION_TIME, THRESHOLD_COUNT
    cdef numpy.ndarray[INT64DTYPE_t, ndim=2] bleached_datamap, current_datamap
    cdef FLOATDTYPE_t duty_cycle
    cdef FLOATDTYPE_t step
    cdef list mask
    cdef int num_steps
    cdef bint uniform_sted, uniform_ex, uniform_pdt, is_uniform

    if seed == 0:
        # if no seed is passed, calculates a 'pseudo-random' seed form the time in ns
        srand(int(str(time.time_ns())[-5:-1]))
    else:
        srand(seed)

    i_ex, i_sted, _ = self.cache(datamap.pixelsize)
    # Calculate the bleaching rate once if the scanning powers and dwelltimes do not vary to
    # increase speed
    uniform_ex = numpy.all(p_ex_roi == p_ex_roi[0, 0])
    uniform_sted = numpy.all(p_sted_roi == p_sted_roi[0, 0])
    uniform_pdt = numpy.all(pdt_roi == pdt_roi[0, 0])
    if uniform_sted and uniform_ex and uniform_pdt:
        p_ex = p_ex_roi[0, 0]
        p_sted = p_sted_roi[0, 0]
        pdt = pdt_roi[0, 0]
        photons_ex = self.fluo.get_photons(i_ex * p_ex, self.excitation.lambda_)
        duty_cycle = self.sted.tau * self.sted.rate
        photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle, self.sted.lambda_)
        k_sted = self.fluo.get_k_bleach(self.excitation.lambda_, self.sted.lambda_, photons_ex, photons_sted, self.sted.tau, 1/self.sted.rate, pdt,)
        k_ex = k_sted * 0.
    else:
        k_sted = None
        k_ex = None

    pre_effective = self.get_effective(datamap.pixelsize, p_ex_roi[0, 0], p_sted_roi[0, 0])
    h, w = pre_effective.shape[0], pre_effective.shape[1]

    # Extracts DyMIN parameters
    SCALE_POWER = self.opts["scale_power"]
    DECISION_TIME = self.opts["decision_time"]
    THRESHOLD_COUNT = self.opts["threshold_count"]
    num_steps = len(self.opts["scale_power"])

    is_uniform = uniform_sted and uniform_ex and uniform_pdt
    if is_uniform:
        # Pre-calculates necessary variables
        effectives = numpy.zeros((num_steps, h, w), dtype=numpy.float64)
        k_steds = numpy.zeros((num_steps, k_sted.shape[0], k_sted.shape[1]), dtype=numpy.float64)
        k_exs = numpy.zeros((num_steps, k_ex.shape[0], k_ex.shape[1]), dtype=numpy.float64)

        for i in range(num_steps):
            effective = self.get_effective(datamap.pixelsize, p_ex, SCALE_POWER[i] * p_sted)
            effectives[i] = effective

            decision_time = DECISION_TIME[i]
            if decision_time < 0.:
                decision_time = pdt_roi[0, 0]

            # Must be recalculated
            photons_ex = self.fluo.get_photons(i_ex * p_ex, self.excitation.lambda_)
            photons_sted = self.fluo.get_photons(i_sted * SCALE_POWER[i] * p_sted * duty_cycle, self.sted.lambda_)

            k_steds[i] = self.fluo.get_k_bleach(self.excitation.lambda_, self.sted.lambda_, photons_ex, photons_sted, self.sted.tau, 1/self.sted.rate, decision_time, )
            k_exs[i] = k_steds[i] * 0.

    pdts, p_exs, p_steds = numpy.zeros(num_steps, dtype=numpy.float64), numpy.zeros(num_steps, dtype=numpy.float64), numpy.zeros(num_steps, dtype=numpy.float64)

    for (row, col) in pixel_list:
        pdts = pdts * 0.
        p_exs = p_exs * 0.
        p_steds = p_steds * 0.
        mask = []

        # Uses the bleached datamap
        bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int64)
        for key in bleached_sub_datamaps_dict:
            current_datamap = bleached_sub_datamaps_dict[key]
            for s in range(row, row + h):
                for t in range(col, col + w):
                    bleached_datamap[s, t] += current_datamap[s, t]

        # Creates the masked values
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                if bleach and (bleached_datamap[s, t] > 0):
                    mask.append((s, t))
                tprime += 1
            sprime += 1

        # DyMIN implementation for every step
        for i in range(num_steps):

            scale_power = SCALE_POWER[i]
            decision_time = DECISION_TIME[i]
            threshold_count = THRESHOLD_COUNT[i]

            if decision_time < 0.:
                decision_time = pdt_roi[row, col]

            if not is_uniform:
                effective = self.get_effective(datamap.pixelsize, p_ex_roi[row, col], scale_power * p_sted_roi[row, col])
            else:
                effective = effectives[i]

            # pixel_intensity = numpy.sum(effective * bleached_datamap[row_slice, col_slice])
            value = 0.0
            for (s, t) in mask:
                sprime = s - row
                tprime = t - col
                value += effective[sprime, tprime] * bleached_datamap[s, t]

            pixel_photons = self.detector.get_signal(self.fluo.get_photons(value), decision_time, self.sted.rate)

            # Stores the action taken for futures bleaching
            pdts[i] = decision_time
            p_exs[i] = p_ex_roi[row, col]
            p_steds[i] = scale_power * p_sted_roi[row, col]

            # If signal is less than threshold count then skip pixel
            scaled_power[row, col] = scale_power
            if pixel_photons < threshold_count:
                break

            # Update the photon counts only on the last pixel power scale
            if i == num_steps - 1:
                returned_photons[row, col] += pixel_photons

        # Bleaches the sample after pixel is scanned
        if bleach:
            for i in range(num_steps):
                if pdts[i] > 0:
                    if is_uniform:
                        k_sted, k_ex = k_steds[i], k_exs[i]
                    bleach_func(self, i_ex, i_sted, p_exs[i], p_steds[i],
                                pdts[i], bleached_sub_datamaps_dict,
                                row, col, h, w, mask, prob_ex, prob_sted, k_ex, k_sted)
            sample_func(self, bleached_sub_datamaps_dict, row, col, h, w, mask, prob_ex, prob_sted)

            # We reset the survival probabilty
            reset_prob(mask, prob_ex, prob_sted)
            # prob_ex = numpy.ones_like(prob_ex)
            # prob_sted = numpy.ones_like(prob_sted)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def raster_func_rescue(
        object self,
        object datamap,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] acquired_intensity,
        numpy.ndarray[INTDTYPE_t, ndim=2] pixel_list,
        int ratio,
        int rows_pad,
        int cols_pad,
        int laser_pad,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] returned_photons,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] thresholds,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] pdt_roi,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] p_ex_roi,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] p_sted_roi,
        bint bleach,   # bint is a bool
        dict bleached_sub_datamaps_dict,
        int seed,
        object bleach_func,   # uncertain of the type for a cfunc, but this seems to be working so ???
        object sample_func,
        list steps
):
    cdef int row, col, i
    cdef int sprime, tprime
    cdef int h, w
    cdef int current
    cdef int max_len = len(pixel_list)
    cdef FLOATDTYPE_t value
    cdef int sampled_value
    cdef int prob
    cdef int rsamp
    cdef FLOATDTYPE_t pdt, p_ex, p_sted, _pdt, _p_ex, _p_sted
    cdef float scale_power, decision_time, threshold_count
    cdef int pixel_photons
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] pre_effective, effective
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=3] effectives, k_exs, k_steds
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex, k_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex, i_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=1] pdts, p_exs, p_steds
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=1] LOWER_THRESHOLD, UPPER_THRESHOLD, DECISION_TIME
    cdef numpy.ndarray[INT64DTYPE_t, ndim=2] bleached_datamap, current_datamap
    cdef FLOATDTYPE_t duty_cycle
    cdef FLOATDTYPE_t step
    cdef list mask
    cdef int num_steps
    cdef bint uniform_sted, uniform_ex, uniform_pdt, is_uniform

    if seed == 0:
        # if no seed is passed, calculates a 'pseudo-random' seed form the time in ns
        srand(int(str(time.time_ns())[-5:-1]))
    else:
        srand(seed)

    i_ex, i_sted, _ = self.cache(datamap.pixelsize)
    # Calculate the bleaching rate once if the scanning powers and dwelltimes do not vary to
    # increase speed
    uniform_ex = numpy.all(p_ex_roi == p_ex_roi[0, 0])
    uniform_sted = numpy.all(p_sted_roi == p_sted_roi[0, 0])
    uniform_pdt = numpy.all(pdt_roi == pdt_roi[0, 0])
    if uniform_sted and uniform_ex and uniform_pdt:
        p_ex = p_ex_roi[0, 0]
        p_sted = p_sted_roi[0, 0]
        pdt = pdt_roi[0, 0]
        photons_ex = self.fluo.get_photons(i_ex * p_ex, self.excitation.lambda_)
        duty_cycle = self.sted.tau * self.sted.rate
        photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle, self.sted.lambda_)
        k_sted = self.fluo.get_k_bleach(self.excitation.lambda_, self.sted.lambda_, photons_ex, photons_sted, self.sted.tau, 1/self.sted.rate, pdt,)
        k_ex = k_sted * 0.
    else:
        k_sted = None
        k_ex = None

    pre_effective = self.get_effective(datamap.pixelsize, p_ex_roi[0, 0], p_sted_roi[0, 0])
    h, w = pre_effective.shape[0], pre_effective.shape[1]

    # Extracts RESCue parameters
    LOWER_THRESHOLD = self.opts["lower_threshold"]
    UPPER_THRESHOLD = self.opts["upper_threshold"]
    DECISION_TIME = self.opts["decision_time"]
    num_steps = len(self.opts["decision_time"])

    is_uniform = uniform_sted and uniform_ex and uniform_pdt
    if is_uniform:
        # Pre-calculates necessary variables
        effectives = numpy.zeros((num_steps, h, w), dtype=numpy.float64)
        k_steds = numpy.zeros((num_steps, k_sted.shape[0], k_sted.shape[1]), dtype=numpy.float64)
        k_exs = numpy.zeros((num_steps, k_ex.shape[0], k_ex.shape[1]), dtype=numpy.float64)

        for i in range(num_steps):
            effective = self.get_effective(datamap.pixelsize, p_ex, p_sted)
            effectives[i] = effective

            decision_time = DECISION_TIME[i]
            if decision_time < 0.:
                decision_time = pdt_roi[0, 0]
            k_steds[i] = self.fluo.get_k_bleach(self.excitation.lambda_, self.sted.lambda_, photons_ex, photons_sted, self.sted.tau, 1/self.sted.rate, decision_time, )
            k_exs[i] = k_steds[i] * 0.

    pdts, p_exs, p_steds = numpy.zeros(num_steps, dtype=numpy.float64), numpy.zeros(num_steps, dtype=numpy.float64), numpy.zeros(num_steps, dtype=numpy.float64)
    for (row, col) in pixel_list:
        pdts = pdts * 0.
        p_exs = p_exs * 0.
        p_steds = p_steds * 0.
        mask = []

        # Uses the bleached datamap
        bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int64)
        for key in bleached_sub_datamaps_dict:
            current_datamap = bleached_sub_datamaps_dict[key]
            for s in range(row, row + h):
                for t in range(col, col + w):
                    bleached_datamap[s, t] += current_datamap[s, t]

        # Creates the masked values
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                if bleach and (bleached_datamap[s, t] > 0):
                    mask.append((s, t))
                tprime += 1
            sprime += 1

        # RESCue steps
        for i in range(num_steps):

            # Extract parameters
            lower_threshold = LOWER_THRESHOLD[i]
            upper_threshold = UPPER_THRESHOLD[i]
            decision_time = DECISION_TIME[i]

            if decision_time < 0.:
                decision_time = pdt_roi[row, col]

            # If non uniform laser powers we recalculate
            if not is_uniform:
                effective = self.get_effective(datamap.pixelsize, p_ex_roi[row, col], p_sted_roi[row, col])
            else:
                effective = effectives[i]

            # Convolve the effective and the datamap
            value = 0.0
            for (s, t) in mask:
                sprime = s - row
                tprime = t - col
                value += effective[sprime, tprime] * bleached_datamap[s, t]

            pixel_photons = self.detector.get_signal(self.fluo.get_photons(value), decision_time, self.sted.rate)

            # Stores the action taken for futures bleaching
            pdts[i] = decision_time
            p_exs[i] = p_ex_roi[row, col]
            p_steds[i] = scale_power * p_sted_roi[row, col]

            # STEPS
            # if number of photons is less than lower_threshold
            # we skip
            # if number of photons is higher than upper_threshold
            # we stop acquisition and assign number of count as total_time/decision_time
            # if number of photons is between
            # We continue to the next step
            # At the final step we assign the number of acquired photons

            if (lower_threshold > 0) and (pixel_photons < lower_threshold):
                thresholds[row, col] = 0
                # returned_photons[row, col] += (pixel_photons * pdt[row, col] / decision_time).astype(int)
                break
            elif (upper_threshold > 0) and (pixel_photons > upper_threshold):
                thresholds[row, col] = 2
                returned_photons[row, col] += pixel_photons * pdt_roi[row, col] / decision_time
                break
            else:
                thresholds[row, col] = 1
                returned_photons[row, col] += pixel_photons

        # Bleaches the sample once the pixel was acquired
        if bleach:
            for i in range(num_steps):
                if pdts[i] > 0:
                    if is_uniform:
                        k_sted, k_ex = k_steds[i], k_exs[i]
                    bleach_func(self, i_ex, i_sted, p_exs[i], p_steds[i],
                                pdts[i], bleached_sub_datamaps_dict,
                                row, col, h, w, mask, prob_ex, prob_sted, k_ex, k_sted)
            sample_func(self, bleached_sub_datamaps_dict, row, col, h, w, mask, prob_ex, prob_sted)

            reset_prob(mask, prob_ex, prob_sted)
            # prob_ex = numpy.ones_like(prob_ex)
            # prob_sted = numpy.ones_like(prob_sted)
