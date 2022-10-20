
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
FLOATDTYPE = numpy.float64

ctypedef numpy.int32_t INTDTYPE_t
ctypedef numpy.float64_t FLOATDTYPE_t


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
#def raster_func_c_self_bleach_split_g(
#        object self,
#        object datamap,
#        numpy.ndarray[FLOATDTYPE_t, ndim=2] acquired_intensity,
#        numpy.ndarray[INTDTYPE_t, ndim=2] pixel_list,
#        int ratio,
#        int rows_pad,
#        int cols_pad,
#        int laser_pad,
#        numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_ex,
#        numpy.ndarray[FLOATDTYPE_t, ndim=2] prob_sted,
#        numpy.ndarray[FLOATDTYPE_t, ndim=2] pdt_roi,
#        numpy.ndarray[FLOATDTYPE_t, ndim=2] p_ex_roi,
#        numpy.ndarray[FLOATDTYPE_t, ndim=2] p_sted_roi,
#        bint bleach,   # bint is a bool
#        dict bleached_sub_datamaps_dict,
#        int seed,
#        object bleach_func,   # uncertain of the type for a cfunc, but this seems to be working so ???
#        object sample_func,
#        list steps
#):
#    cdef int row, col
#    cdef int sprime, tprime
#    cdef int h, w
#    cdef int current
#    cdef int max_len = len(pixel_list)
#    cdef FLOATDTYPE_t value
#    cdef int sampled_value
#    cdef int prob
#    cdef int rsamp
#    cdef FLOATDTYPE_t pdt, p_ex, p_sted
#    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] pre_effective, effective
#    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] k_ex, k_sted
#    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] i_ex, i_sted
#    cdef numpy.ndarray[FLOATDTYPE_t, ndim=2] photons_ex, photons_sted
#
#    cdef FLOATDTYPE_t duty_cycle
#    cdef FLOATDTYPE_t step
#
#    """
#    raster_func_c_self_bleach executes the simultaneous acquisition and bleaching routine for the case where the
#    excitation power (p_ex) AND/OR sted power (p_sted) vary through the sample. This function thus requires these
#    parameters to be passed as arrays of floats the same size as the ROI being imaged.
#
#    Additionally, this function seperately bleaches the different parts composing the datamap (i.e. the base and flash
#    components of the datamap are bleached separately).
#    """
#
#    if seed == 0:
#        # if no seed is passed, calculates a 'pseudo-random' seed form the time in ns
#        srand(int(str(time.time_ns())[-5:-1]))
#    else:
#        srand(seed)
#
#    i_ex, i_sted, _ = self.cache(datamap.pixelsize)
#    # Calculate the bleaching rate once if the scanning powers and dwelltimes do not vary to
#    # increase speed
#    uniform_ex = numpy.all(p_ex_roi == p_ex_roi[0, 0])
#    uniform_sted = numpy.all(p_sted_roi == p_sted_roi[0, 0])
#    uniform_pdt = numpy.all(pdt_roi == pdt_roi[0, 0])
#    if uniform_sted and uniform_ex and uniform_pdt:
#        p_ex = p_ex_roi[0, 0]
#        p_sted = p_sted_roi[0, 0]
#        step = pdt_roi[0, 0]
#        photons_ex = self.fluo.get_photons(i_ex * p_ex, self.excitation.lambda_)
#        duty_cycle = self.sted.tau * self.sted.rate
#        photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle, self.sted.lambda_)
#        k_sted = self.fluo.get_k_bleach(self.excitation.lambda_, self.sted.lambda_, photons_ex, photons_sted, self.sted.tau, 1/self.sted.rate, step,)
#        k_ex = k_sted * 0.
#    else:
#        k_sted = None
#        k_ex = None
#
#    pre_effective = self.get_effective(datamap.pixelsize, p_ex_roi[0, 0], p_sted_roi[0, 0])
#    h, w = pre_effective.shape[0], pre_effective.shape[1]
#
#    for (row, col) in pixel_list:
#        pdt = pdt_roi[row, col]
#        p_ex = p_ex_roi[row, col]
#        p_sted = p_sted_roi[row, col]
#        effective = self.get_effective(datamap.pixelsize, p_ex, p_sted)
#        # i think resetting each time ensures that we are acquiring on the dmap while it is
#        # being bleached. Either way, it doesn't affect speed, so I will keep it here
#        bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int32)
#        for key in bleached_sub_datamaps_dict:
#            bleached_datamap += bleached_sub_datamaps_dict[key]
#
#        value = 0.0
#        sprime = 0
#        for s in range(row, row + h):
#            tprime = 0
#            for t in range(col, col + w):
#                value += effective[sprime, tprime] * bleached_datamap[s, t]
#                tprime += 1
#            sprime += 1
#        # acquired_intensity[int(row / ratio), int(col / ratio)] = value
#        acquired_intensity[int(row / ratio), int(col / ratio)] += value
#
#        if bleach:
##           from pprint import pprint; pprint(dict(
##               i=1, 
##               prob_ex = prob_ex.mean(),
##                prob_sted_mean = prob_sted.mean(),
##                ))
#           prob_sted = prob_sted*0+1
#           bleach_func(self, i_ex, i_sted, p_ex, p_sted, pdt, bleached_sub_datamaps_dict, row, col, h, w, prob_ex,
#                        prob_sted, k_ex, k_sted)
##           from pprint import pprint; pprint(dict(
##               i=2, 
##               prob_ex = prob_ex.mean(),
##                prob_sted_mean = prob_sted.mean(),
##                log_ratio = numpy.log(prob_sted.mean())/numpy.log(0.9999881424494522),
##                prob_sted_shape = numpy.array(prob_sted).shape,
##                ))
#           sample_func(self, bleached_sub_datamaps_dict, row, col, h, w, prob_ex, prob_sted)
#           if row%10==0 and col%10==0:
#               print("row, col =", row, col)
#               
##           if row == 400 and col == 40:
##               import plotly.express as px
##               px.imshow(numpy.array(bleached_sub_datamaps_dict["base"])).show()
##               px.imshow(prob_sted).show()
##               px.imshow(numpy.exp(-k_sted*15e-6)).show()
##               px.imshow(k_sted).show()
##               print(numpy.prod(numpy.exp(-k_sted*15e-6)))
##               print(numpy.exp((-k_sted*15e-6).sum()))
##               from pprint import pprint
##               pprint(dict(
##                   pixel_list_len = len(pixel_list),
##                   pixel_list0_len = len(pixel_list[0]),
##                   ))
##               import pdb; pdb.set_trace()
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
    cdef numpy.ndarray[int, ndim=2] bleached_datamap
    cdef FLOATDTYPE_t duty_cycle
    cdef FLOATDTYPE_t step

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

    #TODO: remove prob_ex and prob_sted from the arguments of this function???
    #TODO: put this bloc maybe lower for better organization 
#    prob_ex = numpy.ones(numpy.array(i_ex).shape).astype("float64")
#    prob_sted = numpy.ones(numpy.array(i_sted).shape).astype("float64")
    #TODO: initialize those two arrays in a more legit way: ask Benoit and Antho
    #TODO: THis leads weidly to less than otpimized array access. What happened here??? Daratype???
    prob_ex = prob_ex[:numpy.array(i_ex).shape[0],:numpy.array(i_ex).shape[1]]
    prob_sted =prob_ex[:numpy.array(i_sted).shape[0],:numpy.array(i_sted).shape[1]]

    # Calculate the bleaching rate once if the scanning powers and dwelltimes do not vary to
    # increase speed
    uniform_ex = numpy.all(p_ex_roi == p_ex_roi[0, 0])
    uniform_sted = numpy.all(p_sted_roi == p_sted_roi[0, 0])
    uniform_pdt = numpy.all(pdt_roi == pdt_roi[0, 0])
    if uniform_sted and uniform_ex and uniform_pdt:
        p_ex = p_ex_roi[0, 0]
        p_sted = p_sted_roi[0, 0]
        step = pdt_roi[0, 0]
        photons_ex = self.fluo.get_photons(i_ex * p_ex, self.excitation.lambda_)
        duty_cycle = self.sted.tau * self.sted.rate
        photons_sted = self.fluo.get_photons(i_sted * p_sted * duty_cycle, self.sted.lambda_)
        k_sted = self.fluo.get_k_bleach(self.excitation.lambda_, self.sted.lambda_, photons_ex, photons_sted, self.sted.tau, 1/self.sted.rate, step,)
        k_ex = k_sted * 0.
    else:
        k_sted = None
        k_ex = None

        effective = self.get_effective(datamap.pixelsize, p_ex, p_sted)

    pre_effective = self.get_effective(datamap.pixelsize, p_ex_roi[0, 0], p_sted_roi[0, 0])
    h, w = pre_effective.shape[0], pre_effective.shape[1]

#    # Useful for "lasier" photobleaching calculations. Assumed image is squared
#    max_col = numpy.max([tple[1] for tple in pixel_list])
#    min_col = numpy.min([tple[1] for tple in pixel_list])
#    max_row = numpy.max([tple[0] for tple in pixel_list])
#    min_row = numpy.min([tple[0] for tple in pixel_list])

    effective = self.get_effective(datamap.pixelsize, p_ex, p_sted)        
    for (row, col) in pixel_list:
        t0 = time.time()
        pdt = pdt_roi[row, col]
        if (not uniform_sted) or (not uniform_ex):
            print("yo")
#        import pdb ;pdb.set_trace()
#        print(p_ex, type(p_ex), p_sted, type(p_sted))
#        p_ex = p_ex_roi[row, col]
#        p_sted = p_sted_roi[row, col]
#        print(p_ex, type(p_ex), p_sted, type(p_sted))
#        effective = self.get_effective(datamap.pixelsize, p_ex, p_sted)
   # i think resetting each time ensures that we are acquiring on the dmap while it is
        # being bleached. Either way, it doesn't affect speed, so I will keep it here
        bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int32)
        for key in bleached_sub_datamaps_dict:
            bleached_datamap += bleached_sub_datamaps_dict[key]

        value = 0.0
        sprime = 0
        for s in range(row, row + h):
            tprime = 0
            for t in range(col, col + w):
                value += effective[sprime, tprime] * bleached_datamap[s, t]
                tprime += 1
            sprime += 1
        # acquired_intensity[int(row / ratio), int(col / ratio)] = value
        acquired_intensity[int(row / ratio), int(col / ratio)] += value
#        print("tacq = ", time.time()-t0)


        if bleach:
            if numpy.min(prob_sted)==1 and numpy.min(prob_ex) ==1:
                if p_sted !=0:
#                    print("calculating prob_ex prob_sted")
                    prob_sted = prob_sted*0+1#reset to array of 1s
                    prob_ex = prob_ex*0+1#reset to array of 1s
    #                t0 = time.time()
                    bleach_func(self, i_ex, i_sted, p_ex, p_sted, pdt, bleached_sub_datamaps_dict, row, col, h, w, prob_ex,
                                prob_sted, k_ex, k_sted)
#                    print(numpy.min(prob_sted), numpy.min(prob_ex))
#                print("bleaching time = ", time.time()-t0)
#            if (row == 30 and col == 30) :
#               import plotly.express as px
#               px.imshow(prob_sted, title="prob_sted", template="plotly_dark").show()

#            t0 = time.time()
            sample_func(self, bleached_sub_datamaps_dict, row, col, h, w, prob_ex, prob_sted)
#            print("tbleach = ", time.time()-t0)
#            if col % 20 == 0:
#                print(row, col)
#            if row == 50 and col==50:
#                print("row == 50 and col==5")
#            if row>=30 and col>=30:
#                from pprint import pprint
#                pprint(dict(tbf=tbf, tsf=tsf))
#            print("total pixeltime=", time.time() - t0)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def raster_func_c_self_bleach_dymin(
        object self,
        object datamap,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] acquired_intensity,
        numpy.ndarray[FLOATDTYPE_t, ndim=2] scaled_power,
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
        object bleach_func   # uncertain of the type for a cfunc, but this seems to be working so ???
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
    cdef numpy.ndarray[FLOATDTYPE_t, ndim=1] pdts, p_exs, p_steds
    cdef numpy.ndarray[int, ndim=2] bleached_datamap
    cdef FLOATDTYPE_t duty_cycle

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
    pre_effective = self.get_effective(datamap.pixelsize, p_ex_roi[0, 0], p_sted_roi[0, 0])
    h, w = pre_effective.shape[0], pre_effective.shape[1]

    scale_powers = self.opts["scale_power"]
    decision_times = self.opts["decision_time"]
    threshold_counts = self.opts["threshold_count"]

    for (row, col) in (pixel_list):
        pdts, p_exs, p_steds = numpy.zeros(len(scale_powers)), numpy.zeros(len(scale_powers)), numpy.zeros(len(scale_powers))
        for i in range(len(scale_powers)):
            pdt = decision_times[i]
            scale_power = scale_powers[i]
            threshold_count = threshold_counts[i]
            if pdt < 0.:
                pdt = pdt_roi[row, col]

            p_ex = p_ex_roi[row, col]
            p_sted = scale_power * p_sted_roi[row, col]
            effective = self.get_effective(datamap.pixelsize, p_ex, p_sted)
            # i think resetting each time ensures that we are acquiring on the dmap while it is
            # being bleached. Either way, it doesn't affect speed, so I will keep it here
            bleached_datamap = numpy.zeros(bleached_sub_datamaps_dict["base"].shape, dtype=numpy.int32)
            for key in bleached_sub_datamaps_dict:
                bleached_datamap += bleached_sub_datamaps_dict[key]

            value = 0.0
            sprime = 0
            for s in range(row, row + h):
                tprime = 0
                for t in range(col, col + w):
                    value += effective[sprime, tprime] * bleached_datamap[s, t]
                    tprime += 1
                sprime += 1
            acquired_intensity[int(row / ratio), int(col / ratio)] += value

            pdts[i] = pdt
            p_exs[i] = p_ex
            p_steds[i] = p_sted

            scaled_power[row, col] = scale_power
            photons = numpy.array(self.fluo.get_photons(value))
            photons = self.detector.get_signal(photons, pdt)
            if photons < threshold_count:
                acquired_intensity[int(row / ratio), int(col / ratio)] = 0
                break

        if bleach:
            bleach_func(self, i_ex, i_sted, p_exs, p_steds, pdts,
                        bleached_sub_datamaps_dict, row, col, h, w,
                        prob_ex, prob_sted)
