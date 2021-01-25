
'''This module implements the essential components of a microscopy setup. By
assembling an excitation beam, a STED beam, a detector, and fluorescence
molecules, to obtain a microscope setup. The following code gives an example of
how to create such setup and use it to measure and plot the detected signal
given some ``data_model``.
    
.. code-block:: python
    
    laser_ex = base.GaussianBeam(488e-9)
    laser_sted = base.DonutBeam(575e-9, zero_residual=0.04)
    detector = base.Detector(def=0.02)
    objective = base.Objective()
    fluo = base.Fluorescence(535e-9)
    microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)
    
    data_map = io.read_data_map(data_model, 1)
    
    # imaging parameters
    pdt = 10e-6
    p_ex = 1e-6
    p_sted = 30e-3

    signal, _ = microscope.get_signal(data_map, 10e-9, pdt, p_ex, p_sted)
    
    from matplotlib import pyplot
    pyplot.imshow(signal)
    pyplot.colorbar()
    pyplot.show()

Code written by Benoit Turcotte, benoit.turcotte.4@ulaval.ca, October 2020
For use by FLClab (@CERVO) authorized people

.. rubric:: References


.. [Deng2010] Deng, S., Liu, L., Cheng, Y., Li, R., & Xu, Z. (2010).
   Effects of primary aberrations on the fluorescence depletion patterns
   of STED microscopy. Optics Express, 18(2), 1657–1666.

.. [Garcia2000] Garcia-Parajo, M. F., Segers-Nolten, G. M. J., Veerman, J.-A-.,
   Greve, J., & Van Hulst, N. F. (2000).
   Real-time light-driven dynamics of the fluorescence emission in single green
   fluorescent protein molecules.
   Proceedings of the National Academy of Sciences, 97(13), 7237-7242.

.. [Holler2011] Höller, M. (2011).
   Advanced fluorescence fluctuation spectroscopy with pulsed
   interleaved excitation. PhD thesis, LMU, Munich (Germany).

.. [Jerker1999] Jerker, W., Ülo, M., & Rudolf, R. (1999).
   Photodynamic properties of green fluorescent proteins investigated by
   fluorescence correlation spectroscopy. Chemical Physics, 250(2), 171-186.
   
.. [Leutenegger2010] Leutenegger, M., Eggeling, C., & Hell, S. W. (2010).
    Analytical description of STED microscopy performance.
    Optics Express, 18(25), 26417–26429.

.. [RPPhoto2015] RP Photonics Consulting GmbH (Accessed 2015).
   Optical intensity. Encyclopedia of laser physics and technology at
   <https://www.rp-photonics.com/optical_intensity.html>.

.. [Staudt2009] Staudt, T. M. (2009).
   Strategies to reduce photobleaching, dark state transitions and phototoxicity
   in subdiffraction optical microscopy. Dissertation, University of Heidelberg,
   Heidelberg (Germany).

.. [Willig2006] Willig, K. I., Keller, J., Bossi, M., & Hell, S. W. (2006).
   STED microscopy resolves nanoparticle assemblies.
   New Journal of Physics, 8(6), 106.
   
.. [Xie2013] Xie, H., Liu, Y., Jin, D., Santangelo, P. J., & Xi, P. (2013).
   Analytical description of high-aperture STED resolution with 0–2pi
   vortex phase modulation.
   Journal of the Optical Society of America A (JOSA A), 30(8), 1640–1645.
'''

import numpy
import scipy.constants
import scipy.signal

# from pysted import cUtils, utils   # je dois changer ce import en les 2 autres en dessous pour que ça marche
from pysted import utils, bleach_functions, cUtils, raster
# import cUtils

# import mis par BT pour des tests
import configparser
import ast
import warnings
from matplotlib import pyplot
import time
from functools import partial


class GaussianBeam:
    '''This class implements a Gaussian beam (excitation).
    
    :param lambda_: The wavelength of the beam (m).
    :param kwargs: One or more parameters as described in the following table,
                   optional.
    
    +------------------+--------------+----------------------------------------+
    | Parameter        | Default      | Details                                |
    +==================+==============+========================================+
    | ``polarization`` | ``pi/2``     | The phase difference between :math:`x` |
    |                  |              | and :math:`y` oscillations (rad).      |
    +------------------+--------------+----------------------------------------+
    | ``beta``         | ``pi/4``     | The beam incident angle, in            |
    |                  |              | :math:`[0, \pi/2]` (rad).              |
    +------------------+--------------+----------------------------------------+
    
    Polarization :
        * :math:`\pi/2` is left-circular
        * :math:`0` is linear
        * :math:`-\pi/2` is right-circular
    '''
    
    def __init__(self, lambda_, **kwargs):
        self.lambda_ = lambda_
        self.polarization = kwargs.get("polarization", numpy.pi/2)
        self.beta = kwargs.get("beta", numpy.pi/4)
    
    # FIXME: pass Objective object instead of f, n, na, transmission
    def get_intensity(self, power, f, n, na, transmission, datamap_pixelsize):
        '''Compute the transmitted excitation intensity field (W/m²). The
        technique essentially follows the method described in [Xie2013]_,
        where :math:`z = 0`, along with some equations from [Deng2010]_, and
        [RPPhoto2015]_.
        
        :param power: The power of the beam (W).
        :param f: The focal length of the objective (m).
        :param n: The refractive index of the objective.
        :param na: The numerical aperture of the objective.
        :param transmission: The transmission ratio of the objective (given the
                             wavelength of the excitation beam).
        :param datamap_pixelsize: The size of an element in the intensity matrix (m).
        :returns: A 2D array.
        '''

        def fun1(theta, kr):
            return numpy.sqrt(numpy.cos(theta)) * numpy.sin(theta) *\
                   scipy.special.jv(0, kr * numpy.sin(theta)) * (1 + numpy.cos(theta))
        def fun2(theta, kr):
            return numpy.sqrt(numpy.cos(theta)) * numpy.sin(theta)**2 *\
                   scipy.special.jv(1, kr * numpy.sin(theta))
        def fun3(theta, kr):
            return numpy.sqrt(numpy.cos(theta)) * numpy.sin(theta) *\
                   scipy.special.jv(2, kr * numpy.sin(theta)) * (1 - numpy.cos(theta))
        
        alpha = numpy.arcsin(na / n)
        
        diameter = 2.233 * self.lambda_ / (na * datamap_pixelsize)
        n_pixels = int(diameter / 2) * 2 + 1 # odd number of pixels
        center = int(n_pixels / 2)
        
        # [Deng2010]
        k = 2 * numpy.pi * n / self.lambda_
        
        # compute the focal plane integrations i1 to i3 [Xie2013]
        i1 = numpy.empty((n_pixels, n_pixels))
        i2 = numpy.empty((n_pixels, n_pixels))
        i3 = numpy.empty((n_pixels, n_pixels))
        phi = numpy.empty((n_pixels, n_pixels))
        for y in range(n_pixels):
            h_rel = (center - y)
            for x in range(n_pixels):
                w_rel = (x - center)

                angle, radius = utils.cart2pol(w_rel, h_rel)
                
                kr = k * radius * datamap_pixelsize
                i1[y, x] = scipy.integrate.quad(fun1, 0, alpha, (kr,))[0]
                i2[y, x] = scipy.integrate.quad(fun2, 0, alpha, (kr,))[0]
                i3[y, x] = scipy.integrate.quad(fun3, 0, alpha, (kr,))[0]
                phi[y, x] = angle
        
        ax = numpy.sin(self.beta)
        ay = numpy.cos(self.beta) * numpy.exp(1j * self.polarization)

        # [Xie2013] eq. 1, where exdx = e_x, eydx = e_y, and ezdx = e_z
        exdx = -ax * 1j * (i1 + i3 * numpy.cos(2*phi))
        eydx = -ax * 1j * i3 * numpy.sin(2*phi)
        ezdx = -ax * 2 * i2 * numpy.cos(phi)
        # [Xie2013] appendix A, where exdy = e'_{1x'}, eydy = e'_{1y'}, and ezdy = e'_{1z'}
        exdy = -ay * 1j * (i1 - i3 * numpy.cos(2*phi))
        eydy = ay * 1j * i3 * numpy.sin(2*phi)
        ezdy = -ay * 2 * i2 * numpy.sin(phi)
        # [Xie2013] eq. 3
        electromagfieldx = exdx - eydy
        electromagfieldy = eydx + exdy
        electromagfieldz = ezdx + ezdy
        
        # [Xie2013] I = E_x E_x* + E_y E_y* + E_z E_z* (p. 1642)
        intensity = electromagfieldx * numpy.conj(electromagfieldx) +\
                    electromagfieldy * numpy.conj(electromagfieldy) +\
                    electromagfieldz * numpy.conj(electromagfieldz)
        
        # keep it real
        intensity = numpy.real_if_close(intensity)
        # normalize
        intensity /= numpy.max(intensity)

        # Here, the laser should be perfectly symmetrical, however, it is not because of the way python/computers handle
        # floating point values. In order to make it symmetrical, as it should be, we flip the upper right corner of
        # the laser over the rest of the laser, in order to "patch" it to be symmetrical. The asymetries are typically
        # values 10^16 times smaller than the values of the laser.
        intensity_flipped = numpy.zeros(intensity.shape)
        intensity_tr = intensity[0: n_pixels // 2 + 1, n_pixels // 2:]
        intensity_flipped[0: n_pixels // 2 + 1, n_pixels // 2:] = intensity_tr
        intensity_flipped[0: n_pixels // 2 + 1, 0: n_pixels // 2 + 1] = numpy.flip(intensity_tr, 1)
        intensity_flipped[n_pixels // 2:, 0: n_pixels // 2 + 1] = numpy.flip(intensity_tr)
        intensity_flipped[n_pixels // 2:, n_pixels // 2:] = numpy.flip(intensity_tr, 0)
        
        idx_mid = int((intensity.shape[0]-1) / 2)
        r = utils.fwhm(intensity[idx_mid])
        area_fwhm = numpy.pi * (r * datamap_pixelsize) ** 2 / 2
        # [RPPhoto2015]
        return intensity_flipped * 2 * transmission * power / area_fwhm


class DonutBeam:
    '''This class implements a donut beam (STED).
    
    :param lambda_: The wavelength of the beam (m).
    :param parameters: One or more parameters as described in the following
                       table, optional.
    
    +------------------+--------------+----------------------------------------+
    | Parameter        | Default      | Details                                |
    +==================+==============+========================================+
    | ``polarization`` | ``pi/2``     | The phase difference between :math:`x` |
    |                  |              | and :math:`y` oscillations (rad).      |
    +------------------+--------------+----------------------------------------+
    | ``beta``         | ``pi/4``     | The beam incident angle, in            |
    |                  |              | :math:`[0, \pi/2]` (rad).              |
    +------------------+--------------+----------------------------------------+
    | ``tau``          | ``200e-12``  | The beam pulse length (s).             |
    +------------------+--------------+----------------------------------------+
    | ``rate``         | ``80e6``     | The beam pulse rate (Hz).              |
    +------------------+--------------+----------------------------------------+
    | ``zero_residual``| ``0``        | The ratio between minimum and maximum  |
    |                  |              | intensity (ratio).                     |
    +------------------+--------------+----------------------------------------+
    
    Polarization :
        * :math:`\pi/2` is left-circular
        * :math:`0` is linear
        * :math:`-\pi/2` is right-circular
    '''
    
    def __init__(self, lambda_, **kwargs):
        self.lambda_ = lambda_
        self.polarization = kwargs.get("polarization", numpy.pi/2)
        self.beta = kwargs.get("beta", numpy.pi/4)
        self.tau = kwargs.get("tau", 200e-12)
        self.rate = kwargs.get("rate", 80e6)
        self.zero_residual = kwargs.get("zero_residual", 0)
    
    # FIXME: pass Objective object instead of f, n, na, transmission
    def get_intensity(self, power, f, n, na, transmission, datamap_pixelsize):
        '''Compute the transmitted STED intensity field (W/m²). The technique
        essentially follows the method described in [Xie2013]_, where
        :math:`z = 0`, along with some equations from [Deng2010]_, and
        [RPPhoto2015]_.
        
        :param power: The power of the beam (W).
        :param f: The focal length of the objective (m).
        :param n: The refractive index of the objective.
        :param na: The numerical aperture.
        :param transmission: The transmission ratio of the objective (given the
                             wavelength of the STED beam).
        :param datamap_pixelsize: The size of an element in the intensity matrix (m).
        '''

        def fun1(theta, kr):
            return numpy.sqrt(numpy.cos(theta)) * numpy.sin(theta) *\
                   scipy.special.jv(1, kr * numpy.sin(theta)) * (1 + numpy.cos(theta))
        def fun2(theta, kr):
            return numpy.sqrt(numpy.cos(theta)) * numpy.sin(theta) *\
                   scipy.special.jv(1, kr * numpy.sin(theta)) * (1 - numpy.cos(theta))
        def fun3(theta, kr):
            return numpy.sqrt(numpy.cos(theta)) * numpy.sin(theta) *\
                   scipy.special.jv(3, kr * numpy.sin(theta)) * (1 - numpy.cos(theta))
        def fun4(theta, kr):
            return numpy.sqrt(numpy.cos(theta)) * numpy.sin(theta)**2 *\
                   scipy.special.jv(0, kr * numpy.sin(theta))
        def fun5(theta, kr):
            return numpy.sqrt(numpy.cos(theta)) * numpy.sin(theta)**2 *\
                   scipy.special.jv(2, kr * numpy.sin(theta))
        
        alpha = numpy.arcsin(na / n)
        
        diameter = 2.233 * self.lambda_ / (na * datamap_pixelsize)
        n_pixels = int(diameter / 2) * 2 + 1 # odd number of pixels
        center = int(n_pixels / 2)
        
        # [Deng2010]
        k = 2 * numpy.pi * n / self.lambda_
        
        # compute the angular integrations i1 to i5 [Xie2013]
        i1 = numpy.zeros((n_pixels, n_pixels))
        i2 = numpy.zeros((n_pixels, n_pixels))
        i3 = numpy.zeros((n_pixels, n_pixels))
        i4 = numpy.zeros((n_pixels, n_pixels))
        i5 = numpy.zeros((n_pixels, n_pixels))
        phi = numpy.zeros((n_pixels, n_pixels))
        for y in range(n_pixels):
            h_rel = (center - y)
            for x in range(n_pixels):
                w_rel = (x - center)
                
                angle, radius = utils.cart2pol(w_rel, h_rel)
                
                kr = k * radius * datamap_pixelsize
                
                i1[y, x] = scipy.integrate.quad(fun1, 0, alpha, (kr,))[0]
                i2[y, x] = scipy.integrate.quad(fun2, 0, alpha, (kr,))[0]
                i3[y, x] = scipy.integrate.quad(fun3, 0, alpha, (kr,))[0]
                i4[y, x] = scipy.integrate.quad(fun4, 0, alpha, (kr,))[0]
                i5[y, x] = scipy.integrate.quad(fun5, 0, alpha, (kr,))[0]
                phi[y, x] = angle
        
        ax = numpy.sin(self.beta)
        ay = numpy.cos(self.beta) * numpy.exp(1j * self.polarization)
        
        # [Xie2013] eq. 2, where exdx = e_x, eydx = e_y, and ezdx = e_z
        exdx = ax * (i1 * numpy.exp(1j * phi) -\
                     i2 / 2 * numpy.exp(-1j * phi) +\
                     i3 / 2 * numpy.exp(3j * phi))
        eydx = -ax * 1j / 2 * (i2 * numpy.exp(-1j * phi) +\
                               i3 * numpy.exp(3j * phi))
        ezdx = ax * 1j * (i4 - i5 * numpy.exp(2j * phi))
        
        # [Xie2013] annexe A, where exdy = e'_{2x'}, eydy = e'_{2y'}, and ezdy = e'_{2z'}
        exdy = ay * (i1 * numpy.exp(1j * phi) +\
                     i2 / 2 * numpy.exp(-1j * phi) -\
                     i3 / 2 * numpy.exp(3j * phi))
        eydy = ay * 1j / 2 * (i2 * numpy.exp(-1j * phi) +\
                              i3 * numpy.exp(3j * phi))
        ezdy = -ay * (i4 + i5 * numpy.exp(2j * phi))
        
        # [Xie2013] eq. 3
        electromagfieldx = exdx - eydy
        electromagfieldy = eydx + exdy
        electromagfieldz = ezdx + ezdy
        
        # [Xie2013] I = E_x E_x* + E_y E_y* + E_z E_z* (p. 1642)
        intensity = electromagfieldx * numpy.conj(electromagfieldx) +\
                    electromagfieldy * numpy.conj(electromagfieldy) +\
                    electromagfieldz * numpy.conj(electromagfieldz)
        
        # keep it real
        intensity = numpy.real_if_close(intensity)
        
        # normalize
        intensity /= numpy.max(intensity)

        # Here, the laser should be perfectly symmetrical, however, it is not because of the way python/computers handle
        # floating point values. In order to make it symmetrical, as it should be, we flip the upper right corner of
        # the laser over the rest of the laser, in order to "patch" it to be symmetrical. The asymetries are typically
        # values 10^16 times smaller than the values of the laser.
        intensity_flipped = numpy.zeros(intensity.shape)
        intensity_tr = intensity[0: n_pixels // 2 + 1, n_pixels // 2:]
        intensity_flipped[0: n_pixels // 2 + 1, n_pixels // 2:] = intensity_tr
        intensity_flipped[0: n_pixels // 2 + 1, 0: n_pixels // 2 + 1] = numpy.flip(intensity_tr, 1)
        intensity_flipped[n_pixels // 2:, 0: n_pixels // 2 + 1] = numpy.flip(intensity_tr)
        intensity_flipped[n_pixels // 2:, n_pixels // 2:] = numpy.flip(intensity_tr, 0)
        intensity = intensity_flipped
        
        # for peak intensity
        duty_cycle = self.tau * self.rate
        intensity /= duty_cycle
        
        idx_mid = int((intensity.shape[0]-1) / 2)
        r_out, r_in = utils.fwhm_donut(intensity[idx_mid])
        big_area = numpy.pi * (r_out * datamap_pixelsize) ** 2 / 2
        small_area = numpy.pi * (r_in * datamap_pixelsize) ** 2 / 2
        area_fwhm = big_area - small_area
        
        # [RPPhoto2015]
        intensity *= 2 * transmission * power / area_fwhm
        
        if power > 0:
            # zero_residual ~= min(intensity) / max(intensity)
            old_max = numpy.max(intensity)
            intensity += self.zero_residual * old_max
            intensity /= numpy.max(intensity)
            intensity *= old_max

        return intensity


class Detector:
    '''This class implements the photon detector component.
    
    :param parameters: One or more parameters as described in the following
                       table, optional.
    
    +------------------+--------------+----------------------------------------+
    | Parameter        | Default      | Details                                |
    +==================+==============+========================================+
    | ``n_airy``       | ``0.7``      | The number of airy disks used to       |
    |                  |              | compute the pinhole radius             |
    |                  |              | :math:`r_b = n_{airy} 0.61 \lambda/NA`.|
    +------------------+--------------+----------------------------------------+
    | ``noise``        | ``False``    | Whether to add poisson noise to the    |
    |                  |              | signal (boolean).                      |
    +------------------+--------------+----------------------------------------+
    | ``background``   | ``0``        | The average number of photon counts per|
    |                  |              | second due to the background [#]_.     |
    +------------------+--------------+----------------------------------------+
    | ``darkcount``    | ``0``        | The average number of photon counts per|
    |                  |              | second due to dark counts.             |
    +------------------+--------------+----------------------------------------+
    | ``pcef``         | ``0.1``      | The photon collection efficiency factor|
    |                  |              | is the ratio of emitted photons that   |
    |                  |              | could be detected (ratio).             |
    +------------------+--------------+----------------------------------------+
    | ``pdef``         | ``0.5`` [#]_ | The photon detection efficiency factor |
    |                  |              | is the ratio of collected photons that |
    |                  |              | are perceived by the detector (ratio). |
    +------------------+--------------+----------------------------------------+
    
    .. [#] The actual number is sampled from a poisson distribution with given
       mean.
    
    .. [#] Excelitas Technologies. (2011). Photon Detection Solutions.
    '''
    
    def __init__(self, **kwargs):
        # detection pinhole
        self.n_airy = kwargs.get("n_airy", 0.7)
        
        # detection noise
        self.noise = kwargs.get("noise", False)
        self.background = kwargs.get("background", 0)
        self.darkcount = kwargs.get("darkcount", 0)
        
        # photon detection
        self.pcef = kwargs.get("pcef", 0.1)
        self.pdef = kwargs.get("pdef", 0.5)
    
    def get_detection_psf(self, lambda_, psf, na, transmission, datamap_pixelsize):
        '''Compute the detection PSF as a convolution between the fluorscence
        PSF and a pinhole, as described by the equation from [Willig2006]_. The
        pinhole raidus is determined using the :attr:`n_airy`, the fluorescence
        wavelength, and the numerical aperture of the objective.
        
        :param lambda_: The fluorescence wavelength (m).
        :param psf: The fluorescence PSF that can the obtained using
                    :meth:`~pysted.base.Fluorescence.get_psf`.
        :param na: The numerical aperture of the objective.
        :param transmission: The transmission ratio of the objective for the
                             given fluorescence wavelength *lambda_*.
        :param datamap_pixelsize: The size of a pixel in the simulated image (m).
        :returns: A 2D array.
        '''
        radius = self.n_airy * 0.61 * lambda_ / na
        pinhole = utils.pinhole(radius, datamap_pixelsize, psf.shape[0])
        # convolution [Willig2006] eq. 3
        psf_det = scipy.signal.convolve2d(psf, pinhole, "same")
        # normalization to 1
        psf_det = psf_det / numpy.max(psf_det)

        # Here, the psf should be perfectly symmetrical, however, it is not because of the way python/computers handle
        # floating point values. In order to make it symmetrical, as it should be, we flip the upper right corner of
        # the psf over the rest of the laser, in order to "patch" it to be symmetrical. The asymetries are typically
        # values 10^16 times smaller than the values of the psf.
        returned_array = psf_det * transmission
        ra_flipped = numpy.zeros(returned_array.shape)
        intensity_tr = returned_array[0: returned_array.shape[0] // 2 + 1, returned_array.shape[1] // 2:]
        ra_flipped[0: returned_array.shape[0] // 2 + 1, returned_array.shape[1] // 2:] = intensity_tr
        ra_flipped[0: returned_array.shape[0] // 2 + 1, 0: returned_array.shape[1] // 2 + 1] = numpy.flip(intensity_tr,
                                                                                                          1)
        ra_flipped[returned_array.shape[0] // 2:, 0: returned_array.shape[1] // 2 + 1] = numpy.flip(intensity_tr)
        ra_flipped[returned_array.shape[0] // 2:, returned_array.shape[1] // 2:] = numpy.flip(intensity_tr, 0)

        return ra_flipped
    
    def get_signal(self, photons, dwelltime):
        '''Compute the detected signal (in photons) given the number of emitted
        photons and the time spent by the detector.
        
        :param photons: An array of number of emitted photons.
        :param dwelltime: The time spent to detect the emitted photons (s). It is
                          either a scalar or an array shaped like *nb_photons*.
        :returns: An array shaped like *nb_photons*.
        '''
        detection_efficiency = self.pcef * self.pdef # ratio
        try:
            signal = numpy.random.binomial(photons.astype(numpy.int64),
                                           detection_efficiency,
                                           photons.shape) * dwelltime
        except:
            # on Windows numpy.random.binomial cannot generate 64-bit integers
            # MARCHE PAS QUAND C'EST JUSTE UN SCALAIRE QUI EST PASSÉ
            signal = utils.approx_binomial(photons.astype(numpy.int64),
                                           detection_efficiency,
                                           photons.shape) * dwelltime
        # add noise, background, and dark counts

        if self.noise:
            signal = numpy.random.poisson(signal, signal.shape)
        if self.background > 0:
            # background counts per second
            cts = numpy.random.poisson(self.background, signal.shape)
            # background counts during dwell time
            cts = (cts * dwelltime).astype(numpy.int64)
            signal += cts
        if self.darkcount > 0:
            # dark counts per second
            cts = numpy.random.poisson(self.darkcount, signal.shape)
            # dark counts during dwell time
            cts = (cts * dwelltime).astype(numpy.int64)
            signal += cts
        return signal


class Objective:
    '''
    This class implements the microscope objective component.
    
    :param parameters: One or more parameters as described in the following
                       table, optional.
    
    +------------------+-------------------+-----------------------------------+
    | Parameter        | Default [#]_      | Details                           |
    +==================+===================+===================================+
    | ``f``            | ``2e-3``          | The objective focal length (m).   |
    +------------------+-------------------+-----------------------------------+
    | ``n``            | ``1.5``           | The refractive index.             |
    +------------------+-------------------+-----------------------------------+
    | ``na``           | ``1.4``           | The numerical aperture.           |
    +------------------+-------------------+-----------------------------------+
    | ``transmission`` | ``488: 0.84,``    | A dictionary mapping wavelengths  |
    |                  | ``535: 0.85,``    | (nm), as integer, to objective    |
    |                  | ``550: 0.86,``    | transmission factors (ratio).     |
    |                  | ``585: 0.85,``    |                                   |
    |                  | ``575: 0.85``     |                                   |
    +------------------+-------------------+-----------------------------------+
    
    .. [#] Leica 100x tube lense
    '''
    
    def __init__(self, **kwargs):
        self.f = kwargs.get("f", 2e-3)
        self.n = kwargs.get("n", 1.5)
        self.na = kwargs.get("na", 1.4)
        self.transmission = kwargs.get("transmission", {488: 0.84,
                                                        535: 0.85,
                                                        550: 0.86,
                                                        585: 0.85,
                                                        575: 0.85})
    
    def get_transmission(self, lambda_):
        return self.transmission[int(lambda_ * 1e9)]


class Fluorescence:
    '''This class implements a fluorescence molecule.
    
    :param lambda_: The fluorescence wavelength (m).
    :param parameters: One or more parameters as described in the following
                       table, optional.
    
    +------------------+--------------+----------------------------------------+
    | Parameter        | Default [#]_ | Details                                |
    +==================+==============+========================================+
    | ``sigma_ste``    |``575: 1e-21``| A dictionnary mapping STED wavelengths |
    |                  |              | as integer (nm) to stimulated emission |
    |                  |              | cross-section (m²).                    |
    +------------------+--------------+----------------------------------------+
    | ``sigma_abs``    |``488: 3e-20``| A dictionnary mapping excitation       |
    |                  |              | wavelengths as integer (nm) to         |
    |                  |              | absorption cross-section (m²).         |
    +------------------+--------------+----------------------------------------+
    | ``sigma_tri``    |``1e-21``     | The cross-section for triplet-triplet  |
    |                  |              | absorption (m²).                       |
    +------------------+--------------+----------------------------------------+
    | ``tau``          | ``3e-9``     | The fluorescence lifetime (s).         |
    +------------------+--------------+----------------------------------------+
    | ``tau_vib``      | ``1e-12``    | The vibrational relaxation (s).        |
    +------------------+--------------+----------------------------------------+
    | ``tau_tri``      | ``5e-6``     | The triplet state lifetime (s).        |
    +------------------+--------------+----------------------------------------+
    | ``qy``           | ``0.6``      | The quantum yield (ratio).             |
    +------------------+--------------+----------------------------------------+
    | ``phy_react``    | ``488: 1e-3``| A dictionnary mapping wavelengths as   |
    |                  | ``575: 1e-5``| integer (nm) to the probability of     |
    |                  |              | reaction once the molecule is in       |
    |                  |              | triplet state T_1 (ratio).             |
    +------------------+--------------+----------------------------------------+
    | ``k_isc``        | ``1e6``      | The intersystem crossing rate (s⁻¹).   |
    +------------------+--------------+----------------------------------------+
    
    .. [#] EGFP
    '''
    def __init__(self, lambda_, **kwargs):
        # psf parameters
        self.lambda_ = lambda_
        
        self.sigma_ste = kwargs.get("sigma_ste", {575: 1e-21})
        self.sigma_abs = kwargs.get("sigma_abs", {488: 3e-20})
        self.sigma_tri = kwargs.get("sigma_tri", 1e-21)
        self.tau = kwargs.get("tau", 3e-9)
        self.tau_vib = kwargs.get("tau_vib", 1e-12)
        self.tau_tri = kwargs.get("tau_tri", 5e-6)
        self.qy = kwargs.get("qy", 0.6)
        self.phy_react = kwargs.get("phy_react", {488: 1e-3, 575: 1e-5})
        self.k_isc = kwargs.get("k_isc", 0.26e6)
    
    def get_sigma_ste(self, lambda_):
        '''Return the stimulated emission cross-section of the fluorescence
        molecule given the wavelength.
        
        :param lambda_: The STED wavelength (m).
        :returns: The stimulated emission cross-section (m²).
        '''
        return self.sigma_ste[int(lambda_ * 1e9)]
        
    def get_sigma_abs(self, lambda_):
        '''Return the absorption cross-section of the fluorescence molecule
        given the wavelength.
        
        :param lambda_: The STED wavelength (m).
        :returns: The absorption cross-section (m²).
        '''
        return self.sigma_abs[int(lambda_ * 1e9)]
    
    def get_phy_react(self, lambda_):
        '''Return the reaction probability of the fluorescence molecule once it
        is in triplet state T_1 given the wavelength.
        
        :param lambda_: The STED wavelength (m).
        :returns: The probability of reaction (ratio).
        '''
        return self.phy_react[int(lambda_ * 1e9)]
    
    def get_psf(self, na, datamap_pixelsize):
        '''Compute the Gaussian-shaped fluorescence PSF.
        
        :param na: The numerical aperture of the objective.
        :param datamap_pixelsize: The size of an element in the intensity matrix (m).
        :returns: A 2D array.
        '''
        diameter = 2.233 * self.lambda_ / (na * datamap_pixelsize)
        n_pixels = int(diameter / 2) * 2 + 1 # odd number of pixels
        center = int(n_pixels / 2)
        
        fwhm = self.lambda_ / (2 * na)
        
        half_pixelsize = datamap_pixelsize / 2
        gauss = numpy.zeros((n_pixels, n_pixels))
        for y in range(n_pixels):
            h_rel = (center - y) * datamap_pixelsize
            h_lb = h_rel - half_pixelsize
            h_ub = h_rel + half_pixelsize
            for x in range(n_pixels):
                w_rel = (x - center) * datamap_pixelsize
                w_lb = w_rel - half_pixelsize
                w_ub = w_rel + half_pixelsize
                gauss[y, x] = scipy.integrate.nquad(cUtils.calculate_amplitude,
                                                    ((h_lb, h_ub), (w_lb, w_ub)),
                                                    (fwhm,))[0]
        return numpy.real_if_close(gauss / numpy.max(gauss))
    
    def get_photons(self, intensity):
        e_photon = scipy.constants.c * scipy.constants.h / self.lambda_
        return intensity // e_photon
    
    def get_k_bleach(self, lambda_, photons):
        sigma_abs = self.get_sigma_abs(lambda_)
        phy_react = self.get_phy_react(lambda_)
        T_1 = self.k_isc * sigma_abs * photons /\
                (sigma_abs * photons * (1/self.tau_tri + self.k_isc) +\
                (self.tau_tri * self.tau))
        return T_1 * photons * self.sigma_tri * phy_react


class Microscope:
    '''This class implements a microscopy setup described by an excitation beam,
    a STED (depletion) beam, a detector, some fluorescence molecules, and the
    parameters of the objective.
    
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
    '''
    
    def __init__(self, excitation, sted, detector, objective, fluo, bleach_func="default_bleach"):
        self.excitation = excitation
        self.sted = sted
        self.detector = detector
        self.objective = objective
        self.fluo = fluo
                
        # caching system
        self.__cache = {}

        if bleach_func not in bleach_functions.functions_dict:
            raise ValueError("Not a valid bleaching function")
        else:
            self.bleach_func = bleach_functions.functions_dict[bleach_func]

        # This will be used during the acquisition routine to make a better correspondance
        # between the microscope acquisition time steps and the Ca2+ flash time steps
        self.pixel_bank = 0
    
    def __str__(self):
        return str(self.__cache.keys())
    
    def is_cached(self, datamap_pixelsize):
        '''Indicate the presence of a cache entry for the given pixel size.
        
        :param datamap_pixelsize: The size of a pixel in the simulated image (m).
        :returns: A boolean.
        '''

        datamap_pixelsize_nm = int(datamap_pixelsize * 1e9)
        return datamap_pixelsize_nm in self.__cache
    
    def cache(self, datamap_pixelsize):
        '''Compute and cache the excitation and STED intensities, and the
        fluorescence PSF. These intensities are computed with a power of 1 W
        such that they can serve as a basis to compute intensities with any
        power.
        
        :param datamap_pixelsize: The size of a pixel in the simulated image (m).
        :returns: A tuple containing:
        
                  * A 2D array of the excitation intensity for a power of 1 W;
                  * A 2D array of the STED intensity for a a power of 1 W;
                  * A 2D array of the detection PSF.
        '''

        datamap_pixelsize_nm = int(datamap_pixelsize * 1e9)
        if datamap_pixelsize_nm not in self.__cache:
            f, n, na = self.objective.f, self.objective.n, self.objective.na
            
            transmission = self.objective.get_transmission(self.excitation.lambda_)
            i_ex = self.excitation.get_intensity(1, f, n, na,
                                                 transmission, datamap_pixelsize)
            
            transmission = self.objective.get_transmission(self.sted.lambda_)
            i_sted = self.sted.get_intensity(1, f, n, na,
                                             transmission, datamap_pixelsize)
            
            transmission = self.objective.get_transmission(self.fluo.lambda_)
            psf = self.fluo.get_psf(na, datamap_pixelsize)
            # should take data_pixelsize instead of pixelsize, right? same for psf above?
            psf_det = self.detector.get_detection_psf(self.fluo.lambda_, psf,
                                                      na, transmission,
                                                      datamap_pixelsize)
            self.__cache[datamap_pixelsize_nm] = utils.resize(i_ex, i_sted, psf_det)

        return self.__cache[datamap_pixelsize_nm]
    
    def clear_cache(self):
        '''Empty the cache.
        
        .. important::
           It is important to empty the cache if any of the components
           :attr:`excitation`, :attr:`sted`, :attr:`detector`,
           :attr:`objective`, or :attr:`fluorescence` are internally modified
           or replaced.
        '''
        self.__cache = {}
    
    def get_effective(self, datamap_pixelsize, p_ex, p_sted):
        '''Compute the detected signal given some molecules disposition.
        
        :param datamap_pixelsize: The size of one pixel of the simulated image (m).
        :param p_ex: The power of the depletion beam (W).
        :param p_sted: The power of the STED beam (W).
        :param data_pixelsize: The size of one pixel of the raw data (m).
        :returns: A 2D array of the effective intensity (W) of a single molecule.
        
        The technique follows the method and equations described in
        [Willig2006]_, [Leutenegger2010]_ and [Holler2011]_.
        '''

        h, c = scipy.constants.h, scipy.constants.c
        f, n, na = self.objective.f, self.objective.n, self.objective.na
        
        __i_ex, __i_sted, psf_det = self.cache(datamap_pixelsize)
        i_ex = __i_ex * p_ex
        i_sted = __i_sted * p_sted
        
        # saturation intensity (W/m²) [Leutenegger2010] p. 26419
        sigma_ste = self.fluo.get_sigma_ste(self.sted.lambda_)
        i_s = (h * c) / (self.fluo.tau * self.sted.lambda_ * sigma_ste)
        
        # [Leutenegger2010] eq. 3
        zeta = i_sted / i_s
        k_vib = 1 / self.fluo.tau_vib
        k_s1 = 1 / self.fluo.tau
        gamma = (zeta * k_vib) / (zeta * k_s1 + k_vib)
        T = 1 / self.sted.rate
        # probability of fluorescence given the donut
        eta = (((1 + gamma * numpy.exp(-k_s1 * self.sted.tau * (1 + gamma))) / (1 + gamma)) -
              numpy.exp(-k_s1 * (gamma * self.sted.tau + T))) / (1 - numpy.exp(-k_s1 * T))
        
        # molecular brigthness [Holler2011]
        sigma_abs = self.fluo.get_sigma_abs(self.excitation.lambda_)
        excitation_probability = sigma_abs * i_ex * self.fluo.qy

        # effective intensity of a single molecule (W) [Willig2006] eq. 3
        return excitation_probability * eta * psf_det

    def laser_dans_face(self, pixelsize, pixel_list=None):
        """
        Test function to visualize how much laser each pixel receives
        :param pixelsize: Grid size for the laser movement. Has to be a multiple of datamap_obj.datamap_pixelsize. (m)
        :param pixel_list: List of pixels on which the laser will be applied. If None, a normal raster scan of every
                           pixel will be done.
        :returns: laser_received, an array containing the quantity of laser received per pixel,
                  sampled, an array containing the number of times each pixel has been covered by the laser array
        """
        datamap_roi = self.datamap.whole_datamap[self.datamap.roi]
        pixel_list = utils.pixel_list_filter(datamap_roi, pixel_list, pixelsize, self.datamap.pixelsize)

        i_ex, i_sted, psf_det = self.cache(self.datamap.pixelsize)

        laser_received = numpy.zeros(self.datamap.whole_datamap.shape)
        sampled = numpy.zeros(self.datamap.whole_datamap.shape)
        rows_pad, cols_pad = self.datamap.roi_corners['tl'][0], self.datamap.roi_corners['tl'][1]
        laser_pad = self.datamap.laser.shape[0] // 2
        pdt_roi = self.datamap.pdt[self.datamap.roi]
        p_ex_roi = self.datamap.p_ex[self.datamap.roi]
        p_sted_roi = self.datamap.p_sted[self.datamap.roi]

        for (row, col) in pixel_list:
            laser_applied = (i_ex * p_ex_roi[row, col] + i_sted * p_sted_roi[row, col]) * pdt_roi[row, col]
            sampled[row + rows_pad - laser_pad: row + rows_pad + laser_pad + 1,
                    col + cols_pad - laser_pad: col + cols_pad + laser_pad + 1] += 1
            laser_received[row + rows_pad - laser_pad: row + rows_pad + laser_pad + 1,
                           col + cols_pad - laser_pad: col + cols_pad + laser_pad + 1] += laser_applied

        return laser_received, sampled

    def get_signal_and_bleach(self, datamap, pixelsize, pdt, p_ex, p_sted, pixel_list=None, bleach=True, update=True):
        """
        Function to bleach the datamap as the signal is acquired.
        :param datamap: A Datamap object containing the relevant molecule disposition information, pixel size and ROI.
        :param pixelsize: Grid size for the laser movement. Has to be a multiple of datamap.pixelsize. (m)
        :param pdt: Time spent by the lasers on each pixel. Can be a float or an array of floats of same shape as
                    as the datamap ROI (datamap.whole_datamap[datamap.roi]) (s)
        :param p_ex: Power of the excitation beam. Can be a float or an array of floats of same shape as
                     as the datamap ROI (datamap.whole_datamap[datamap.roi]) (W)
        :param p_sted: Power of the STED beam. Can be a float or an array of floats of same shape as
                       as the datamap ROI (datamap.whole_datamap[datamap.roi])(W)
        :param pixel_list: List of pixels on which the laser will be applied. If None, a normal raster scan of every
                           pixel of the ROI will be done. Set to None by default.
        :param bleach: A bool which determines whether or not bleaching will occur. Set to True by default.
        :param update: A bool which determines whether or not the Datamap object will be updated with the bleaching.
                       Set to True by default.
        :returns: The acquired detected photons, and the bleached datamap.
        """
        datamap_pixelsize = datamap.pixelsize
        i_ex, i_sted, psf_det = self.cache(datamap_pixelsize)
        if datamap.roi is None:
            # demander au dude de setter une roi
            datamap.set_roi(i_ex)

        datamap_roi = datamap.whole_datamap[datamap.roi]
        pdt = utils.float_to_array_verifier(pdt, datamap_roi.shape)
        p_ex = utils.float_to_array_verifier(p_ex, datamap_roi.shape)
        p_sted = utils.float_to_array_verifier(p_sted, datamap_roi.shape)

        pixel_list = utils.pixel_list_filter(datamap_roi, pixel_list, pixelsize, datamap_pixelsize)

        # effective = self.get_effective(datamap_pixelsize, p_ex, p_sted)

        ratio = utils.pxsize_ratio(pixelsize, datamap_pixelsize)
        acquired_intensity = numpy.zeros((int(numpy.ceil(datamap_roi.shape[0] / ratio)),
                                          int(numpy.ceil(datamap_roi.shape[1] / ratio))))
        rows_pad, cols_pad = datamap.roi_corners['tl'][0], datamap.roi_corners['tl'][1]
        laser_pad = i_ex.shape[0] // 2

        prob_ex = numpy.ones(datamap.whole_datamap.shape)
        prob_sted = numpy.ones(datamap.whole_datamap.shape)
        bleached_datamap = numpy.copy(datamap.whole_datamap)

        for (row, col) in pixel_list:
            effective = self.get_effective(datamap_pixelsize, p_ex[row, col], p_sted[row, col])
            row_slice = slice(row + rows_pad - laser_pad, row + rows_pad + laser_pad + 1)
            col_slice = slice(col + cols_pad - laser_pad, col + cols_pad + laser_pad + 1)
            acquired_intensity[int(row / ratio), int(col / ratio)] = numpy.sum(effective *
                                                                                bleached_datamap
                                                                                [row_slice, col_slice])

            if bleach is True:
                kwargs = {'i_ex': i_ex, 'i_sted': i_sted, 'fluo': self.fluo, 'excitation': self.excitation,
                          'sted': self.sted, 'p_ex': p_ex[row, col], 'p_sted': p_sted[row, col],
                          'pdt': pdt[row, col], 'prob_ex': prob_ex, 'prob_sted': prob_sted,
                          'region': (row_slice, col_slice)}
                prob_ex, prob_sted = self.bleach_func(**kwargs)
                bleached_datamap[row_slice, col_slice] = \
                    numpy.random.binomial(bleached_datamap[row_slice, col_slice],
                                          prob_ex[row_slice, col_slice] * prob_sted[row_slice, col_slice])

        # Bleaching is done, the rest is for intensity calculation
        photons = self.fluo.get_photons(acquired_intensity)

        if photons.shape == pdt.shape:
            returned_intensity = self.detector.get_signal(photons, pdt)
        else:
            pixeldwelltime_reshaped = numpy.zeros((int(numpy.ceil(pdt.shape[0] / ratio)),
                                                   int(numpy.ceil(pdt.shape[1] / ratio))))
            new_pdt_plist = utils.pixel_sampling(pixeldwelltime_reshaped, mode='all')
            for (row, col) in new_pdt_plist:
                pixeldwelltime_reshaped[row, col] = pdt[row * ratio, col * ratio]
            returned_intensity = self.detector.get_signal(photons, pixeldwelltime_reshaped)

        if update:
            datamap.whole_datamap = bleached_datamap

        return returned_intensity, bleached_datamap

    def get_signal_and_bleach_fast(self, datamap, pixelsize, pdt, p_ex, p_sted, pixel_list=None, bleach=True,
                                   update=True, seed=None, filter_bypass=False):
        """
        MODIFY ANTHO'S VERSION TO WORK WITH THE GOOD DATAMAP / MICROSCOPE IMPLEMENTATION
        Function to bleach the datamap as the signal is acquired.
        :param pixelsize: Grid size for the laser movement. Has to be a multiple of datamap_obj.datamap_pixelsize. (m)
        :param p_ex: Power of the excitation beam. (W)
        :param p_sted: Power of the STED beam. (W)
        :param pixel_list: List of pixels on which the laser will be applied. If None, a normal raster scan of every
                           pixel of the ROI will be done.
        :param bleach: A bool which determines whether or not bleaching wil occur
        :returns: An array with the acquired pixelwise intensities, and the updated (bleached) datamap_obj
        """
        if seed is not None:
            numpy.random.seed(seed)
        datamap_pixelsize = datamap.pixelsize
        i_ex, i_sted, psf_det = self.cache(datamap_pixelsize)
        if datamap.roi is None:
            # demander au dude de setter une roi
            datamap.set_roi(i_ex)

        datamap_roi = datamap.whole_datamap[datamap.roi]
        pdt = utils.float_to_array_verifier(pdt, datamap_roi.shape)

        powers_array_verifier = False
        if type(p_ex) is numpy.ndarray or type(p_sted) is numpy.ndarray:
            if bleach:
                raster_func = raster.raster_func_c_self_bleach
            else:
                raster_func = raster.raster_func_c_self
            powers_array_verifier = True
        else:
            if bleach:
                raster_func = raster.raster_func_wbleach_c
            else:
                raster_func = raster.raster_func_c
        p_ex = utils.float_to_array_verifier(p_ex, datamap_roi.shape)
        p_sted = utils.float_to_array_verifier(p_sted, datamap_roi.shape)

        if not filter_bypass:
            pixel_list = utils.pixel_list_filter(datamap_roi, pixel_list, pixelsize, datamap_pixelsize)

        # TODO: need to find a way to compute this inside the C function on a pixel per pixel basis
        effective = self.get_effective(datamap_pixelsize, p_ex[0, 0], p_sted[0, 0])

        ratio = utils.pxsize_ratio(pixelsize, datamap_pixelsize)
        acquired_intensity = numpy.zeros((int(numpy.ceil(datamap_roi.shape[0] / ratio)),
                                          int(numpy.ceil(datamap_roi.shape[1] / ratio))))
        rows_pad, cols_pad = datamap.roi_corners['tl'][0], datamap.roi_corners['tl'][1]
        laser_pad = i_ex.shape[0] // 2

        # TODO: figure out comment faire ça pour un p_ex, p_sted variable par pixel
        #       un dict qui contient les photons_ex pour chaque (row, col)?? seems dumb, maybe I have no other choice
        photons_ex = self.fluo.get_photons(i_ex * p_ex[0, 0])
        k_ex = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)


        duty_cycle = self.sted.tau * self.sted.rate
        photons_sted = self.fluo.get_photons(i_sted * p_sted[0, 0] * duty_cycle)
        k_sted = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)

        prob_ex = numpy.ones(datamap.whole_datamap.shape)
        prob_sted = numpy.ones(datamap.whole_datamap.shape)
        bleached_datamap = numpy.copy(datamap.whole_datamap)

        if isinstance(raster_func, type(None)):
            for (row, col) in pixel_list:
                effective = self.get_effective(datamap_pixelsize, p_ex[row, col], p_sted[row, col])
                row_slice = slice(row + rows_pad - laser_pad, row + rows_pad + laser_pad + 1)
                col_slice = slice(col + cols_pad - laser_pad, col + cols_pad + laser_pad + 1)
                acquired_intensity[int(row / ratio), int(col / ratio)] = numpy.sum(effective *
                                                                                    bleached_datamap
                                                                                    [row_slice, col_slice])

                if bleach is True:
                    kwargs = {'i_ex': i_ex, 'i_sted': i_sted, 'fluo': self.fluo, 'excitation': self.excitation,
                              'sted': self.sted, 'p_ex': p_ex[row, col], 'p_sted': p_sted[row, col],
                              'pdt': pdt[row, col], 'prob_ex': prob_ex, 'prob_sted': prob_sted,
                              'region': (row_slice, col_slice)}
                    prob_ex, prob_sted = self.bleach_func(**kwargs)
                    bleached_datamap[row_slice, col_slice] = \
                        numpy.random.binomial(bleached_datamap[row_slice, col_slice],
                                              prob_ex[row_slice, col_slice] * prob_sted[row_slice, col_slice])
        else:
            if seed is None:
                seed = 0
            if powers_array_verifier:
                raster_func(
                    self, datamap, acquired_intensity, numpy.array(pixel_list).astype(numpy.int32), ratio, rows_pad,
                    cols_pad, laser_pad, prob_ex, prob_sted, pdt, p_ex, p_sted, bleached_datamap, seed
                )
                pass
            else:
                raster_func(
                    acquired_intensity, numpy.array(pixel_list).astype(numpy.int32),
                    ratio, effective, rows_pad, cols_pad, laser_pad, datamap.whole_datamap,
                    bleach, prob_ex, prob_sted, k_ex, k_sted, pdt, bleached_datamap, seed
                )

        # Bleaching is done, the rest is for intensity calculation
        photons = self.fluo.get_photons(acquired_intensity)

        if photons.shape == pdt.shape:
            returned_intensity = self.detector.get_signal(photons, pdt)
        else:
            pixeldwelltime_reshaped = numpy.zeros((int(numpy.ceil(pdt.shape[0] / ratio)),
                                                   int(numpy.ceil(pdt.shape[1] / ratio))))
            new_pdt_plist = utils.pixel_sampling(pixeldwelltime_reshaped, mode='all')
            for (row, col) in new_pdt_plist:
                pixeldwelltime_reshaped[row, col] = pdt[row * ratio, col * ratio]
            returned_intensity = self.detector.get_signal(photons, pixeldwelltime_reshaped)

        if update:
            datamap.whole_datamap = bleached_datamap

        return returned_intensity, bleached_datamap

    def get_signal_rescue(self, datamap, pixelsize, pdt, p_ex, p_sted, pixel_list=None, bleach=True, update=True,
                          lower_th=1, ltr=0.1, upper_th=100):
        """
        Function to bleach the datamap as the signal is acquired using RESCue method (EN PARLER PLUS ET METTRE UNE
        CITATION AU PAPIER OU QQCHOSE UNE FOIS QUE J'AURAI RELU L'ARTICLE ET TOUT)
        :param datamap: A Datamap object containing the relevant molecule disposition information, pixel size and ROI.
        :param pixelsize: Grid size for the laser movement. Has to be a multiple of datamap.pixelsize. (m)
        :param pdt: Time spent by the lasers on each pixel. Can be a float or an array of floats of same shape as
                    as the datamap ROI (datamap.whole_datamap[datamap.roi]) (s)
        :param p_ex: Power of the excitation beam. Can be a float or an array of floats of same shape as
                     as the datamap ROI (datamap.whole_datamap[datamap.roi]) (W)
        :param p_sted: Power of the STED beam. Can be a float or an array of floats of same shape as
                       as the datamap ROI (datamap.whole_datamap[datamap.roi])(W)
        :param pixel_list: List of pixels on which the laser will be applied. If None, a normal raster scan of every
                           pixel of the ROI will be done. Set to None by default.
        :param bleach: A bool which determines whether or not bleaching will occur. Set to True by default.
        :param update: A bool which determines whether or not the Datamap object will be updated with the bleaching.
                       Set to True by default.
        :param lower_th: The minimum number of photons we wish to detect on a pixel in (pdt * ltr) seconds to determine
                         if we continue illuminating the pixel.
        :param ltr: The ratio of the pdt time in which we will decide whether or not we continue illuminating the pixel.
        :param upper_th: The maximum number of photons we wish to detect on a pixel in (pdt * utr) seconds before moving
                         to the next pixel.
        :param utr: The ratio of the pdt time in which we will decide wether or not enough photons have already been
                    detected in order to move on to the next pixel before ellapsing the whole pdt time.
        :returns: The acquired detected photons, and the bleached datamap.
        """
        #TODO : Pouvoir mettre None à upper_th
        datamap_roi = datamap.whole_datamap[datamap.roi]
        pdt = utils.float_to_array_verifier(pdt, datamap_roi.shape)
        p_ex = utils.float_to_array_verifier(p_ex, datamap_roi.shape)
        p_sted = utils.float_to_array_verifier(p_sted, datamap_roi.shape)

        datamap_pixelsize = datamap.pixelsize
        i_ex, i_sted, psf_det = self.cache(datamap_pixelsize)
        if datamap.roi is None:
            # demander au dude de setter une roi
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

        for (row, col) in pixel_list:
            effective = self.get_effective(datamap_pixelsize, p_ex[row, col], p_sted[row, col])
            row_slice = slice(row + rows_pad - laser_pad, row + rows_pad + laser_pad + 1)
            col_slice = slice(col + cols_pad - laser_pad, col + cols_pad + laser_pad + 1)

            pixel_intensity = numpy.sum(effective * datamap.whole_datamap[row_slice, col_slice])
            pixel_photons = self.detector.get_signal(self.fluo.get_photons(pixel_intensity), pdt[row, col])
            photons_per_sec = pixel_photons / pdt[row, col]
            lt_time_photons = photons_per_sec * ltr * pdt[row, col]
            if lower_th is None:
                pass
            elif lt_time_photons < lower_th:
                pdt[row, col] = ltr * pdt[row, col]
            else:
                if upper_th is None:
                    pass
                elif pixel_photons > upper_th:
                    # jpense pas que je dois mult par utr, je dois mult par le temps requis pour arriver à upper_th
                    time_to_ut = upper_th / photons_per_sec
                    pdt[row, col] = time_to_ut
                else:
                    pass

            # ça serait tu mieux de calculer mon nombre de photons avec le photons_per_sec?
            # i think so right?
            rescue_pixel_photons = photons_per_sec * pdt[row, col]
            # rescue_pixel_photons = self.detector.get_signal(self.fluo.get_photons(pixel_intensity), pdt[row, col])
            returned_photons[int(row / ratio), int(col / ratio)] = rescue_pixel_photons

            if bleach is True:
                kwargs = {'i_ex': i_ex, 'i_sted': i_sted, 'fluo': self.fluo, 'excitation': self.excitation,
                          'sted': self.sted, 'p_ex': p_ex[row, col], 'p_sted': p_sted[row, col],
                          'pdt': pdt[row, col], 'prob_ex': prob_ex, 'prob_sted': prob_sted,
                          'region': (row_slice, col_slice)}
                prob_ex, prob_sted = self.bleach_func(**kwargs)
                bleached_datamap[row_slice, col_slice] = \
                    numpy.random.binomial(bleached_datamap[row_slice, col_slice],
                                          prob_ex[row_slice, col_slice] * prob_sted[row_slice, col_slice])

        if update:
            datamap.whole_datamap = bleached_datamap

        return returned_photons, bleached_datamap

    def add_to_pixel_bank(self, n_pixels_per_tstep):
        """
        Adds the residual pixels to the pixel bank
        :param n_pixels_per_tstep: The number of pixels which the microscope has the time to acquire during 1
                                   time step of the Ca2+ flash event
        """
        integer_n_pixels_per_tstep = int(n_pixels_per_tstep)
        self.pixel_bank += n_pixels_per_tstep - integer_n_pixels_per_tstep

    def take_from_pixel_bank(self):
        """

        :return:
        """
        pass


class Datamap:
    """This class implements a datamap

    :param whole_datamap: The disposition of the molecules in the sample. This represents the whole sample, from which
                          only a region will be imaged (roi). (numpy array)
    :param roi: A dict of tuples containing the rows and columns intervals for the ROI to image. The size of the ROI is
                limited by the size of the lasers (and thus by the datamap_pixelsize).
    :param datamap_pixelsize: The size of a pixel of the datamap. (m)
    """

    def __init__(self, whole_datamap, datamap_pixelsize):
        self.whole_datamap = numpy.copy(whole_datamap.astype(numpy.int32))
        self.whole_shape = self.whole_datamap.shape
        self.pixelsize = datamap_pixelsize
        self.roi = None
        self.roi_corners = None

    def set_roi(self, laser, intervals=None):
        """
        Uses a laser generated by the microscope object to determine the biggest ROI allowed, sets the ROI if valid
        :param laser: An array of the same shape as the lasers which will be used on the datamap
        :param intervals: Values to set the ROI to. Either 'max', a dict like {'rows': [min_row, max_row],
                          'cols': [min_col, max_col]} or None. If 'max', the whole datamap will be padded with 0s, and
                          the original array will be used as ROI. If None, will prompt the user to enter an ROI.
        """
        rows_min, cols_min = laser.shape[0] // 2, laser.shape[1] // 2
        rows_max, cols_max = self.whole_datamap.shape[0] - rows_min - 1, self.whole_datamap.shape[1] - cols_min - 1
        # il faut que je gère le cas où la datamap est trop petite et que je n'ai pas le choix de padder de 0
        # assumer qu'il voudra itérer sur tout la datamap alors

        if intervals is None:
            # l'utilisateur n'a pas définit de ROI, on lui demande de la définir ici
            print(f"ROI must be within rows [{rows_min}, {rows_max}] inclusively, "
                  f"columns must be within [{cols_min}, {cols_max}] inclusively")
            roi = {'rows': None, 'cols': None}
            rows_start = int(input(f"Enter a starting row : "))
            rows_end = int(input(f"Enter an ending row : "))
            cols_start = int(input(f"Enter a starting column : "))
            cols_end = int(input(f"Enter an ending column : "))
            roi['rows'] = [rows_start, rows_end]
            roi['cols'] = [cols_start, cols_end]

            if roi['rows'][0] < rows_min or roi['rows'][0] > rows_max or \
               roi['rows'][1] < rows_min or roi['rows'][1] > rows_max or \
               roi['cols'][0] < cols_min or roi['cols'][0] > cols_max or \
               roi['cols'][1] < cols_min or roi['cols'][1] > cols_max:
                raise ValueError(f"ROI missplaced for datamap of shape {self.whole_datamap.shape} with lasers of shape"
                                 f"{laser.shape}. ROI intervals must be within bounds "
                                 f"rows:[{rows_min}, {rows_max}], cols:[{cols_min}, {cols_max}].")

            self.roi = (slice(roi['rows'][0], roi['rows'][1] + 1), slice(roi['cols'][0], roi['cols'][1] + 1))
            # Having a slice object is useful, but having the idx values of the 4 corners is also useful
            self.roi_corners = {'tl': (roi['rows'][0], roi['cols'][0]), 'tr': (roi['rows'][0], roi['cols'][1]),
                                'bl': (roi['rows'][1], roi['cols'][0]), 'br': (roi['rows'][1], roi['cols'][1])}

        elif intervals == 'max':
            # l'utilisateur veut itérer sur la whole_datamap, alors la padder de 0 comme normal
            self.whole_datamap, rows_pad, cols_pad = utils.array_padder(self.whole_datamap, laser)
            # def mes 4 coins et ma slice
            self.roi = (slice(rows_pad, self.whole_datamap.shape[0] - rows_pad),
                        slice(cols_pad, self.whole_datamap.shape[1] - cols_pad))
            self.roi_corners = {'tl': (rows_pad, cols_pad),
                                'tr': (rows_pad, self.whole_datamap.shape[1] - cols_pad - 1),
                                'bl': (self.whole_datamap.shape[0] - rows_pad - 1, cols_pad),
                                'br': (self.whole_datamap.shape[0] - rows_pad - 1,
                                       self.whole_datamap.shape[1] - cols_pad - 1)}

        elif type(intervals) is dict:
            # j'assume que l'utilisateur a passé un dictionnaire avec roi['rows'] et roi['cols'] comme intervalles
            if intervals['rows'][0] < rows_min or intervals['rows'][0] > rows_max or \
               intervals['rows'][1] < rows_min or intervals['rows'][1] > rows_max or \
               intervals['cols'][0] < cols_min or intervals['cols'][0] > cols_max or \
               intervals['cols'][1] < cols_min or intervals['cols'][1] > cols_max:
                raise ValueError(f"ROI missplaced for datamap of shape {self.whole_datamap.shape} with lasers of shape"
                                 f"{laser.shape}. ROI intervals must be within bounds "
                                 f"rows:[{rows_min}, {rows_max}], cols:[{cols_min}, {cols_max}].")
            self.roi_corners = {'tl': (intervals['rows'][0], intervals['cols'][0]),
                                'tr': (intervals['rows'][0], intervals['cols'][1]),
                                'bl': (intervals['rows'][1], intervals['cols'][0]),
                                'br': (intervals['rows'][1], intervals['cols'][1])}
            self.roi = (slice(self.roi_corners['tl'][0], self.roi_corners['bl'][0] + 1),
                        slice(self.roi_corners['tl'][1], self.roi_corners['br'][1] + 1))
        else:
            raise ValueError("intervals parameter must be either None, 'max' or dict")

    def set_bleached_datamap(self, bleached_datamap):
        """
        This functions updates the datamap.whole_datamap attribute to the bleached version. I put this in case the user
        does not want to update the datamap after bleaching it in order to do multiple experiments on the same setting.
        :param bleached_datamap: An array of the datamap after the lasers have passed over it (after it has bleached).
                                 Has to be of the same shape as self.whole_datamap
        """
        if bleached_datamap.shape != self.whole_datamap.shape:
            raise ValueError("Bleached datamap to set as new datamap has to be of the same shape as the datamap pre "
                             "bleaching.")
        self.whole_datamap = bleached_datamap

    def test_iter_over_roi(self, laser):
        """
        The goal of this function is just to pass the laser over the whole ROI to make sure it fits right :)
        """
        ones_laser = numpy.ones(laser.shape)
        acquisition_pad, rows_pad, cols_pad = utils.array_padder(numpy.zeros(self.whole_datamap[self.roi].shape),
                                                                 ones_laser)
        pixel_list = utils.pixel_sampling(self.whole_datamap[self.roi])
        for (row, col) in pixel_list:
            acquisition_pad[row:row+2*rows_pad+1, col:col+2*cols_pad+1] += ones_laser

        return utils.array_unpadder(acquisition_pad, ones_laser)

    def add_sphere(self, width, position, max_molecs=3, randomness=1, distribution="random"):
        """
        Function to add a sphere containing molecules at a certain position
        *** Probably need to revisit this if, if I even want to keep it ***
        """
        valid_distributions = ["random", "gaussian", "periphery"]
        if distribution not in valid_distributions:
            print(f"Wrong distribution choice, retard")
            print(f"Valid distributions are : ")
            for possibilites in valid_distributions:
                print(possibilites)
            raise Exception("Invalid distribution choice")

        pixels_width = utils.pxsize_ratio(width, self.pixelsize)

        # padder la datamap pour m'assurer que je puisse ajouter une sphère en périphérie de l'img aussi
        pad = pixels_width
        padded_molecules = numpy.pad(self.whole_datamap, pad, mode="constant")

        if distribution == "random":
            for i in range(0, 360):
                x = pixels_width / 2 * numpy.cos(i * numpy.pi / 180)
                y = pixels_width / 2 * numpy.sin(i * numpy.pi / 180)

                area_covered_shape = padded_molecules[int(pad + position[0] - x): int(x + pad + position[0]),
                                                      int(y + pad + position[1])].shape
                molecs_to_place = numpy.ones(area_covered_shape[0]).astype(numpy.int) * max_molecs

                probability = randomness

                padded_molecules[int(pad + position[0] - x): int(x + pad + position[0]), int(y + pad + position[1])] = \
                    numpy.random.binomial(molecs_to_place, probability)

        elif distribution == "gaussian":
            x, y = numpy.meshgrid(numpy.linspace(-1, 1, pixels_width),
                                  numpy.linspace(-1, 1, pixels_width))
            d = numpy.sqrt(x*x+y*y)
            sigma, mu = randomness, 0.0
            g = numpy.exp(-((d-mu)**2 / (2.0 * sigma**2)))

            probabilities = numpy.zeros(padded_molecules.shape)
            probabilities[position[0] + pad // 2: position[0] + pad + pad // 2,
                          position[1] + pad // 2: position[1] + pad + pad // 2] = g

            for i in range(0, 360):
                x = pixels_width / 2 * numpy.cos(i * numpy.pi / 180)
                y = pixels_width / 2 * numpy.sin(i * numpy.pi / 180)

                area_covered_shape = padded_molecules[int(pad + position[0] - x): int(x + pad + position[0]),
                                                      int(y + pad + position[1])].shape
                molecs_to_place = numpy.ones(area_covered_shape[0]).astype(numpy.int) * max_molecs

                padded_molecules[int(pad + position[0] - x): int(x + pad + position[0]), int(y + pad + position[1])] = \
                    numpy.random.binomial(molecs_to_place,
                                          probabilities[int(pad + position[0] - x): int(x + pad + position[0]),
                                                        int(y + pad + position[1])])

        elif distribution == "periphery":
            x, y = numpy.meshgrid(numpy.linspace(-1, 1, pixels_width),
                                  numpy.linspace(-1, 1, pixels_width))
            d = numpy.sqrt(x*x+y*y)
            sigma, mu = randomness, 0.0
            g = numpy.exp(-((d-mu)**2 / (2.0 * sigma**2)))

            probabilities = numpy.zeros(padded_molecules.shape)
            probabilities[position[0] + pad // 2: position[0] + pad + pad // 2,
                          position[1] + pad // 2: position[1] + pad + pad // 2] = g
            probabilities = 1 - probabilities

            for i in range(0, 360):
                x = pixels_width / 2 * numpy.cos(i * numpy.pi / 180)
                y = pixels_width / 2 * numpy.sin(i * numpy.pi / 180)

                area_covered_shape = padded_molecules[int(pad + position[0] - x): int(x + pad + position[0]),
                                                      int(y + pad + position[1])].shape
                molecs_to_place = numpy.ones(area_covered_shape[0]).astype(numpy.int) * max_molecs

                padded_molecules[int(pad + position[0] - x): int(x + pad + position[0]), int(y + pad + position[1])] = \
                    numpy.random.binomial(molecs_to_place,
                                          probabilities[int(pad + position[0] - x): int(x + pad + position[0]),
                                                        int(y + pad + position[1])])

        # ligne pour un-pad
        self.whole_datamap = padded_molecules[pad:-pad, pad:-pad]


def load_experiment_parameters(path, molecule_disposition):
    """
    This function handles the generation of all the objects required for an experiment, which are :
    GaussianBeam
    DonutBeam
    Detector
    Objective
    Fluorescence
    Microscope
    Datamap
    It loads the information required to generate these objects from a .txt (?) file.
    :param path: Path to .txt (?) file used to generate the experiment objects
    :param molecule_disposition: An array representing the disposition of molecules :)
    :returns: The generated objects required for an experiment
    """
    config = configparser.ConfigParser()
    config.read(path)

    exc_args = {s: float(config['GaussianBeam'][s]) for s in config['GaussianBeam']}

    sted_args = {s: float(config['DonutBeam'][s]) for s in config['DonutBeam']}

    detector_noise = bool(config['Detector']['noise'])
    config['Detector']['noise'] = '0'
    detector_args = {s: float(config['Detector'][s]) for s in config['Detector']}
    detector_args['noise'] = detector_noise

    objective_transmission = ast.literal_eval(config['Objective']['transmission'])
    config['Objective']['transmission'] = '0'
    objective_args = {s: float(config['Objective'][s]) for s in config['Objective']}
    objective_args['transmission'] = objective_transmission


    fluo_dict_args, fluo_dicts = ['sigma_ste', 'sigma_abs', 'phy_react'], {}
    for dict_arg in fluo_dict_args:
        fluo_dicts[dict_arg] = config['Fluorescence'][dict_arg]
        config['Fluorescence'][dict_arg] = '0'
    fluo_args = {s: float(config['Fluorescence'][s]) for s in config['Fluorescence']}
    for dict_arg in fluo_dict_args:
        fluo_args[dict_arg] = fluo_dicts[dict_arg]

    microscope_args = config['Microscope']

    datamap_args = {'pixelsize': float(config['Datamap']['pixelsize']), 'roi': config['Datamap']['roi']}

    laser_ex = GaussianBeam(**exc_args)
    laser_sted = DonutBeam(**sted_args)
    detector = Detector(**detector_args)
    objective = Objective(**objective_args)
    fluo = Fluorescence(**fluo_args)
    microscope = Microscope(laser_ex, laser_sted, detector, objective, fluo, microscope_args['bleach_func'])
    datamap = Datamap(molecule_disposition, datamap_args['pixelsize'])
    i_ex, _, _ = microscope.cache(datamap.pixelsize)
    datamap.set_roi(i_ex, datamap_args['roi'])

    return laser_ex, laser_sted, detector, objective, fluo, microscope, datamap