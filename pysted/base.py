
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

    signal, _ = microscope.get_signal_and_bleach(data_map, 10e-9, pdt, p_ex, p_sted)

    from matplotlib import pyplot
    pyplot.imshow(signal)
    pyplot.colorbar()
    pyplot.show()

Code written by Benoit Turcotte, Albert Michaud-Gagnon, Anthony Bilodeau, Audrey Durand, and Flavie Lavoie-Cardinal

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

.. [Oracz2017] Oracz, Joanna, et al. (2017)
    Photobleaching in STED Nanoscopy and Its Dependence on the Photon Flux
    Applied for Reversible Silencing of the Fluorophore.
    Scientific Reports, vol. 7, no. 1, Sept. 2017, p. 11354.
    www.nature.com.

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

import logging
import numpy
import scipy.constants
import scipy.signal
import pickle
import copy
import os

from pysted import utils, cUtils, raster, bleach_funcs

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

        :param power: The time averaged power of the beam (W).
        :param f: The focal length of the objective (m).
        :param n: The refractive index of the objective.
        :param na: The numerical aperture of the objective.
        :param transmission: The transmission ratio of the objective (given the
                             wavelength of the excitation beam).
        :param datamap_pixelsize: The size of an element in the intensity matrix (m).
        :return: A 2D array of the time averaged intensity (W/m^2).
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
        intensity /= (intensity*datamap_pixelsize**2).sum()

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
        return intensity_flipped * transmission * power

    def __eq__(self, other):
        """
        Overloads the equal method of the `GaussianBeam` object. Two `GaussianBeam`
        objects are equal if all of their constituent are equals.

        :param other: Any type of python objects

        :return : A `bool` wheter both objects are equal
        """
        if not isinstance(other, GaussianBeam):
            return False
        return all([
            getattr(self, key) == getattr(other, key) for key in vars(self).keys()
        ])

    def __ne__(self, other):
        """
        Overloads the not equal method of the `GaussianBeam` object. Two `GaussianBeam`
        objects are not equal if not all of their constituent are equals.

        :param other: Any type of python objects

        :return : A `bool` wheter both objects are not equal
        """
        return not self == other

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
    | ``tau``          | ``400e-12``  | The beam pulse length (s).             |
    +------------------+--------------+----------------------------------------+
    | ``rate``         | ``40e6``     | The beam pulse rate (Hz).              |
    +------------------+--------------+----------------------------------------+
    | ``zero_residual``| ``0``        | The ratio between minimum and maximum  |
    |                  |              | intensity (ratio).                     |
    +------------------+--------------+----------------------------------------+
    | ``anti_stoke``   | ``True``     | Presence of anti-stoke (sted beam)     |
    |                  |              | excitation                             |
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
        self.tau = kwargs.get("tau", 400e-12)
        self.rate = kwargs.get("rate", 40e6)
        self.zero_residual = kwargs.get("zero_residual", 0)
        self.anti_stoke = kwargs.get("anti_stoke", True)

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
        :return: A 2D array of the instant intensity (W/m^2).
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
        intensity /= (intensity*datamap_pixelsize**2).sum()

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
        intensity *= transmission * power

        if power > 0:
            # zero_residual ~= min(intensity) / max(intensity)
            old_max = numpy.max(intensity)
            intensity += self.zero_residual * old_max
            intensity /= numpy.max(intensity)
            intensity *= old_max

        return intensity

    def __eq__(self, other):
        """
        Overloads the equal method of the `DonutBeam` object. Two `DonutBeam`
        objects are equal if all of their constituent are equals.

        :param other: Any type of python objects

        :return : A `bool` wheter both objects are equal
        """
        if not isinstance(other, DonutBeam):
            return False
        return all([
            getattr(self, key) == getattr(other, key) for key in vars(self).keys()
        ])

    def __ne__(self, other):
        """
        Overloads the not equal method of the `DonutBeam` object. Two `DonutBeam`
        objects are not equal if not all of their constituent are equals.

        :param other: Any type of python objects

        :return : A `bool` wheter both objects are not equal
        """
        return not self == other

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
    | ``noise``        | ``True``     | Whether to add poisson noise to the    |
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
    | ``det_delay``    | ``750e-12``  | Delay between the beginning of a period|
    |                  |              | the start of the detection             |
    +------------------+--------------+----------------------------------------+
    | ``det_width``    | ``8e-9``     | Detection duration                     |
    +------------------+--------------+----------------------------------------+


    .. [#] The actual number is sampled from a poisson distribution with given
       mean.

    .. [#] Excelitas Technologies. (2011). Photon Detection Solutions.
    '''

    def __init__(self, **kwargs):
        # detection pinhole
        self.n_airy = kwargs.get("n_airy", 0.7)

        # detection noise
        self.noise = kwargs.get("noise", True)
        self.background = kwargs.get("background", 0)
        self.darkcount = kwargs.get("darkcount", 0)

        # photon detection
        self.pcef = kwargs.get("pcef", 0.1)
        self.pdef = kwargs.get("pdef", 0.5)

        # Gating
        self.det_delay = kwargs.get("det_delay", 750e-12)
        self.det_width = kwargs.get("det_width", 8e-9)
        assert self.det_delay >= 0 #Verify the detection delay is not negative

    def get_detection_psf(self, lambda_, psf, na, transmission, datamap_pixelsize):
        '''Compute the detection PSF as a convolution between the fluorscence
        PSF and a pinhole, as described by the equation from [Willig2006]_. 
        
        The pinhole radius is determined using the :attr:`n_airy`, the fluorescence 
        wavelength, and the numerical aperture of the objective.

        :param lambda_: The fluorescence wavelength (m).
        :param psf: The fluorescence PSF that can the obtained using
                    :meth:`~pysted.base.Fluorescence.get_psf`.
        :param na: The numerical aperture of the objective.
        :param transmission: The transmission ratio of the objective for the
                             given fluorescence wavelength *lambda_*.
        :param datamap_pixelsize: The size of a pixel in the simulated image (m).
        :return: A 2D array.
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

    def get_signal(self, photons, dwelltime, rate, seed=None):
        '''Compute the detected signal (in photons) given the number of emitted
        photons and the time spent by the detector.

        :param photons: An array of number of emitted photons.
        :param dwelltime: The time spent to detect the emitted photons (s). It is
                          either a scalar or an array shaped like *nb_photons*.
        :return: An array shaped like *nb_photons*.
        '''
        if isinstance(photons, int):
            photons = numpy.array([photons], dtype=numpy.int64)

        detection_efficiency = self.pcef * self.pdef # ratio
        if seed is None:
            # On Windows this seems to be causing some problems when get_signal
            # is called repeatedly since time_ns may not be fast enough...
            # Leaving the seed to None, seems to do the trick.
            # seed = int(str(time.time_ns())[-5:-1])
            # numpy.random.seed(seed)
            pass
        else:
            numpy.random.seed(seed)
        try:
            signal = numpy.random.binomial(photons.astype(numpy.int64),
                                           detection_efficiency,
                                           photons.shape) * dwelltime
        except:
            # on Windows numpy.random.binomial cannot generate 64-bit integers
            signal = utils.approx_binomial(photons.astype(numpy.int64),
                                           detection_efficiency,
                                           photons.shape) * dwelltime
        # add noise, background, and dark counts
        if self.noise:
            signal = numpy.random.poisson(signal, signal.shape)
        if self.background > 0:
            # background counts per second, accounting for the detection gating
            cts = numpy.random.poisson(self.background * self.det_width * rate * dwelltime, signal.shape)
            signal += cts
        if self.darkcount > 0:
            # Dark counts per second, accounting for the detection gating
            cts = numpy.random.poisson(self.darkcount * self.det_width * rate * dwelltime, signal.shape)
            signal += cts
        return signal

    def __eq__(self, other):
        """
        Overloads the equal method of the `Detector` object. Two `Detector`
        objects are equal if all of their constituent are equals.

        :param other: Any type of python objects

        :return : A `bool` wheter both objects are equal
        """
        if not isinstance(other, Detector):
            return False
        return all([
            getattr(self, key) == getattr(other, key) for key in vars(self).keys()
        ])

    def __ne__(self, other):
        """
        Overloads the not equal method of the `Detector` object. Two `Detector`
        objects are not equal if not all of their constituent are equals.

        :param other: Any type of python objects

        :return : A `bool` wheter both objects are not equal
        """
        return not self == other

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
                                                        575: 0.85,
                                                        })

    def get_transmission(self, lambda_):
        return self.transmission[int(lambda_ * 1e9)]

    def __eq__(self, other):
        """
        Overloads the equal method of the `Objective` object. Two `Objective`
        objects are equal if all of their constituent are equals.

        :param other: Any type of python objects

        :return : A `bool` wheter both objects are equal
        """
        if not isinstance(other, Objective):
            return False
        return all([
            getattr(self, key) == getattr(other, key) for key in vars(self).keys()
        ])

    def __ne__(self, other):
        """
        Overloads the not equal method of the `Objective` object. Two `Objective`
        objects are not equal if not all of their constituent are equals.

        :param other: Any type of python objects

        :return : A `bool` wheter both objects are not equal
        """
        return not self == other


class Fluorescence:
    '''This class implements a fluorescence molecule.

    :param lambda_: The fluorescence wavelength (m).
    :param parameters: One or more parameters as described in the following
                       table, optional.

    +--------------------------+--------------+----------------------------------------+
    | Parameter                | Default [#]_ | Details                                |
    +==========================+==============+========================================+
    | ``sigma_ste``            |``575: 1e-21``| A dictionnary mapping STED wavelengths |
    |                          |              | as integer (nm) to stimulated emission |
    |                          |              | cross-section (m²).                    |
    +--------------------------+--------------+----------------------------------------+
    | ``sigma_abs``            |``488: 3e-20``| A dictionnary mapping excitation       |
    |                          |              | wavelengths as integer (nm) to         |
    |                          |              | absorption cross-section (m²).         |
    +--------------------------+--------------+----------------------------------------+
    | ``tau``                  | ``3e-9``     | The fluorescence lifetime (s).         |
    +--------------------------+--------------+----------------------------------------+
    | ``tau_vib``              | ``1e-12``    | The vibrational relaxation (s).        |
    +--------------------------+--------------+----------------------------------------+
    | ``tau_tri``              | ``5e-6``     | The triplet state lifetime (s).        |
    +--------------------------+--------------+----------------------------------------+
    | ``qy``                   | ``0.6``      | The quantum yield (ratio).             |
    +--------------------------+--------------+----------------------------------------+
    | ``k0``                   | ``0``        | Coefficient of the first first order   |
    |                          |              | term of the photobleaching rate        |
    +--------------------------+--------------+----------------------------------------+
    | ``k1``                   | ``1.3e-15``  | Coefficient of the :math:`b^{th}` first|
    |                          |              | order term of the photobleaching rate  |
    +--------------------------+--------------+----------------------------------------+
    | ``b``                    | ``1.4``      | The intersystem crossing rate (s⁻¹).   |
    +--------------------------+--------------+----------------------------------------+
    | ``triplet_dynamics_frac``| ``0``        | Fraction of the bleaching which is due |
    |                          |              | to the (very long) triplets dynamics.  |
    |                          |              | Caution: not based on rigorous theory  |
    +--------------------------+--------------+----------------------------------------+

    .. [#] EGFP (k1 and b for ATTO647N from [Oracz2017]_)
    '''
    #TODO: note the sources for the egfps defaults?
    def __init__(self, lambda_, **kwargs):
        # psf parameters
        self.lambda_ = lambda_

        self.sigma_ste = kwargs.get("sigma_ste", {575: 1e-21})
        self.sigma_abs = kwargs.get("sigma_abs", {488: 3e-20})
        self.tau = kwargs.get("tau", 3e-9)
        self.tau_vib = kwargs.get("tau_vib", 1e-12)
        self.tau_tri = kwargs.get("tau_tri", 5e-6)
        self.qy = kwargs.get("qy", 0.6)
        self.k0 = kwargs.get("k0", 0)
        self.k1 = kwargs.get("k1", 1.3e-15) #Note: divided by (100**2)**1.4, assuming units where wrong in the paper (cm^2 instead of m^2)
        self.b = kwargs.get("b", 1.4)
        self.triplet_dynamic_frac = kwargs.get("triplet_dynamic_frac", 0)

    def __eq__(self, other):
        """
        Overloads the equal method of the `Fluorescence` object. Two fluorescence
        objects are equal if all of their constituent are equals.

        :param other: Any type of python objects

        :return : A `bool` wheter both objects are equal
        """
        if not isinstance(other, Fluorescence):
            return False
        return all([
            getattr(self, key) == getattr(other, key) for key in vars(self).keys()
        ])

    def __ne__(self, other):
        """
        Overloads the not equal method of the fluorescence object. Two fluorescence
        objects are not equal if not all of their constituent are equals.

        :param other: Any type of python objects

        :return : A `bool` wheter both objects are not equal
        """
        return not self == other

    def get_sigma_ste(self, lambda_):
        '''Return the stimulated emission cross-section of the fluorescence
        molecule given the wavelength.

        :param lambda_: The STED wavelength (m).
        :return: The stimulated emission cross-section (m²).
        '''
        return self.sigma_ste[int(lambda_ * 1e9)]

    def get_sigma_abs(self, lambda_):
        '''Return the absorption cross-section of the fluorescence molecule
        given the wavelength.

        :param lambda_: The STED wavelength (m).
        :return: The absorption cross-section (m²).
        '''
        return self.sigma_abs[int(lambda_ * 1e9)]


    def get_psf(self, na, datamap_pixelsize):
        '''Compute the Gaussian-shaped fluorescence PSF.

        :param na: The numerical aperture of the objective.
        :param datamap_pixelsize: The size of an element in the intensity matrix (m).
        :return: A 2D array.
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

    def get_photons(self, intensity, lambda_=None):
        '''Translate a light intensity to a photon flux.

        :param intensity: Light intensity (:math:`W/m^{-2}`).
        :param lambda_: Wavelenght. If None, default to the emission wavelenght.

        :return: Photon flux (:math:`m^{-2}s^{-1}`).
        '''
        if lambda_ is None:
            lambda_ = self.lambda_
        e_photon = scipy.constants.c * scipy.constants.h / lambda_
        return numpy.floor(intensity / e_photon)

    def get_k_bleach(self, lambda_ex, lambda_sted, phi_ex, phi_sted, tau_sted, tau_rep, dwelltime):
        '''Compute a spatial map of the photobleaching rate.

        The photobleaching rate for a 4 level system from [Oracz2017]_ is used. The
        population of S1+S1*, from where the photobleaching happen, was estimated using
        the simplified model used for eq. 3 in [Leutenegger2010]_. The triplet dynamic
        (dependent on the dwelltime) was estimated from the tree level system rate
        equations 2.14 in [Staudt2009]_, approximating that the population S1 is constant.        

        :param lambda_ex: Wavelength of the the excitation beam (m).
        :param lambda_sted: Wavelength of the STED beam (m).
        :param phi_ex: Spatial map of the excitation photon flux (:math:`m^{-2}s^{-1}`).
        :param phi_sted: Spatial map of the STED photon flux (:math:`m^{-2}s^{-1}`).
        :param tau_sted: STED pulse temporal width (s).
        :param tau_rep: Period of the lasers (s).
        :param dwelltime: Time continuously passed centered on a pixel (s).

        :return: A 2D array of the bleaching rate d(Bleached)/dt (:math:`s^{-1}`).
        '''

        exc_lambda_ = numpy.round(lambda_ex/1e-9)
        sted_lambda_ = numpy.round(lambda_sted/1e-9)
        phi_sted = phi_sted * tau_rep/tau_sted

        # Constants used for [Leutenegger2010] eq. 3
        sigma_ste = self.sigma_ste[sted_lambda_]
        phi_s = 1 / (self.tau * sigma_ste) # Saturation flux, at which k_sted = k_s1
        zeta = phi_sted / phi_s # Saturation factor => k_sted = zeta*k_ex
        k_vib = 1 / self.tau_vib
        k_s1 = 1 / self.tau
        gamma = (zeta * k_vib) / (zeta * k_s1 + k_vib) # Effective saturation factor (takes
        # into account rexcitation of S0_star by sted beam) => S1 decay rate = k1*(1-gamma)

        # Excited fluorophores population assuming an infinitely small
        # pulse and that SO=1 is constant.
        S1_ini = 1-numpy.exp(-self.sigma_abs[exc_lambda_] * phi_ex * tau_rep)
        I_sted = phi_sted * (scipy.constants.c * scipy.constants.h / lambda_sted)

        # Suppl. Eq. 16 from Oracz et al.
        # k is defined by Suppl. Eq. 10 from Oracz et al.: d(Bleached)/dt = k * (S1+S1*)
        k = self.k0*I_sted + self.k1*I_sted**self.b

        # Now, S1 <- S1+S1* (put the two sublevels together, like in [Leutenegger2010])
        # Integral from 0 to tau_sted of k*S1, where S1 = exp(-k_s1*t(1+gamma))
        B =  k * S1_ini * (1 - numpy.exp(-k_s1*tau_sted*(1+gamma))) / (k_s1*(1+gamma))

        # The average bleaching rate over a period
        mean_k_bleach = B / tau_rep

        # Add a approximate triplet dynamic, if any.
        # Based on three level system rate equations 2.14 in [Staudt2009]_,
        # approximating that S1 is constant
        k_tri = 1/self.tau_tri
        dwelltime += 1e-15
        k_dwell = (k_tri*dwelltime + numpy.exp(-k_tri*dwelltime) - 1) / (k_tri*dwelltime)
        mean_k_bleach = mean_k_bleach * ((1-self.triplet_dynamic_frac) + self.triplet_dynamic_frac*k_dwell)

        return mean_k_bleach

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
    :param load_cache: A bool which determines whether or not the microscope's lasers will be generated from scratch
                       (load_cache=False) or if they will be loaded from the previous save (load_cache=True). Generating
                       the lasers from scratch can take a long time (takes longer as the pixel_size decreases), so
                       loading the cache can save time when doing multiple experiments using the same pixel_size.
    '''

    def __init__(self, excitation, sted, detector, objective, fluo, load_cache=False, verbose=False):
        self.excitation = excitation
        self.sted = sted
        self.detector = detector
        self.objective = objective
        self.fluo = fluo
        self.T_det = self.detector.det_delay + self.detector.det_width #Time when the detection stop
        assert self.T_det <= 1 / self.sted.rate # verify the detection window does not end after the period

        # caching system
        self.__cache = {}   # add all the elements used to compute lasers in the cache
        cache_keys = ["lasers", "objective", "excitation", "sted", "fluo"]
        if load_cache:
            if os.path.isfile(".microscope_cache.pkl"):
                try:
                    self.__cache = pickle.load(open(".microscope_cache.pkl", "rb"))
                    flag = False
                    for key, values in self.__cache.items():
                        if not all([ckey in values.keys() for ckey in cache_keys]):
                            flag = True
                    if flag:
                        self.__cache = {}

                except Exception as e:
                    if verbose:
                        logging.warning("-----------------")
                        logging.warning(f"An exception was caught while trying to load the microscope cache")
                        logging.warning(f"The microscope cache will be built...")
                        logging.warning(e)
                        logging.warning("-----------------")

        # This will be used during the acquisition routine to make a better correspondance
        # between the microscope acquisition time steps and the Ca2+ flash time steps
        self.pixel_bank = 0
        self.time_bank = 0

    def __str__(self):
        '''Return a string representation of the microscope setup.
        '''
        return str(self.__cache.keys())

    def is_cached(self, datamap_pixelsize):
        '''Indicate the presence of a cache entry for the given pixel size.

        :param datamap_pixelsize: The size of a pixel in the simulated image (m).

        :return: A boolean.
        '''
        datamap_pixelsize_nm = int(datamap_pixelsize * 1e9)
        return datamap_pixelsize_nm in self.__cache

    def cache(self, datamap_pixelsize, save_cache=False):
        '''Compute and cache the excitation and STED intensities, and the
        fluorescence PSF. 
        
        These intensities are computed with a power of 1 W such that they can 
        serve as a basis to compute intensities with any power.

        :param datamap_pixelsize: The size of a pixel in the simulated image (m).
        :param save_cache: A bool which determines whether or not the lasers will be saved to allow for faster load
                           times for future experiments

        :return: A tuple containing:

                  * A 2D array of the excitation intensity for a power of 1 W;
                  * A 2D array of the STED intensity for a a power of 1 W;
                  * A 2D array of the detection PSF.
        '''

        def compute_lasers(datamap_pixelsize_nm, reset_cache=True):
            """
            We only reset the components in the cache if it is necessary
            """
            if reset_cache:
                self.__cache[datamap_pixelsize_nm] = {}

            f, n, na = self.objective.f, self.objective.n, self.objective.na

            # Verifies excitation laser
            if self.excitation == self.__cache[datamap_pixelsize_nm].get("excitation", None):
                i_ex, _, _ = self.__cache[datamap_pixelsize_nm]["lasers"]
            else:
                transmission = self.objective.get_transmission(self.excitation.lambda_)
                i_ex = self.excitation.get_intensity(1, f, n, na,
                                                     transmission, datamap_pixelsize)

            # Verifies STED laser
            if self.sted == self.__cache[datamap_pixelsize_nm].get("sted", None):
                _, i_sted, _ = self.__cache[datamap_pixelsize_nm]["lasers"]
            else:
                transmission = self.objective.get_transmission(self.sted.lambda_)
                i_sted = self.sted.get_intensity(1, f, n, na,
                                                 transmission, datamap_pixelsize)

            # Verifies fluo
            if self.fluo == self.__cache[datamap_pixelsize_nm].get("fluo", None):
                _, _, psf_det = self.__cache[datamap_pixelsize_nm]["lasers"]
            else:
                transmission = self.objective.get_transmission(self.fluo.lambda_)
                psf = self.fluo.get_psf(na, datamap_pixelsize)
                # should take data_pixelsize instead of pixelsize, right? same for psf above?
                psf_det = self.detector.get_detection_psf(self.fluo.lambda_, psf,
                                                          na, transmission,
                                                          datamap_pixelsize)

            self.__cache[datamap_pixelsize_nm]["lasers"] = utils.resize(i_ex, i_sted, psf_det)
            self.__cache[datamap_pixelsize_nm]["objective"] = self.objective
            self.__cache[datamap_pixelsize_nm]["excitation"] = self.excitation
            self.__cache[datamap_pixelsize_nm]["sted"] = self.sted
            self.__cache[datamap_pixelsize_nm]["fluo"] = self.fluo

        datamap_pixelsize_nm = int(datamap_pixelsize * 1e9)
        if datamap_pixelsize_nm not in self.__cache:
            compute_lasers(datamap_pixelsize_nm, reset_cache=True)
        elif (self.__cache[datamap_pixelsize_nm]["objective"] != self.objective) or \
             (self.__cache[datamap_pixelsize_nm]["excitation"] != self.excitation) or \
             (self.__cache[datamap_pixelsize_nm]["sted"] != self.sted) or \
             (self.__cache[datamap_pixelsize_nm]["fluo"] != self.fluo):
            compute_lasers(datamap_pixelsize_nm, reset_cache=False)

        if save_cache:
            pickle.dump(self.__cache, open(".microscope_cache.pkl", "wb"))
        return self.__cache[datamap_pixelsize_nm]["lasers"]

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
        '''Computes the effective point spread function.
        
        Defined here as the spatial map of time averaged detected power per molecule, 
        taking the sted de-excitation, anti-stoke excitation and the detector properties 
        (detection psf and gating) into account.

        The technique follows the method and equations described in
        [Willig2006]_, [Leutenegger2010]_ and [Holler2011]_. Notable approximations from [Leutenegger2010]_ include the assumption that the excitation pulse width is infinitely small and that the sted pulse is of perfect rectangular shape and starts at the beginning of the period. Also, a two energy levels (plus their vibrational sub-levels) with two rate equations is used. To include the vibrational decay dynamics (parametrized by the vibrational decay rate), an effective saturation factor is used.

        To account for the detection gating, the bounds in the integral from [Leutenegger2010]_ eq. 3 were adjusted.

        Anti-stokes excitation at the beginnning of the period was by added by modeling the sted beam as an infinitely small pulse, similarly to the excitation pulse. This leads to an underestimation of its effect on the detected signal, since excitation by the STED beam near the end of the STED beam, for example, would have less time to be depleted.        

        :param datamap_pixelsize: The size of one pixel of the simulated image (m).
        :param p_ex: The time averaged power of the excitation beam (W).
        :param p_sted: The power of the STED beam (W).
        :param data_pixelsize: The size of one pixel of the raw data (m).

        :return: A 2D array of the intensity (W/molecule)
        '''

        h, c = scipy.constants.h, scipy.constants.c
        f, n, na = self.objective.f, self.objective.n, self.objective.na

        __i_ex, __i_sted, psf_det = self.cache(datamap_pixelsize)
        i_ex = __i_ex * p_ex #the time averaged excitation intensity
        i_sted = __i_sted * p_sted #the instant sted intensity (instant p_sted = p_sted/(self.sted.tau * self.sted.rate))

        # saturation intensity (W/m²) [Leutenegger2010] p. 26419
        sigma_ste = self.fluo.get_sigma_ste(self.sted.lambda_)
        i_s = (h * c) / (self.fluo.tau * self.sted.lambda_ * sigma_ste)


        # Equivalent of [Leutenegger2010] eq. 3, but with detector gating parameters to modify the bounds of the integral
        zeta = i_sted / i_s
        k_vib = 1 / self.fluo.tau_vib
        k_s1 = 1 / self.fluo.tau
        gamma = (zeta * k_vib) / (zeta * k_s1 + k_vib)
        T = 1 / self.sted.rate


        # eta=(probability of fluorescence)/(initial probability of
        # fluorescence) given the donut
        if self.detector.det_delay < self.sted.tau: # The detection starts before the end the sted pulse
            nom = (((numpy.exp(-k_s1 * self.detector.det_delay * (1 + gamma)) \
                  + gamma * numpy.exp(-k_s1 * self.sted.tau * (1 + gamma))) / (1 + gamma)) \
                  - numpy.exp(-k_s1 * (gamma * self.sted.tau + self.T_det)))
        elif self.sted.tau <= self.detector.det_delay: # The detection starts not before the end of the sted pulse
            nom = numpy.exp(-k_s1 * gamma * self.sted.tau)\
                  * (numpy.exp(-k_s1 * self.detector.det_delay)\
                  - numpy.exp(-k_s1 * self.T_det))
        eta = nom/(1 - numpy.exp(-k_s1 * T))


        # molecular brigthness [Holler2011]. This would be the spatial map of
        # time averaged power emitted per molecule if there was no deexcitation
        # caused by the depletion beam
        sigma_abs = self.fluo.get_sigma_abs(self.excitation.lambda_)
        phi_ex = i_ex / ((h*c)/self.excitation.lambda_)
        probexc = (1 - numpy.exp( - sigma_abs * phi_ex * 1/self.sted.rate)) * self.fluo.qy
        excitation_probability = probexc/(1/self.sted.rate  /(h*c/self.sted.lambda_))
        if self.sted.anti_stoke:
            sigma_abs_sted = self.fluo.get_sigma_abs(self.sted.lambda_)
            excitation_probability += sigma_abs_sted * (i_sted * self.sted.tau * self.sted.rate) * self.fluo.qy

        # effective intensity of a single molecule (W) [Willig2006] eq. 3
        return excitation_probability * eta * psf_det

    def get_signal_and_bleach(self, datamap, pixelsize, pdt, p_ex, p_sted, indices=None, acquired_intensity=None,
                              pixel_list=None, bleach=True, update=True, seed=None, filter_bypass=False,
                              bleach_func=bleach_funcs.default_update_survival_probabilities, steps=None,
                              prob_ex=None, prob_sted=None, bleach_mode="default", *args, **kwargs):
        """
        Acquires the signal and bleaches simultaneously. 
        
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

        if seed is not None:
            numpy.random.seed(seed)
        datamap_pixelsize = datamap.pixelsize
        i_ex, i_sted, psf_det = self.cache(datamap_pixelsize)

        # maybe I should just throw an error here instead
        if datamap.roi is None:
            # demander au dude de setter une roi
            datamap.set_roi(i_ex)

        datamap_roi = datamap.whole_datamap[datamap.roi]

        # convert scalar values to arrays if they aren't already arrays
        # C funcs need pre defined types, so in order to only have 1 general case C func, I convert scalars to arrays
        pdt = utils.float_to_array_verifier(pdt, datamap_roi.shape)
        p_ex = utils.float_to_array_verifier(p_ex, datamap_roi.shape)
        p_sted = utils.float_to_array_verifier(p_sted, datamap_roi.shape)

        if not filter_bypass:
            pixel_list = utils.pixel_list_filter(datamap_roi, pixel_list, pixelsize, datamap_pixelsize)

        ratio = utils.pxsize_ratio(pixelsize, datamap_pixelsize)
        if acquired_intensity is None:
            acquired_intensity = numpy.zeros((int(numpy.ceil(datamap_roi.shape[0] / ratio)),
                                              int(numpy.ceil(datamap_roi.shape[1] / ratio))))

        rows_pad, cols_pad = datamap.roi_corners['tl'][0], datamap.roi_corners['tl'][1]
        laser_pad = i_ex.shape[0] // 2

        if prob_ex is None:
            prob_ex = numpy.ones(datamap.whole_datamap.shape)
        if prob_sted is None:
            prob_sted = numpy.ones(datamap.whole_datamap.shape)
        
        bleached_sub_datamaps_dict = {}
        if isinstance(indices, type(None)):
            indices = {"flashes": 0}
        for key in datamap.sub_datamaps_dict:
            bleached_sub_datamaps_dict[key] = numpy.copy(datamap.sub_datamaps_dict[key].astype(numpy.int64))

        if seed is None:
            seed = 0

        if steps is None:
            steps = [pdt]
        else:
            for idx, step in enumerate(steps):
                steps[idx] = utils.float_to_array_verifier(step, datamap_roi.shape)

        raster_func = raster.raster_func_c_self_bleach_split_g
        sample_func = bleach_funcs.sample_molecules
        raster_func(self, datamap, acquired_intensity, numpy.array(pixel_list).astype(numpy.int32), ratio, rows_pad,
                    cols_pad, laser_pad, prob_ex, prob_sted, pdt, p_ex, p_sted, bleach, bleached_sub_datamaps_dict,
                    seed, bleach_func, sample_func, steps)

        # Bleaching is done, the rest is for intensity calculation
        photons = self.fluo.get_photons(acquired_intensity)

        if photons.shape == pdt.shape:
            returned_acquired_photons = self.detector.get_signal(photons, pdt, self.sted.rate, seed=seed)
        else:
            pixeldwelltime_reshaped = numpy.zeros((int(numpy.ceil(pdt.shape[0] / ratio)),
                                                   int(numpy.ceil(pdt.shape[1] / ratio))))
            new_pdt_plist = utils.pixel_sampling(pixeldwelltime_reshaped, mode='all')
            for (row, col) in new_pdt_plist:
                pixeldwelltime_reshaped[row, col] = pdt[row * ratio, col * ratio]
            returned_acquired_photons = self.detector.get_signal(photons, pixeldwelltime_reshaped, self.sted.rate, seed=seed)

        unbleached_whole_datamap = numpy.copy(datamap.whole_datamap)

        if update and bleach:
            datamap.sub_datamaps_dict = bleached_sub_datamaps_dict
            datamap.base_datamap = datamap.sub_datamaps_dict["base"]
            datamap.whole_datamap = numpy.copy(datamap.base_datamap)
            if datamap.contains_sub_datamaps["flashes"] and indices["flashes"] < datamap.flash_tstack.shape[0]:
                if bleach_mode == "default":
                    datamap.bleach_future(indices, bleached_sub_datamaps_dict)
                elif bleach_mode == "proportional":
                    datamap.bleach_future_proportional(indices, bleached_sub_datamaps_dict, unbleached_whole_datamap)

        temporal_acq_elts = {"intensity": acquired_intensity,
                             "prob_ex": prob_ex,
                             "prob_sted": prob_sted}

        return returned_acquired_photons, bleached_sub_datamaps_dict, temporal_acq_elts

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
        Verifies the amount stored in the pixel_bank, return the integer part if greater or equal to 1

        :return: The integer part of the pixel_bank of the microscope
        """
        integer_pixel_bank = int(self.pixel_bank)
        if integer_pixel_bank >= 1:
            self.pixel_bank -= integer_pixel_bank
            return integer_pixel_bank
        else:
            return 0

    def empty_pixel_bank(self):
        """
        Empties the pixel bank
        """
        self.pixel_bank = 0

class Datamap:
    """
    This class implements a datamap, containing a disposition of molecules and a ROI to image.

    The Datamap can be a composition of multiple parts, for instance, a 'base', which is static, a 'flashes' part,
    which represents only the flashes occuring in the Datamap, or a 'diffusing' part, which would represent only
    the moving molecules in the Datamap.
    The ROI represents the portion of the Datamap that will be imaged. Since the microscope's lasers are represented by
    arrays, we must ensure that the laser array's edges are contained within the whole Datamap array for every pixel
    of the ROI. To facilitated this, the ROI can be set to 'max', which will simply 0 pad the passed whole_datamap so
    the laser stays confined when scanning over the pixels of the whole_datamap.

    :param whole_datamap: The disposition of the molecules in the sample. This represents the whole sample, from which
                          only a region will be imaged (roi). (numpy array)
    :param datamap_pixelsize: The size of a pixel of the datamap. (m)

    """

    def __init__(self, whole_datamap, datamap_pixelsize):
        self.whole_datamap = numpy.copy(whole_datamap.astype(numpy.int32))
        self.whole_shape = self.whole_datamap.shape
        self.pixelsize = datamap_pixelsize
        self.roi = None
        self.roi_corners = None
        self.contains_sub_datamaps = {"base": True,
                                      "flashes": False}
        self.sub_datamaps_dict = {}

    def __getitem__(self, key):
        return self.sub_datamaps_dict[key]

    def set_roi(self, laser, intervals=None):
        """
        Sets the Region of Interest for the acquisition. 

        Uses a laser generated by the microscope object to determine the biggest ROI allowed, sets the ROI if valid

        :param laser: An array of the same shape as the lasers which will be used on the datamap
        :param intervals: Values to set the ROI to. Either 'max', a dict like {'rows': [min_row, max_row],
                          'cols': [min_col, max_col]} or None. If 'max', the whole datamap will be padded with 0s, and
                          the original array will be used as ROI. If None, will prompt the user to enter an ROI.
        """
        rows_min, cols_min = laser.shape[0] // 2, laser.shape[1] // 2
        rows_max, cols_max = self.whole_datamap.shape[0] - rows_min - 1, self.whole_datamap.shape[1] - cols_min - 1

        if intervals is None:
            # User did not provide intervals
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
            # User wants the maximal interval
            self.whole_datamap, rows_pad, cols_pad = utils.array_padder(self.whole_datamap, laser)
            self.roi = (slice(rows_pad, self.whole_datamap.shape[0] - rows_pad),
                        slice(cols_pad, self.whole_datamap.shape[1] - cols_pad))
            self.roi_corners = {'tl': (rows_pad, cols_pad),
                                'tr': (rows_pad, self.whole_datamap.shape[1] - cols_pad - 1),
                                'bl': (self.whole_datamap.shape[0] - rows_pad - 1, cols_pad),
                                'br': (self.whole_datamap.shape[0] - rows_pad - 1,
                                       self.whole_datamap.shape[1] - cols_pad - 1)}

        elif type(intervals) is dict:
            # User used a dict of intervals with rows/cols
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

        self.base_datamap = numpy.copy(self.whole_datamap)
        self.sub_datamaps_dict["base"] = self.base_datamap

    def set_bleached_datamap(self, bleached_datamap):
        """
        Updates the datamap.

        This functions updates the ``datamap.whole_datamap`` attribute to the bleached version. I put this in case the user
        does not want to update the datamap after bleaching it directly through the microscope's get_signal_and_bleach
        method, in order to do multiple experiments on the same setting.

        :param bleached_datamap: An array of the datamap after the lasers have passed over it (after it has bleached).
                                 Has to be of the same shape as self.whole_datamap
        """
        if bleached_datamap.shape != self.whole_datamap.shape:
            raise ValueError("Bleached datamap to set as new datamap has to be of the same shape as the datamap pre "
                             "bleaching.")
        self.whole_datamap = bleached_datamap

class TemporalDatamap(Datamap):
    """
    Implements a dynamic datamap

    This class inherits from Datamap, adding the t dimension to it for managing Ca2+ flashes and diffusion.
    The TemporalDatamap object is split into subdatamaps. In the simplest case, the only subdatamap is the base, which
    does not change with time (unless being acquired on with bleaching). In the case of Ca2+ flashes, we can add a
    subdatamap containing the flashes separately from the base datamap. Thus, for a certain time step t, the
    whole_datamap is the sum of the base and the flash at idx t.

    Currently, as there is only the Ca2+ flash dynamics implemented, the TemporalDatamap is initialized by passing
    the whole molecule disposition and the pixelsize, as for Datamap, with the addition of a list containing all of
    the synapse objects in the Datamap.

    :param whole_datamap: The disposition of the molecules in the sample. This represents the whole sample, from which
                          only a region will be imaged (roi). (numpy array)
    :param datamap_pixelsize: The size of a pixel of the datamap. (m)
    :param synapses: The list of synapses present in the whole_datamap
    """

    def __init__(self, whole_datamap, datamap_pixelsize, synapses):
        super().__init__(whole_datamap, datamap_pixelsize)
        # add flat synapses list as attribute
        self.synapses = synapses
        self.contains_sub_datamaps = {"base": True,
                                      "flashes": False}
        # self.sub_datamaps_dict = {}
        self.sub_datamaps_idx_dict = {}

    def __setitem__(self, key, value):
        if key == "flashes":
            self.sub_datamaps_idx_dict[key] = value
            self.sub_datamaps_dict[key] = self.flash_tstack[value]
        elif key == "base":
            pass

    def create_t_stack_dmap(self, acq_time, min_timestep, fwhm_step_sec_correspondance, curves_path,
                            probability):
        """
        Generates the flashes for the TemporalDatamap.
         
        Updates the dictionnaries to confirm that flash subdatamaps exist
        and to initialize the flash time step at 0. Generates a flash_tstack, a 3D array containing the evolution
        of the flashes for every time step. For time step t, the whole_datamap is thus base + flash_tstack[t]
        
        :param acq_time: The time for which the acquisition will last, determines how many flash steps will occur. (s)
        :param min_timestep: The smallest discrete time steps on which the experiment will be run. For instance, if we
                             want an experiment to last 10 seconds, we need to define a loop that will iterate through
                             minimal time increments, which could be 1s, 0.1s, ...
        :param fwhm_step_sec_correspondance: Tuple containing the correspondance between the width of the FWHM of a
                                             a flash in arbitrary time step units, and how long we want that FWHM to
                                             last. Usually (10, 1.5) is used. MODIFY THIS SO THIS IS A DEFAULT VALUE.
        :param curves_path: Path to the .npy file of the light curves being sampled in order to generate random
                            flashes.
        :param probability: The probability of a flash starting on a synapse.
        """
        synapse_flashing_dict, synapse_flash_idx_dict, synapse_flash_curve_dict, isolated_synapses_frames = \
            utils.generate_synapse_flash_dicts(self.synapses, self.whole_datamap[self.roi].shape)
        n_flash_updates, _ = utils.compute_time_correspondances((fwhm_step_sec_correspondance[0],
                                                                fwhm_step_sec_correspondance[1]), acq_time,
                                                                min_timestep, mode="flash")
        # la base_datamap n'est pas créée tant que set_roi n'a pas été call
        self.sub_datamaps_dict["base"] = self.base_datamap
        self.flash_tstack = numpy.zeros((n_flash_updates + 1, *self.whole_datamap.shape), dtype=numpy.int32)
        for i in range(n_flash_updates):
            synapse_flashing_dict, synapse_flash_idx_dict, \
            synapse_flash_curve_dict, temp_dmap = utils.flash_routine(self.synapses, probability, synapse_flashing_dict,
                                                                      synapse_flash_idx_dict, curves_path,
                                                                      synapse_flash_curve_dict,
                                                                      isolated_synapses_frames,
                                                                      copy.deepcopy(self))   # si je copie pas ça chie

            self.flash_tstack[i] = temp_dmap - self.base_datamap
        self.flash_tstack[-1] = temp_dmap - self.base_datamap   # le petit dernier pour la route
        self.contains_sub_datamaps["flashes"] = True
        self.sub_datamaps_idx_dict["flashes"] = 0
        self.sub_datamaps_dict["flashes"] = self.flash_tstack[0]
        self.update_whole_datamap(0)

    def bleach_future(self, indices, bleached_sub_datamaps_dict):
        """
        Applies bleaching to the future flash subdatamaps according to the bleaching that occured to the current flash
        subdatamap

        :param indices: A dictionary containing the indices of the time steps we are currently at for the subdatamaps.
                        For now, as there is only the flash subdatamap implemented, the dictionary will simply be
                        indices = {"flashes": idx}, with idx being an >=0 integer.
        :param bleached_sub_datamaps_dict: A dictionary containing the bleached subdatamaps (base, flashes)
        """
        what_bleached = self.flash_tstack[indices["flashes"]] - bleached_sub_datamaps_dict["flashes"]
        self.flash_tstack[indices["flashes"]] = bleached_sub_datamaps_dict["flashes"]
        # UPDATE THE FUTURE
        with numpy.errstate(divide='ignore', invalid='ignore'):
            flash_survival = bleached_sub_datamaps_dict["flashes"] / self.flash_tstack[indices["flashes"]]
        flash_survival[numpy.isnan(flash_survival)] = 1
        self.flash_tstack[indices["flashes"] + 1:] -= what_bleached
        self.flash_tstack[indices["flashes"] + 1:] = numpy.multiply(
            self.flash_tstack[indices["flashes"] + 1:],
            flash_survival)
        self.flash_tstack[indices["flashes"] + 1:] = numpy.rint(
            self.flash_tstack[indices["flashes"] + 1:])
        self.flash_tstack[indices["flashes"] + 1:] = numpy.where(
            self.flash_tstack[indices["flashes"] + 1:] < 0,
            0, self.flash_tstack[indices["flashes"] + 1:])
        self.whole_datamap += self.flash_tstack[indices["flashes"]]

    def update_whole_datamap(self, flash_idx):
        """
        Method used to update the whole datamap using the indices of the sub datamaps.

        Whole datamap is the base datamap + all the sub datamaps (for flashes, diffusion, etc).

        :param flash_idx: The index of the flash for the most recent acquisition.
        """
        self.whole_datamap = self.base_datamap + self.flash_tstack[flash_idx]

    def update_dicts(self, indices):
        """
        Method used to update the dicts of the temporal datamap.
        
        :param indices: A dict containing the indices of the time step for the different temporal sub datamaps (so far
                        only flashes).
        """
        self.sub_datamaps_idx_dict = indices
        self.sub_datamaps_dict["flashes"] = self.flash_tstack[indices["flashes"]]


class TemporalSynapseDmap(Datamap):
    """
    Temporal Datamap of a Synaptic region with nanodomains.
    """
    def __init__(self, whole_datamap, datamap_pixelsize, synapse_obj):
        super().__init__(whole_datamap, datamap_pixelsize)
        self.synapse = synapse_obj
        self.contains_sub_datamaps = {"base": True,
                                      "flashes": False}
        self.sub_datamaps_idx_dict = {}
        self.init_molecs_ratio = numpy.ones(self.whole_datamap.shape)

    def __setitem__(self, key, value):
        if key == "flashes":
            self.sub_datamaps_idx_dict[key] = value
            self.sub_datamaps_dict[key] = self.flash_tstack[value]
        elif key == "base":
            pass

    def create_t_stack_dmap(self, decay_time_us, delay=2, n_decay_steps=10, n_molecules_multiplier=28, end_pad=0):
        """
        Creates the t stack for the evolution of the flash of the nanodmains in the synapse.

        Very similar implementation to TemporalDatamap's create_t_stack_dmap method
        Assumes the roi is set.

        :param decay_time_us: The time it takes for the flash to decay to 1/e of its initial value. (us)
        :param delay: The delay before the flash starts. (us)
        :param n_decay_steps: The number of time steps the flash will take to decay. (int)
        :param n_molecules_multiplier: The multiplier for the number of molecules in the flash. (int)
        :param end_pad: The number of time steps to pad the flash with at the end. (int)
        """
        self.decay_time_us = decay_time_us
        self.time_usec_between_flash_updates = int(numpy.round(self.decay_time_us / n_decay_steps))
        self.sub_datamaps_dict["base"] = self.base_datamap

        flash_curve = utils.hand_crafted_light_curve(delay=delay, n_decay_steps=n_decay_steps,
                                                     n_molecules_multiplier=n_molecules_multiplier, end_pad=end_pad)

        self.flash_tstack = numpy.zeros((flash_curve.shape[0], *self.whole_datamap.shape))
        self.nanodomains_active = []
        for t, nanodomains_multiplier in enumerate(flash_curve):
            # -1 makes it so the whole_datamap at flash values of 1 are equal to the base datamap
            nd_mult = int(numpy.round(nanodomains_multiplier)) - 1
            if nd_mult < 0:
                nd_mult = 0
            for nanodomain in self.synapse.nanodomains:
                self.flash_tstack[t][self.roi][nanodomain.coords[0], nanodomain.coords[1]] = \
                    self.synapse.n_molecs_base * nd_mult    # - self.synapse.n_molecs_base
            if self.flash_tstack[t].max() > 0:
                self.nanodomains_active.append(True)
            else:
                self.nanodomains_active.append(False)

        self.nanodomains_active_currently = self.nanodomains_active[0]
        self.contains_sub_datamaps["flashes"] = True
        self.sub_datamaps_idx_dict["flashes"] = 0
        self.sub_datamaps_dict["flashes"] = self.flash_tstack[0]
        self.update_whole_datamap(0)

    def create_t_stack_dmap_smooth(self, decay_time_us, delay=2, n_decay_steps=10, n_molecules_multiplier=None,
                                   end_pad=0, individual_flashes=False):
        """
        Creates the t stack for the evolution of the flash of the nanodmains in the synapse.

        Very similar implementation to TemporalDatamap's create_t_stack_dmap method.
        Assumes the roi is set.

        :param decay_time_us: The time it takes for the flash to decay to 1/e of its initial value. (us)
        :param delay: The delay before the flash starts. (us)
        :param n_decay_steps: The number of time steps the flash will take to decay. (int)
        :param n_molecules_multiplier: The multiplier for the number of molecules in the flash. (int)
        :param end_pad: The number of time steps to pad the flash with at the end. (int)        
        """
        self.decay_time_us = decay_time_us
        self.time_usec_between_flash_updates = int(numpy.round(self.decay_time_us / n_decay_steps))
        self.sub_datamaps_dict["base"] = self.base_datamap

        if type(delay) is tuple:
            delay = numpy.random.randint(delay[0], delay[1])

        flash_curves = []
        for i in range(len(self.synapse.nanodomains)):
            flash_curve = utils.smooth_ramp_hand_crafted_light_curve(
                delay=delay,
                n_decay_steps=n_decay_steps,
                n_molecules_multiplier=n_molecules_multiplier,
                end_pad=end_pad
            )
            flash_curves.append(numpy.copy(flash_curve))

        self.flash_tstack = numpy.zeros((flash_curve.shape[0], *self.whole_datamap.shape))
        self.nanodomains_active = []
        for t, nanodomains_multiplier in enumerate(flash_curve):
            # -1 makes it so the whole_datamap at flash values of 1 are equal to the base datamap, which I think I want
            # nd_mult = int(numpy.round(nanodomains_multiplier)) - 1
            # if nd_mult < 0:
            #     nd_mult = 0
            for nd_idx, nanodomain in enumerate(self.synapse.nanodomains):
                if not individual_flashes:
                    nd_mult = int(numpy.round(nanodomains_multiplier)) - 1
                    if nd_mult < 0:
                        nd_mult = 0
                else:
                    nd_mult = int(numpy.round(flash_curves[nd_idx][t])) - 1
                    if nd_mult < 0:
                        nd_mult = 0
                self.flash_tstack[t][self.roi][nanodomain.coords[0], nanodomain.coords[1]] = \
                    self.synapse.n_molecs_base * nd_mult    # - self.synapse.n_molecs_base
            if self.flash_tstack[t].max() > 0:
                self.nanodomains_active.append(True)
            else:
                self.nanodomains_active.append(False)

        self.nanodomains_active_currently = self.nanodomains_active[0]
        self.contains_sub_datamaps["flashes"] = True
        self.sub_datamaps_idx_dict["flashes"] = 0
        self.sub_datamaps_dict["flashes"] = self.flash_tstack[0]
        self.update_whole_datamap(0)
        
    def create_t_stack_dmap_smooth_2(self, time_usec_step_correspondance, n_steps_rise=100,
                                     n_steps_decay=25, delay=0, end_pad=0, n_molecules_multiplier=20,
                                     individual_flashes=False, **kwargs):
        n_steps_light_curve = delay + n_steps_rise + n_steps_decay
        self.time_usec_between_flash_updates = int(time_usec_step_correspondance)
        self.decay_time_us = int(n_steps_decay / self.time_usec_between_flash_updates)
        self.sub_datamaps_dict["base"] = self.base_datamap
        exp_time_us = kwargs.get("exp_time_us", 2000000)

        if type(delay) is tuple:
            delay = numpy.random.randint(delay[0], delay[1])

        flash_curves = []
        for i in range(len(self.synapse.nanodomains)):
            flash_curve = utils.smooth_ramp_hand_crafted_light_curve_2(
                delay=delay,
                n_steps_decay=n_steps_decay,
                n_molecules_multiplier=n_molecules_multiplier,
                n_steps_rise=n_steps_rise,
                end_pad=end_pad
            )
            n_steps_total = flash_curve.shape[0]
            if exp_time_us > n_steps_total * time_usec_step_correspondance:
                n_usec_missing = int(exp_time_us - (n_steps_total * time_usec_step_correspondance))
                n_steps_missing = int(n_usec_missing / time_usec_step_correspondance)
                missing_steps = numpy.ones(n_steps_missing + 1)
                flash_curve = numpy.append(flash_curve, missing_steps)
            flash_curves.append(numpy.copy(flash_curve))

        self.flash_tstack = numpy.zeros((flash_curve.shape[0], *self.whole_datamap.shape))
        self.nanodomains_active = []
        for t, nanodomains_multiplier in enumerate(flash_curve):
            # -1 makes it so the whole_datamap at flash values of 1 are equal to the base datamap, which I think I want
            # nd_mult = int(numpy.round(nanodomains_multiplier)) - 1
            # if nd_mult < 0:
            #     nd_mult = 0
            for nd_idx, nanodomain in enumerate(self.synapse.nanodomains):
                if not individual_flashes:
                    # nd_mult = int(numpy.round(nanodomains_multiplier)) - 1
                    nd_mult = nanodomains_multiplier - 1
                    if nd_mult < 0:
                        nd_mult = 0
                else:
                    # nd_mult = int(numpy.round(nanodomains_multiplier)) - 1
                    nd_mult = nanodomains_multiplier - 1
                    if nd_mult < 0:
                        nd_mult = 0
                self.flash_tstack[t][self.roi][nanodomain.coords[0], nanodomain.coords[1]] = \
                    int(self.synapse.n_molecs_base * nd_mult)    # - self.synapse.n_molecs_base
            if self.flash_tstack[t].max() > 0:
                self.nanodomains_active.append(True)
            else:
                self.nanodomains_active.append(False)

        self.nanodomains_active_currently = self.nanodomains_active[0]
        self.contains_sub_datamaps["flashes"] = True
        self.sub_datamaps_idx_dict["flashes"] = 0
        self.sub_datamaps_dict["flashes"] = self.flash_tstack[0]
        self.update_whole_datamap(0)

    def create_t_stack_dmap_sampled(self, decay_time_us, delay=0, n_decay_steps=10, curves_path=None,
                                    individual_flashes=False):
        """
        Creates the t stack for the evolution of the flash of the nanodmains in the synapse.

        Very similar implementation to TemporalDatamap's create_t_stack_dmap method.
        Assumes the roi is set.

        :param decay_time_us: The time it takes for the flash to decay to 1/e of its initial value. (us)
        :param delay: The delay before the flash starts. (us)
        :param n_decay_steps: The number of time steps the flash will take to decay. (int)
        :param curves_path: Path to the .npy file of the light curves being sampled in order to generate random
        :param individual_flashes: If True, each nanodomain will have its own flash curve
        """
        # even though in this case the decay_time_us and n_decay_steps wont help generate the light curve, they will
        # help define de frequency of updates
        self.decay_time_us = decay_time_us
        self.time_usec_between_flash_updates = int(numpy.round(self.decay_time_us / n_decay_steps))
        self.sub_datamaps_dict["base"] = self.base_datamap

        if type(delay) is tuple:
            delay = numpy.random.randint(delay[0], delay[1])

        if curves_path is None:
            curves_path = "flash_files/events_curves.npy"
        flash_curve = utils.sampled_flash_manipulations(curves_path, delay, rescale=True, seed=None)

        if individual_flashes:
            flash_peak = numpy.argmax(flash_curve)
            flash_variances = []
            for i in range(len(self.synapse.nanodomains)):
                flash_variances.append(numpy.random.randint(-4, 4))
        else:
            flash_peak = 0

        self.flash_tstack = numpy.zeros((flash_curve.shape[0], *self.whole_datamap.shape))
        self.nanodomains_active = []
        for t, nanodomains_multiplier in enumerate(flash_curve):
            # -1 makes it so the whole_datamap at flash values of 1 are equal to the base datamap, which I think I want
            nd_mult = int(numpy.round(nanodomains_multiplier)) - 1
            if nd_mult < 0:
                nd_mult = 0
            for nd_idx, nanodomain in enumerate(self.synapse.nanodomains):
                if not individual_flashes or t < flash_peak:
                    flash_variance = 0
                else:
                    flash_variance = flash_variances[nd_idx]
                    if nd_mult + flash_variance < 0:
                        flash_variance = 0
                self.flash_tstack[t][self.roi][nanodomain.coords[0], nanodomain.coords[1]] = \
                    self.synapse.n_molecs_base * (nd_mult + flash_variance) # - self.synapse.n_molecs_base
            if self.flash_tstack[t].max() > 0:
                self.nanodomains_active.append(True)
            else:
                self.nanodomains_active.append(False)

        self.nanodomains_active_currently = self.nanodomains_active[0]
        self.contains_sub_datamaps["flashes"] = True
        self.sub_datamaps_idx_dict["flashes"] = 0
        self.sub_datamaps_dict["flashes"] = self.flash_tstack[0]
        self.update_whole_datamap(0)

    def bleach_future(self, indices, bleached_sub_datamaps_dict):
        """
        Applies bleaching to the future flash subdatamaps according to the bleaching that occured to the current flash
        subdatamap

        :param indices: A dictionary containing the indices of the time steps we are currently at for the subdatamaps.
                        For now, as there is only the flash subdatamap implemented, the dictionary will simply be
                        indices = {"flashes": idx}, with idx being an >=0 integer.
        :param bleached_sub_datamaps_dict: A dictionary containing the bleached subdatamaps (base, flashes)
        """
        what_bleached = self.flash_tstack[indices["flashes"]] - bleached_sub_datamaps_dict["flashes"]
        self.flash_tstack[indices["flashes"]] = bleached_sub_datamaps_dict["flashes"]
        # UPDATE THE FUTURE
        with numpy.errstate(divide='ignore', invalid='ignore'):
            flash_survival = bleached_sub_datamaps_dict["flashes"] / self.flash_tstack[indices["flashes"]]
        flash_survival[numpy.isnan(flash_survival)] = 1
        self.flash_tstack[indices["flashes"] + 1:] -= what_bleached
        self.flash_tstack[indices["flashes"] + 1:] = numpy.multiply(
            self.flash_tstack[indices["flashes"] + 1:],
            flash_survival)
        self.flash_tstack[indices["flashes"] + 1:] = numpy.rint(
            self.flash_tstack[indices["flashes"] + 1:])
        self.flash_tstack[indices["flashes"] + 1:] = numpy.where(
            self.flash_tstack[indices["flashes"] + 1:] < 0,
            0, self.flash_tstack[indices["flashes"] + 1:])
        self.flash_tstack = self.flash_tstack.astype('int64')
        self.whole_datamap += self.flash_tstack[indices["flashes"]]

    def bleach_future_proportional(self, indices, bleached_sub_datamaps_dict, unbleached_whole_datamap):
        """
        Photobleaches the future flash subdatamaps according to the bleaching that occured to the current flash

        For instance, if 3/5 molecules are left after bleaching, then the number of molecules in the subsequent
        flashes will be multiplied by 3/5.
        """
        bleached_whole_datamap = numpy.zeros(unbleached_whole_datamap.shape)
        for key in bleached_sub_datamaps_dict:
            bleached_whole_datamap += bleached_sub_datamaps_dict[key]
        ratio = bleached_whole_datamap / numpy.where(numpy.logical_and(unbleached_whole_datamap == 0,
                                                                       bleached_whole_datamap == 0),
                                                     1, unbleached_whole_datamap)
        # self.flash_tstack[indices["flashes"]:, :, :] *= ratio
        # self.flash_tstack = numpy.ceil(self.flash_tstack)
        # self.flash_tstack = self.flash_tstack.astype('int64')
        self.flash_tstack[indices["flashes"]:, :, :] = numpy.ceil(self.flash_tstack[indices["flashes"]:, :, :]
                                                                  * ratio).astype('int64')
        self.whole_datamap = bleached_sub_datamaps_dict["base"] + self.flash_tstack[indices["flashes"]]


    def update_whole_datamap(self, flash_idx):
        """
        Update de whole datamap using the indices of the sub datamaps.

        Whole datamap is the base datamap + all the sub datamaps (for flashes, diffusion, etc).
        :param flash_idx: The index of the flash for the most recent acquisition.
        """
        # If the experiment runs longer than the generated flash curve, just keep extending the final value of the curve
        if flash_idx >= self.flash_tstack.shape[0]:
            flash_idx = self.flash_tstack.shape[0] - 1
        self.whole_datamap = self.base_datamap + self.flash_tstack[flash_idx]
        self.nanodomains_active_currently = self.nanodomains_active[flash_idx]   # updates whether or not flashing rn


    def update_dicts(self, indices):
        """
        Method used to update the dicts of the temporal datamap

        :param indices: A dict containing the indices of the time step for the different temporal sub datamaps (so far
                        only flashes).
        """
        self.sub_datamaps_idx_dict = indices
        self.sub_datamaps_dict["flashes"] = self.flash_tstack[indices["flashes"]]


class TestTemporalDmap(Datamap):
    """
    Test class for the TemporalDatamap class
    """
    def __init__(self, whole_datamap, datamap_pixelsize):
        super().__init__(whole_datamap, datamap_pixelsize)
        self.contains_sub_datamaps = {"base": True,
                                      "flashes": False}
        self.sub_datamaps_idx_dict = {}

    def __setitem__(self, key, value):
        if key == "flashes":
            self.sub_datamaps_idx_dict[key] = value
            self.sub_datamaps_dict[key] = self.flash_tstack[value]
        elif key == "base":
            pass

    def create_t_stack_dmap(self, decay_time_us, delay=2, n_decay_steps=10, n_molecules_multiplier=28, end_pad=0):
        """
        Creates the t stack for the evolution of the flash of the nanodmains in the synapse.
        Very similar implementation to TemporalDatamap's create_t_stack_dmap method
        Assumes the roi is set
        """
        self.decay_time_us = decay_time_us
        self.time_usec_between_flash_updates = int(numpy.round(self.decay_time_us / n_decay_steps))
        self.sub_datamaps_dict["base"] = self.base_datamap

        flash_curve = utils.hand_crafted_light_curve(delay=delay, n_decay_steps=n_decay_steps,
                                                     n_molecules_multiplier=n_molecules_multiplier, end_pad=end_pad)

        self.flash_tstack = numpy.zeros((flash_curve.shape[0], *self.whole_datamap.shape))
        for t, nanodomains_multiplier in enumerate(flash_curve):
            # -1 makes it so the whole_datamap at flash values of 1 are equal to the base datamap, which I think I want
            nd_mult = int(numpy.round(nanodomains_multiplier)) - 1
            if nd_mult < 0:
                nd_mult = 0
            self.flash_tstack[t][self.roi] = numpy.max(self.whole_datamap) * nd_mult

        self.contains_sub_datamaps["flashes"] = True
        self.sub_datamaps_idx_dict["flashes"] = 0
        self.sub_datamaps_dict["flashes"] = self.flash_tstack[0]
        self.update_whole_datamap(0)

    def bleach_future(self, indices, bleached_sub_datamaps_dict):
        """
        pass for now
        """
        what_bleached = self.flash_tstack[indices["flashes"]] - bleached_sub_datamaps_dict["flashes"]
        self.flash_tstack[indices["flashes"]] = bleached_sub_datamaps_dict["flashes"]
        # UPDATE THE FUTURE
        with numpy.errstate(divide='ignore', invalid='ignore'):
            flash_survival = bleached_sub_datamaps_dict["flashes"] / self.flash_tstack[indices["flashes"]]
        flash_survival[numpy.isnan(flash_survival)] = 1
        self.flash_tstack[indices["flashes"] + 1:] -= what_bleached
        self.flash_tstack[indices["flashes"] + 1:] = numpy.multiply(
            self.flash_tstack[indices["flashes"] + 1:],
            flash_survival)
        self.flash_tstack[indices["flashes"] + 1:] = numpy.rint(
            self.flash_tstack[indices["flashes"] + 1:])
        self.flash_tstack[indices["flashes"] + 1:] = numpy.where(
            self.flash_tstack[indices["flashes"] + 1:] < 0,
            0, self.flash_tstack[indices["flashes"] + 1:])
        self.flash_tstack = self.flash_tstack.astype('int64')
        self.whole_datamap += self.flash_tstack[indices["flashes"]]

    def update_whole_datamap(self, flash_idx):
        if flash_idx >= self.flash_tstack.shape[0]:
            flash_idx = self.flash_tstack.shape[0] - 1
        self.whole_datamap = self.base_datamap + self.flash_tstack[flash_idx]

    def update_dicts(self, indices):
        self.sub_datamaps_idx_dict = indices
        self.sub_datamaps_dict["flashes"] = self.flash_tstack[indices["flashes"]]


class Clock():
    """
    Clock class to keep track of time in experiments involving time

    :param time_quantum_us: The minimal time increment on which the experiment loop will happen. All other time
                            increments in the experiment should be a multiple of this value (in micro seconds (us))
                            (int)

    .. note:: 
        The ``time_quantum_us`` is an ``int`` and so is the ``current_time`` attribute. This means the longest time an experiment
        can last is determined by the size of the biggest ``int64``, which means it is 9223372036854775807 us, or
        9223372036854.775807 s, which I think should be ample time :)
    """
    def __init__(self, time_quantum_us):
        if type(time_quantum_us) is not int:
            raise TypeError(f"The time_quantum_us value should be an int, but a {type(time_quantum_us)} was passed !")
        self.time_quantum_us = time_quantum_us
        self.current_time = 0

    def update_time(self):
        """
        Updates the current_time by 1 time_quantum_us
        """
        self.current_time += self.time_quantum_us

    def reset(self):
        """
        Resets the current_time to 0
        """
        self.current_time = 0


class RandomActionSelector():
    """
    Class which selects a random action from :
    0 - Confocal acquisition
    1 - STED acquisition
    2 - Wait (for the time of 1 acquisition)

    ..note::
        For now we have are pre setting the pdt, p_ex and p_sted that will be used for the actions.
        A real agent would select the powers / dwellit me individually

    :param pdt: The pixel dwell time that will be used in the acquisitions
    :param p_ex: The excitation beam power that will be used when the selected action is confocal or sted
    :param p_sted: The STED beam power that will be used when the selected action is sted
    """

    def __init__(self, pdt, p_ex, p_sted, roi_shape):
        self.pdt = numpy.ones(roi_shape) * pdt
        self.p_ex = p_ex
        self.p_sted = p_sted
        self.action_selected = None
        self.action_completed = False
        self.valid_actions = {0: "confocal", 1: "sted", 2: "wait"}
        self.n_actions = 3
        self.current_action_pdt = None
        self.current_action_p_ex = None
        self.current_action_p_sted = None

    def select_action(self):
        """
        selects a random action from the current actions
        """
        # self.action_selected = self.valid_actions[numpy.random.randint(0, self.n_actions)]
        # temporary dont want him to select wait for debuggind prupouses
        self.action_selected = self.valid_actions[numpy.random.randint(0, 1)]
        if self.action_selected == "confocal":
            self.current_action_p_ex = self.p_ex
            self.current_action_p_sted = 0.0
            self.current_action_pdt = self.pdt
        elif self.action_selected == "sted":
            self.current_action_p_ex = self.p_ex
            self.current_action_p_sted = self.p_sted
            self.current_action_pdt = self.pdt
        elif self.action_selected == "wait":
            self.current_action_p_ex = 0.0
            self.current_action_p_sted = 0.0
            self.current_action_pdt = self.pdt
        else:
            raise ValueError("Impossible action selected :)")
        self.action_completed = False


class TemporalExperiment():
    """
    This temporal experiment will run on a loop based on the action selections instead of on the time to make it easier
    to integrate the agent/gym stuff :)
    """
    def __init__(self, clock, microscope, temporal_datamap, exp_runtime, bleach=True, bleach_mode="default"):
        self.clock = clock
        self.microscope = microscope
        self.temporal_datamap = temporal_datamap
        self.exp_runtime = exp_runtime
        self.flash_tstep = 0
        self.bleach = bleach
        self.bleach_mode = bleach_mode

    def play_action(self, pdt, p_ex, p_sted):
        """
        l'idée va comme ça
        fait une loop sur X épisodes
            - quand un épisode commence on crée un objet TemporalExperimentV2p1 avec un certain exp_runtime
            - l'agent choisit une action et la joue
                - dans la méthode de jouer l'action (ici) on fait toute la gestion des updates de flash mid acq si c'est
                  le cas, finir l'action early si on run out de temps, ...
        *** pdt can be a float value, I will convert it into an array filled with that value if this is the case ***
        """
        indices = {"flashes": self.flash_tstep}
        intensity = numpy.zeros(self.temporal_datamap.whole_datamap[self.temporal_datamap.roi].shape).astype(float)
        prob_ex = numpy.ones(self.temporal_datamap.whole_datamap.shape).astype(float)
        prob_sted = numpy.ones(self.temporal_datamap.whole_datamap.shape).astype(float)
        action_required_time = numpy.sum(pdt) * 1e6   # this assumes a pdt given in sec * 1e-6
        action_completed_time = self.clock.current_time + action_required_time
        # +1 ensures no weird business if tha last acq completed as the dmap updated
        time_steps_covered_by_acq = numpy.arange(int(self.clock.current_time) + 1, action_completed_time)
        dmap_times = []
        for i in time_steps_covered_by_acq:
            if i % self.temporal_datamap.time_usec_between_flash_updates == 0 and i != 0:
                dmap_times.append(i)
        dmap_update_times = numpy.arange(self.temporal_datamap.time_usec_between_flash_updates, self.exp_runtime + 1,
                                         self.temporal_datamap.time_usec_between_flash_updates)

        # if len(dmap_times) == 0, this means the acquisition is not interupted and we can just do it whole
        # if not, then we need to split the acquisition
        if len(dmap_times) == 0:
            acq, bleached, temporal_acq_elts = self.microscope.get_signal_and_bleach(self.temporal_datamap,
                                                                                     self.temporal_datamap.pixelsize,
                                                                                     pdt, p_ex, p_sted,
                                                                                     indices=indices,
                                                                                     acquired_intensity=intensity,
                                                                                     bleach=self.bleach, update=True,
                                                                                     bleach_mode=self.bleach_mode)

            intensity = temporal_acq_elts["intensity"]
            self.clock.current_time += action_required_time
            return acq, bleached
        else:
            # assume raster pixel scan
            pixel_list = utils.pixel_sampling(intensity, mode="all")
            flash_t_step_pixel_idx_dict = {}
            n_keys = 0
            first_key = self.flash_tstep
            for i in range(len(dmap_times) + 1):
                pdt_cumsum = numpy.cumsum(pdt * 1e6)
                if i < len(dmap_times) and dmap_times[i] >= self.exp_runtime:
                    # the datamap would update, but the experiment will be over before then
                    update_pixel_idx = numpy.argwhere(pdt_cumsum + self.clock.current_time > self.exp_runtime)[0, 0]
                    flash_t_step_pixel_idx_dict[self.flash_tstep] = update_pixel_idx
                    if self.flash_tstep > first_key:
                        flash_t_step_pixel_idx_dict[self.flash_tstep] += flash_t_step_pixel_idx_dict[self.flash_tstep - 1]
                    self.clock.current_time = self.exp_runtime
                    break
                elif i < len(dmap_times):  # mid update split
                    # update_pixel_idx = numpy.argwhere(pdt_cumsum + self.clock.current_time > dmap_update_times[self.flash_tstep])[0, 0]
                    # not sure if the + 1 is legit but it seems to fix my bug of acqs being 1 pixel short
                    update_pixel_idx = \
                    numpy.argwhere(pdt_cumsum + self.clock.current_time > dmap_update_times[self.flash_tstep])[0, 0] + 1
                    flash_t_step_pixel_idx_dict[self.flash_tstep] = update_pixel_idx
                    if self.flash_tstep > first_key:
                        flash_t_step_pixel_idx_dict[self.flash_tstep] += flash_t_step_pixel_idx_dict[self.flash_tstep - 1]
                    n_keys += 1
                    self.flash_tstep += 1
                    self.clock.current_time += pdt_cumsum[update_pixel_idx - 1]
                else:  # from last update to the end of acq
                    update_pixel_idx = pdt_cumsum.shape[0] - 1
                    flash_t_step_pixel_idx_dict[self.flash_tstep] = update_pixel_idx
                    self.clock.current_time += pdt_cumsum[update_pixel_idx] - pdt_cumsum[flash_t_step_pixel_idx_dict[self.flash_tstep - 1] - 1]
            key_counter = 0
            for key in flash_t_step_pixel_idx_dict:
                if key_counter == 0:
                    acq_pixel_list = pixel_list[0:flash_t_step_pixel_idx_dict[key]]
                    # print(f"len(acq_pixel_list) = {len(acq_pixel_list)}")
                elif key_counter == n_keys:
                    acq_pixel_list = pixel_list[
                                     flash_t_step_pixel_idx_dict[key - 1]:flash_t_step_pixel_idx_dict[key] + 1]
                    # print(f"len(acq_pixel_list) = {len(acq_pixel_list)}")
                else:
                    acq_pixel_list = pixel_list[flash_t_step_pixel_idx_dict[key - 1]:flash_t_step_pixel_idx_dict[key]]
                if len(acq_pixel_list) == 0:  # acq is over time to go home
                    # should I still update? PogChampionship I should
                    key_counter += 1
                    indices = {"flashes": key}
                    self.temporal_datamap.update_whole_datamap(key)
                    self.temporal_datamap.update_dicts(indices)
                    break
                    # pass
                key_counter += 1
                indices = {"flashes": key}
                self.temporal_datamap.update_whole_datamap(key)
                self.temporal_datamap.update_dicts(indices)
                acq, bleached, temporal_acq_elts = self.microscope.get_signal_and_bleach(self.temporal_datamap,
                                                                                         self.temporal_datamap.pixelsize,
                                                                                         pdt, p_ex, p_sted,
                                                                                         indices=indices,
                                                                                         acquired_intensity=intensity,
                                                                                         bleach=self.bleach,
                                                                                         bleach_mode=self.bleach_mode,
                                                                                         update=True,
                                                                                         pixel_list=acq_pixel_list,
                                                                                         prob_ex=prob_ex,
                                                                                         prob_sted=prob_sted)

                intensity = temporal_acq_elts["intensity"]
                prob_ex = temporal_acq_elts["prob_ex"]
                prob_sted = temporal_acq_elts["prob_sted"]

            return acq, bleached
