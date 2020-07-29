
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
from pysted import utils
import cUtils

# import mis par BT pour des tests
from matplotlib import pyplot


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
    def get_intensity(self, power, f, n, na, transmission, pixelsize, data_pixelsize=None):
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
        :param pixelsize: The size of an element in the intensity matrix (m).
        :returns: A 2D array.
        '''
        if data_pixelsize is None:
            data_pixelsize = pixelsize
        else:
            pixelsize = data_pixelsize

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
        
        diameter = 2.233 * self.lambda_ / (na * pixelsize)
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
                
                kr = k * radius * pixelsize
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
        
        idx_mid = int((intensity.shape[0]-1) / 2)
        r = utils.fwhm(intensity[idx_mid])
        area_fwhm = numpy.pi * (r  * pixelsize)**2 / 2
        # [RPPhoto2015]
        return intensity * 2 * transmission * power / area_fwhm


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
    def get_intensity(self, power, f, n, na, transmission, pixelsize, data_pixelsize=None):
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
        :param pixelsize: The size of an element in the intensity matrix (m).
        :param data_pixelsize: The size of a pixel in the raw data matrix (m).
        '''
        if data_pixelsize is None:
            data_pixelsize = pixelsize
        else:
            pixelsize = data_pixelsize

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
        
        diameter = 2.233 * self.lambda_ / (na * pixelsize)
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
                
                kr = k * radius * pixelsize
                
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
        
        # for peak intensity
        duty_cycle = self.tau * self.rate
        intensity /= duty_cycle
        
        idx_mid = int((intensity.shape[0]-1) / 2)
        r_out, r_in = utils.fwhm_donut(intensity[idx_mid])
        big_area = numpy.pi * (r_out * pixelsize)**2 / 2
        small_area = numpy.pi * (r_in * pixelsize)**2 / 2
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
    
    def get_detection_psf(self, lambda_, psf, na, transmission, pixelsize):
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
        :param pixelsize: The size of a pixel in the simulated image (m).
        :returns: A 2D array.
        '''
        radius = self.n_airy * 0.61 * lambda_ / na
        pinhole = utils.pinhole(radius, pixelsize, psf.shape[0])
        # convolution [Willig2006] eq. 3
        psf_det = scipy.signal.convolve2d(psf, pinhole, "same")
        # normalization to 1
        psf_det = psf_det / numpy.max(psf_det)
        return psf_det * transmission
    
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
    
    def get_psf(self, na, pixelsize):
        '''Compute the Gaussian-shaped fluorescence PSF.
        
        :param na: The numerical aperture of the objective.
        :param pixelsize: The size of an element in the intensity matrix (m).
        :returns: A 2D array.
        '''
        diameter = 2.233 * self.lambda_ / (na * pixelsize)
        n_pixels = int(diameter / 2) * 2 + 1 # odd number of pixels
        center = int(n_pixels / 2)
        
        fwhm = self.lambda_ / (2 * na)
        
        half_pixelsize = pixelsize / 2
        gauss = numpy.zeros((n_pixels, n_pixels))
        for y in range(n_pixels):
            h_rel = (center - y) * pixelsize
            h_lb = h_rel - half_pixelsize
            h_ub = h_rel + half_pixelsize
            for x in range(n_pixels):
                w_rel = (x - center) * pixelsize
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
    
    def __init__(self, excitation, sted, detector, objective, fluo):
        self.excitation = excitation
        self.sted = sted
        self.detector = detector
        self.objective = objective
        self.fluo = fluo
                
        # caching system
        self.__cache = {}
    
    def __str__(self):
        return str(self.__cache.keys())
    
    def is_cached(self, pixelsize, data_pixelsize=None):
        '''Indicate the presence of a cache entry for the given pixel size.
        
        :param pixelsize: The size of a pixel in the simulated image (m).
        :returns: A boolean.
        '''
        if data_pixelsize is None:
            data_pixelsize = pixelsize
        else:
            pixelsize = data_pixelsize

        pixelsize_nm = int(pixelsize * 1e9)
        return pixelsize_nm in self.__cache
    
    def cache(self, pixelsize, data_pixelsize=None):
        '''Compute and cache the excitation and STED intensities, and the
        fluorescence PSF. These intensities are computed with a power of 1 W
        such that they can serve as a basis to compute intensities with any
        power.
        
        :param pixelsize: The size of a pixel in the simulated image (m).
        :param data_pixelsize: The size of a pixel in the raw data (m)
        :returns: A tuple containing:
        
                  * A 2D array of the excitation intensity for a power of 1 W;
                  * A 2D array of the STED intensity for a a power of 1 W;
                  * A 2D array of the detection PSF.
        '''
        if data_pixelsize is None:
            data_pixelsize = pixelsize
        else:
            pixelsize = data_pixelsize

        pixelsize_nm = int(pixelsize * 1e9)
        if pixelsize_nm not in self.__cache:
            f, n, na = self.objective.f, self.objective.n, self.objective.na
            
            transmission = self.objective.get_transmission(self.excitation.lambda_)
            i_ex = self.excitation.get_intensity(1, f, n, na,
                                                 transmission, pixelsize, data_pixelsize)
            
            transmission = self.objective.get_transmission(self.sted.lambda_)
            i_sted = self.sted.get_intensity(1, f, n, na,
                                             transmission, pixelsize, data_pixelsize)
            
            
            transmission = self.objective.get_transmission(self.fluo.lambda_)
            psf = self.fluo.get_psf(na, pixelsize)
            psf_det = self.detector.get_detection_psf(self.fluo.lambda_, psf,
                                                      na, transmission,
                                                      pixelsize)
            self.__cache[pixelsize_nm] = utils.resize(i_ex, i_sted, psf_det)

        return self.__cache[pixelsize_nm]
    
    def clear_cache(self):
        '''Empty the cache.
        
        .. important::
           It is important to empty the cache if any of the components
           :attr:`excitation`, :attr:`sted`, :attr:`detector`,
           :attr:`objective`, or :attr:`fluorescence` are internally modified
           or replaced.
        '''
        self.__cache = {}
    
    def get_effective(self, pixelsize, p_ex, p_sted, data_pixelsize=None):
        '''Compute the detected signal given some molecules disposition.
        
        :param pixelsize: The size of one pixel of the simulated image (m).
        :param p_ex: The power of the depletion beam (W).
        :param p_sted: The power of the STED beam (W).
        :param data_pixelsize: The size of one pixel of the raw data (m).
        :returns: A 2D array of the effective intensity (W) of a single molecule.
        
        The technique follows the method and equations described in
        [Willig2006]_, [Leutenegger2010]_ and [Holler2011]_.
        '''
        if data_pixelsize is None:
            data_pixelsize = pixelsize
        else:
            pixelsize = data_pixelsize

        h, c = scipy.constants.h, scipy.constants.c
        f, n, na = self.objective.f, self.objective.n, self.objective.na
        
        __i_ex, __i_sted, psf_det = self.cache(pixelsize, data_pixelsize)
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
        eta = (((1 + gamma * numpy.exp(-k_s1 * self.sted.tau * (1 + gamma))) / (1 + gamma)) -\
              numpy.exp(-k_s1 * (gamma * self.sted.tau + T))) / (1 - numpy.exp(-k_s1 * T))
        
        # molecular brigthness [Holler2011]
        sigma_abs = self.fluo.get_sigma_abs(self.excitation.lambda_)
        excitation_probability = sigma_abs * i_ex * self.fluo.qy

        # effective intensity of a single molecule (W) [Willig2006] eq. 3
        return excitation_probability * eta * psf_det
    
    def get_signal(self, datamap, pixelsize, pdt, p_ex, p_sted, datamap_pixelsize=None, pixel_list=None):
        '''Compute the detected signal given some molecules disposition.
        
        :param datamap: A 2D array map of integers indicating how many molecules
                        are contained in each pixel of the simulated image.
        :param pixelsize: The size of one pixel of the simulated image (m).
        :param pdt: The time spent on each pixel of the simulated image (s).
        :param p_ex: The power of the excitation beam (W).
        :param p_sted: The power of the STED beam (W).
        :returns: A 2D array of the number of detected photons on each pixel.
        '''

        """# old version, does not take in count pixelsize ratio, keeping here in case
        # effective intensity across pixels (W)
        effective = self.get_effective(pixelsize, p_ex, p_sted)
        
        # stack one effective per molecule
        intensity = utils.stack(datamap, effective)
        
        photons = self.fluo.get_photons(intensity)

        return self.detector.get_signal(photons, pdt)"""

        # effective intensity across pixels (W)
        # acquisition gaussian is computed using data_pixelsize
        if datamap_pixelsize is None:
            effective = self.get_effective(pixelsize, p_ex, p_sted)
        else:
            effective = self.get_effective(datamap_pixelsize, p_ex, p_sted)

        # stack one effective per molecule
        if pixel_list is None:
            intensity = utils.stack_btmod_pixsize(datamap, effective, datamap_pixelsize, pixelsize)
        else:
            # caller la fonction stack qui prend en compte la liste et le pixelsizes :)
            intensity = utils.stack_btmod_definitive(datamap, effective, datamap_pixelsize, pixelsize, pixel_list)

        photons = self.fluo.get_photons(intensity)

        default_returned_array = self.detector.get_signal(photons, pdt)

        return default_returned_array

    def bleach(self, datamap, pixelsize, pixeldwelltime, p_ex, p_sted, datamap_pixelsize, pixel_list=None):
        '''Compute the bleached data map using the following survival
        probability per molecule:

        .. math:: exp(-k \cdot t)

        where

        .. math::

           k = \\frac{k_{ISC} \sigma_{abs} I^2}{\sigma_{abs} I (\\tau_{triplet}^{-1} + k_{ISC}) + (\\tau_{triplet} \\tau_{fluo})} \sigma_{triplet} {phy}_{react}

        where :math:`c` is a constant, :math:`I` is the intensity, and :math:`t`
        if the pixel dwell time [Jerker1999]_ [Garcia2000]_ [Staudt2009]_.

        This version generates the lasers using datamap_pixelsize, and determines how many pixels to skip by the ratio
        between datamap_pixelsize and pixelsize.

        :param datamap: A 2D array map of integers indicating how many molecules
                        are contained in each pixel of the simulated image.
        :param pixelsize: The size of one pixel of the simulated image (m). This determines how many pixels are skipped
                          between each laser application.
        :param pixeldwelltime: The time spent on each pixel of the simulated
                               image (s).
        :param p_ex: The power of the depletion beam (W).
        :param p_sted: The power of the STED beam (W).
        :param datamap_pixelsize: The size of one pixel of the datamap. This is the resolution at which the lasers are
                                  generated.
        :param pixel_list: List of pixels on which we apply the lasers. If None is passed, a normal raster scan is done.
        :returns: A 2D array of the new data map.

        TODO: correctly skip pixels when a non raster scan list is passed.
        (((: EN THÉORIE, ON S'EN CALISSE DE CETTE FONCTION LÀ MAINTENANT QUE BLEACH2 MARCHE :)))
        '''
        if pixel_list is None:
            print("No pixel list passed, Running a raster scan :)")
            pixel_list = utils.pixel_sampling(datamap, mode="all")
        __i_ex, __i_sted, _ = self.cache(pixelsize, data_pixelsize=datamap_pixelsize)

        photons_ex = self.fluo.get_photons(__i_ex * p_ex)
        k_ex = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)

        duty_cycle = self.sted.tau * self.sted.rate
        photons_sted = self.fluo.get_photons(__i_sted * p_sted * duty_cycle)
        k_sted = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)

        pad = photons_ex.shape[0] // 2 * 2
        h_size, w_size = datamap.shape[0] + pad, datamap.shape[1] + pad

        pixeldwelltime = numpy.asarray(pixeldwelltime)
        # vérifier si pixeldwelltime est un scalaire ou une matrice, si c'est un scalaire, transformer en matrice
        if pixeldwelltime.shape == ():
            pixeldwelltime = numpy.ones(datamap.shape) * pixeldwelltime
        else:
            # live j'assume que si je passe une matrice comme pixeldwelltime, elle est de la même forme que ma datamap,
            # ajouter des trucs pour vérifier que c'est bien le cas ici :)
            verif_array = numpy.asarray([1, 2, 3])
            if type(verif_array) != type(pixeldwelltime):
                # on va tu ever se rendre ici? qq lignes plus haut je transfo pdt en array... w/e
                raise Exception("pixeldwelltime parameter must be array type")
        pdtpad = numpy.pad(pixeldwelltime, pad // 2, mode="constant", constant_values=0)

        # à place de faire ça, je devrais faire comme je fais pour ma modif de pixelsize pour skipper certains pixels :)
        img_pixelsize_int, data_pixelsize_int = utils.pxsize_comp(pixelsize, datamap_pixelsize)
        ratio = img_pixelsize_int / data_pixelsize_int
        # set egal à datamap initiallement, certains pixels seront bleachés dépendant du ratio des pixelsize
        new_datamap = numpy.copy(datamap)
        # new_datamap = numpy.zeros(datamap.shape)
        previous_pixel = None
        for pixel in pixel_list:
            # fait le pixel skipping comme il faut pour une liste raster scan normale, mais ne fonctionne pas pour un
            # ordre abitraire / raster inversé, need to figure out why et faire marche ça pour get_signal aussi :)
            row = pixel[0]
            col = pixel[1]
            if previous_pixel is not None:
                if row - previous_pixel[0] < ratio and col - previous_pixel[1] < ratio:   # absolues?
                    continue

            pdt = pdtpad[row:row + pad + 1, col:col + pad + 1]
            prob_ex = numpy.prod(numpy.exp(-k_ex * pdt))
            prob_sted = numpy.prod(numpy.exp(-k_sted * pdt))
            new_datamap[row, col] = numpy.random.binomial(datamap[row, col], prob_ex * prob_sted)
            previous_pixel = pixel

        return new_datamap

    def am_i_centered(self, datamap, pixelsize, pixeldwelltime, p_ex, p_sted, datamap_pixelsize, pixel_list=None):
        """
        Le but de cette fonction est de vérifier si je me centre bien sur le pixel itéré lorsque je fais mon acquisition
        / bleaching avec mes matrices de lasers superposées à mes données :)
        """
        print("VOUS ÊTES DANS LA FONCTION POUR VÉRIFIER SI VOUS ÊTES BIEN CENTRÉ :)")
        if pixel_list is None:
            print("No pixel list passed, Running a raster scan :)")
            pixel_list = utils.pixel_sampling(datamap, mode="all")
        __i_ex, __i_sted, _ = self.cache(pixelsize, data_pixelsize=datamap_pixelsize)

        photons_ex = self.fluo.get_photons(__i_ex * p_ex)
        k_ex = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)

        duty_cycle = self.sted.tau * self.sted.rate
        photons_sted = self.fluo.get_photons(__i_sted * p_sted * duty_cycle)
        k_sted = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)

        pad = photons_ex.shape[0] // 2 * 2
        h_size, w_size = datamap.shape[0] + pad, datamap.shape[1] + pad

        pixeldwelltime = numpy.asarray(pixeldwelltime)
        # vérifier si pixeldwelltime est un scalaire ou une matrice, si c'est un scalaire, transformer en matrice
        if pixeldwelltime.shape == ():
            pixeldwelltime = numpy.ones(datamap.shape) * pixeldwelltime
            # je devrais tu juste mettre les pixels dans pixel_list à une valeur non nulle? ce serait ce qui est ici
            """pixeldwelltime_arr = numpy.zeros(datamap.shape)
            for (row, col) in pixel_list:
                pixeldwelltime_arr[row, col] += pixeldwelltime
            pixeldwelltime = pixeldwelltime_arr"""
        else:
            # live j'assume que si je passe une matrice comme pixeldwelltime, elle est de la même forme que ma datamap,
            # ajouter des trucs pour vérifier que c'est bien le cas ici :)
            verif_array = numpy.asarray([1, 2, 3])
            if type(verif_array) != type(pixeldwelltime):
                # on va tu ever se rendre ici? qq lignes plus haut je transfo pdt en array... w/e
                raise Exception("pixeldwelltime parameter must be array type")
        pdtpad = numpy.pad(pixeldwelltime, pad // 2, mode="constant", constant_values=0)

        # à place de faire ça, je devrais faire comme je fais pour ma modif de pixelsize pour skipper certains pixels :)
        img_pixelsize_int, data_pixelsize_int = utils.pxsize_comp(pixelsize, datamap_pixelsize)
        ratio = img_pixelsize_int / data_pixelsize_int

        traversed_array_padded = numpy.pad(datamap, pad // 2, mode="constant")
        traversed_array_roi = numpy.copy(datamap)
        acq_matrix = numpy.ones(photons_ex.shape)
        previous_pixel = None
        for (row, col) in pixel_list:
            # fait le pixel skipping comme il faut pour une liste raster scan normale, mais ne fonctionne pas pour un
            # ordre abitraire / raster inversé, need to figure out why et faire marche ça pour get_signal aussi :)
            if previous_pixel is not None:
                if row - previous_pixel[0] < ratio and col - previous_pixel[1] < ratio:   # absolues?
                    continue

            # setting up mes plots pour trouver le pixel itéré
            traversed_array_padded[row:row + pad + 1, col:col + pad + 1] += acq_matrix
            if row + pad // 2 == (row + row + pad + 1) // 2 and col + pad // 2 == (col + col + pad + 1) // 2:
                traversed_array_roi[row, col] += 1
                traversed_array_padded[(row + row + pad + 1) // 2, (col + col + pad + 1) // 2] += 1

            list_over = traversed_array_padded[row:(row + row + pad + 1) // 2, col]
            list_under = traversed_array_padded[(row + row + pad + 1) // 2 + 1:row + pad + 1, col]
            list_left = traversed_array_padded[row, col:(col + col + pad + 1) // 2]
            list_right = traversed_array_padded[row, (col + col + pad + 1) // 2 + 1:col + pad + 1]

            # plot mes shit
            fig, axes = pyplot.subplots(1, 2)

            padded_imshow = axes[0].imshow(traversed_array_padded)
            axes[0].set_title(f"Padded array, with region covered by acquisition array and original datamap highlighted"
                              f"\n"
                              f"Acquisition array shape = "
                              f"{traversed_array_padded[row:row + pad + 1, col:col + pad + 1].shape} \n"
                              f"Center of region covered by acquisition array also highlighted \n"
                              f"{len(list_over)} pixels over iterated pixel inside acquisition array, "
                              f"{len(list_under)} pixels under iterated pixel inside acquisition array, \n"
                              f"{len(list_left)} pixels left of iterated pixel inside acquisition array, "
                              f"{len(list_right)} pixels right of iterated pixel inside acquisition array")
            fig.colorbar(padded_imshow, ax=axes[0], fraction=0.04, pad=0.05)

            roi_imshow = axes[1].imshow(traversed_array_roi)
            axes[1].set_title(f"Pixel iterated over in the original datamap frame of reference")
            fig.colorbar(roi_imshow, ax=axes[1], fraction=0.04, pad=0.05)

            fig.suptitle(f'Iteration on pixel {(row, col)}', fontsize=16)

            figManager = pyplot.get_current_fig_manager()
            figManager.window.showMaximized()

            pyplot.show()

            # reverte les modifs que j'ai fait plus haut :)
            traversed_array_padded[row:row + pad + 1, col:col + pad + 1] -= 1
            if row + pad // 2 == (row + row + pad + 1) // 2 and col + pad // 2 == (col + col + pad + 1) // 2:
                traversed_array_roi[row, col] -= 1
                traversed_array_padded[(row + row + pad + 1) // 2, (col + col + pad + 1) // 2] -= 1

            previous_pixel = (row, col)

        traversed_array_padded = traversed_array_padded[int(pad / 2):-int(pad / 2), int(pad / 2):-int(pad / 2)]
        # print(f"iterated over {counter} pixels out of {datamap.shape[0] * datamap.shape[1]}")
        return traversed_array_padded

    def bleach2(self, datamap, pixelsize, pixeldwelltime, p_ex, p_sted, datamap_pixelsize, pixel_list=None):
        if pixel_list is None:
            print("No pixel list passed, Running a raster scan :)")
            pixel_list = utils.pixel_sampling(datamap, mode="all")
        __i_ex, __i_sted, _ = self.cache(pixelsize, data_pixelsize=datamap_pixelsize)

        photons_ex = self.fluo.get_photons(__i_ex * p_ex)
        k_ex = self.fluo.get_k_bleach(self.excitation.lambda_, photons_ex)

        duty_cycle = self.sted.tau * self.sted.rate
        photons_sted = self.fluo.get_photons(__i_sted * p_sted * duty_cycle)
        k_sted = self.fluo.get_k_bleach(self.sted.lambda_, photons_sted)

        pad = photons_ex.shape[0] // 2 * 2
        h_size, w_size = datamap.shape[0] + pad, datamap.shape[1] + pad

        pixeldwelltime = numpy.asarray(pixeldwelltime)
        # vérifier si pixeldwelltime est un scalaire ou une matrice, si c'est un scalaire, transformer en matrice
        if pixeldwelltime.shape == ():
            pixeldwelltime = numpy.ones(datamap.shape) * pixeldwelltime
        else:
            # live j'assume que si je passe une matrice comme pixeldwelltime, elle est de la même forme que ma datamap,
            # ajouter des trucs pour vérifier que c'est bien le cas ici :)
            verif_array = numpy.asarray([1, 2, 3])
            if type(verif_array) != type(pixeldwelltime):
                # on va tu ever se rendre ici? qq lignes plus haut je transfo pdt en array... w/e
                raise Exception("pixeldwelltime parameter must be array type")
        pdtpad = numpy.pad(pixeldwelltime, pad // 2, mode="constant", constant_values=0)

        # à place de faire ça, je devrais faire comme je fais pour ma modif de pixelsize pour skipper certains pixels :)
        img_pixelsize_int, data_pixelsize_int = utils.pxsize_comp(pixelsize, datamap_pixelsize)
        ratio = img_pixelsize_int / data_pixelsize_int

        prob_ex = numpy.pad((numpy.ones(datamap.shape)).astype(float), pad // 2, mode="constant")
        prob_sted = numpy.pad((numpy.ones(datamap.shape)).astype(float), pad // 2, mode="constant")
        previous_pixel = None
        for (row, col) in pixel_list:
            # fait le pixel skipping comme il faut pour une liste raster scan normale, mais ne fonctionne pas pour un
            # ordre abitraire / raster inversé, need to figure out why et faire marche ça pour get_signal aussi :)
            if previous_pixel is not None:
                if row - previous_pixel[0] < ratio and col - previous_pixel[1] < ratio:   # absolues?
                    continue

            pdt = pdtpad[row:row + pad + 1, col:col + pad + 1]
            prob_ex[row:row + pad + 1, col:col + pad + 1] *= numpy.exp(-k_ex * pdt)
            prob_sted[row:row + pad + 1, col:col + pad + 1] *= numpy.exp(-k_sted * pdt)
            previous_pixel = (row, col)

        datamap = datamap.astype(int)   # sinon il lance une erreur, à vérifier si c'est legit de faire ça
        prob_ex = prob_ex[int(pad / 2):-int(pad / 2), int(pad / 2):-int(pad / 2)]
        prob_sted = prob_sted[int(pad / 2):-int(pad / 2), int(pad / 2):-int(pad / 2)]
        new_datamap = numpy.random.binomial(datamap, prob_ex * prob_sted)

        return new_datamap

    def laser_dans_face(self, datamap, pixelsize, pixeldwelltime, p_ex, p_sted, datamap_pixelsize, pixel_list=None):
        """
        Test function to see how much laser each pixel receives in the face ;)
        """
        # variable qui sera retournée, chaque pixel contiendra la quantité de photons reçus par la datamap à cause
        # des lasers
        laser_received = numpy.zeros(datamap.shape)

        # effective intensity across pixels (W)
        # acquisition gaussian is computed using data_pixelsize
        if datamap_pixelsize is None:
            print(f"datamap_pixelsize is None")
            effective = self.get_effective(pixelsize, p_ex, p_sted)
        else:
            print(f"datamap_pixelsize = {datamap_pixelsize}")
            effective = self.get_effective(datamap_pixelsize, p_ex, p_sted)

        # variable qui sera retournée, chaque pixel contiendra la quantité de photons reçus par la datamap à cause
        # des lasers
        # dans stack, on utilise data pour calculer le pad, qui est effective
        img_pixelsize_int, data_pixelsize_int = utils.pxsize_comp(pixelsize, datamap_pixelsize)
        ratio = int(img_pixelsize_int / data_pixelsize_int)
        h_pad, w_pad = int(effective.shape[0] / 2) * 2, int(effective.shape[1] / 2) * 2
        laser_received = numpy.zeros(
            (int(datamap.shape[0] / ratio) + h_pad, int(datamap.shape[1] / ratio) + w_pad))

        __i_ex, __i_sted, psf_det = self.cache(pixelsize, datamap_pixelsize)

        """# plot les cache shit vs effective pour voir si les pts sont du même ordre de grandeur
        fig, axes = pyplot.subplots(2, 2)

        ex_imshow = axes[0, 0].imshow(__i_ex * p_ex, interpolation="nearest")
        axes[0, 0].set_title(f"Excitation beam, shape = {__i_ex.shape}")
        fig.colorbar(ex_imshow, ax=axes[0, 0], fraction=0.04, pad=0.05)

        sted_imshow = axes[0, 1].imshow(__i_sted * p_sted, interpolation="nearest")
        axes[0, 1].set_title(f"STED beam, shape = {__i_sted.shape}")
        fig.colorbar(sted_imshow, ax=axes[0, 1], fraction=0.04, pad=0.05)

        psf_imshow = axes[1, 0].imshow(psf_det, interpolation="nearest")
        axes[1, 0].set_title(f"PSF det, shape = {psf_det.shape}")
        fig.colorbar(psf_imshow, ax=axes[1, 0], fraction=0.04, pad=0.05)

        effective_imshow = axes[1, 1].imshow(effective, interpolation="nearest")
        axes[1, 1].set_title(f"Effective beam, shape = {effective.shape}")
        fig.colorbar(effective_imshow, ax=axes[1, 1], fraction=0.04, pad=0.05)

        pyplot.show()
        exit()"""

        # verif bullshit, to remove eventually
        iterated_pixels = numpy.zeros(datamap.shape)

        # si aucune pixel_list n'est passée, on fait un raster scan complet :)
        if pixel_list is None:
            pixel_list = utils.pixel_sampling(datamap, mode="all")
        for pixel in pixel_list:
            row = pixel[0]
            col = pixel[1]
            # 1 W = 1 J / s, donc si je multiplie par le pixeldwelltime, mon output va être en J :)
            laser_received[row:row+h_pad+1, col:col+w_pad+1] += (__i_ex * p_ex + __i_sted * p_sted) * pixeldwelltime
            iterated_pixels[row, col] = 1
        # k cool je regarde la qté de laser que chaque pixel reçoit avec effective
        # ceci me donne une idée du laser reçu quand get_signal est utilisé,
        # maintenant je veux regarder ce que la méthode d'application du laser de bleach ferait comme résultat?

        return laser_received[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)], iterated_pixels