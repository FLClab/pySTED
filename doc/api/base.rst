Basic components
================
.. automodule:: pysted.base

GaussianBeam
------------
.. autoclass:: pysted.base.GaussianBeam
	
	.. automethod:: pysted.base.GaussianBeam.get_intensity(power, f, n, na, transmission, pixelsize)

DonutBeam
---------
.. autoclass:: pysted.base.DonutBeam
	
	.. automethod:: pysted.base.DonutBeam.get_intensity(power, f, n, na, transmission, pixelsize)

Detector
--------
.. autoclass:: pysted.base.Detector
	
	.. automethod:: pysted.base.Detector.get_detection_psf(lambda_, psf, na, transmission, pixelsize)
	
	.. automethod:: pysted.base.Detector.get_signal(nb_photons, dwelltime)
	
Objective
---------
.. autoclass:: pysted.base.Objective
	
	.. automethod:: pysted.base.Objective.get_transmission(wavelength)

Fluorescence
------------
.. autoclass:: pysted.base.Fluorescence
	
	.. automethod:: pysted.base.Fluorescence.get_psf(na, pixelsize)

Microscope
----------
.. autoclass:: pysted.base.Microscope
	
	.. automethod:: pysted.base.Microscope.is_cached(pixelsize)
	
	.. automethod:: pysted.base.Microscope.cache(pixelsize)
	
	.. automethod:: pysted.base.Microscope.clear_cache()
	
	.. automethod:: pysted.base.Microscope.get_effective(datamap, pixelsize, pixeldwelltime, p_ex, p_sted)
	
	.. automethod:: pysted.base.Microscope.get_signal(datamap, pixelsize, pixeldwelltime, p_ex, p_sted)
	
	.. automethod:: pysted.base.Microscope.bleach(datamap, pixelsize, pixeldwelltime, p_ex, p_sted, c_ex, c_sted)
	

