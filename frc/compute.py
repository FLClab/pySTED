
import numpy
import time

# import fourierRingCorrelation

def Hamming(w, h):
    ''' This function computes the Hamming function
    :param w: The widht
    :param h: The height
    :reuturns : The hamming window
    '''
    alpha = 0.54
    beta = 1 - alpha

    xv = (alpha - beta * numpy.cos(2 * numpy.pi / (w - 1) * numpy.arange(0, w))).reshape(1, len(numpy.arange(0, w)))
    yv = (alpha - beta * numpy.cos(2 * numpy.pi / (h - 1) * numpy.arange(0, h))).reshape(1, len(numpy.arange(0, h)))

    return (yv.T).dot(xv)

def fourier_shell_corr_giuseppe(img1, img2, precomputed_masks=None):
    ''' This function computes the fourrier shell correlation from two images
    :param img1: A 2D numpy array
    :param img2: A 2D numpy array
    :returns : The fourier shell correlation
    '''
    Hm = Hamming(img1.shape[1], img1.shape[0])
    fimg1 = numpy.fft.fftshift(numpy.fft.fft2(img1 * Hm))
    fimg2 = numpy.fft.fftshift(numpy.fft.fft2(img2 * Hm))

    start = time.time()
    fscList, nPx = [], []
    h, w = fimg1.shape
    yc, xc = int((h + 1) / 2) + 1, int((w + 1) / 2) + 1
    rmax = min([w - xc, h - yc])
    conj_fimg2 = numpy.conjugate(fimg2)
    for r in range(rmax):

        if isinstance(precomputed_masks, type(None)):
            x, y = numpy.ogrid[0:fimg1.shape[0], 0:fimg1.shape[1]]
            maskA = numpy.sqrt((x - xc)**2 + (y - yc)**2) >= r - 0.5
            maskB = numpy.sqrt((x - xc)**2 + (y - yc)**2) < r + 0.5
            mask = maskA * maskB
        else:
            mask = precomputed_masks[r]

        corr = numpy.sum(fimg1[mask] * conj_fimg2[mask])
        absA = numpy.sum(numpy.abs(fimg1[mask]**2))
        absB = numpy.sum(numpy.abs(fimg2[mask]**2))

        fscList.append(numpy.abs(corr) / numpy.sqrt(absA * absB))
        nPx.append(mask.sum())

    # start = time.time()
    # fscList, nPx = [], []
    # h, w = fimg1.shape
    # yc, xc = int((h + 1) / 2) + 1, int((w + 1) / 2) + 1
    # rmax = min([w - xc, h - yc])
    # for r in range(rmax):
    #     corr = 0
    #     absA = 0
    #     absB = 0
    #     n = 0
    #     for x in range(w):
    #         for y in range(h):
    #             if (numpy.sqrt((x - xc)**2 + (y - yc)**2) >= r - 0.5) & (numpy.sqrt((x - xc)**2 + (y - yc)**2) < r + 0.5):
    #                 corr = corr + fimg1[y, x] * numpy.conj(fimg2[y, x])
    #                 absA = absA + numpy.abs(fimg1[y, x]**2)
    #                 absB = absB + numpy.abs(fimg2[y, x]**2)
    #
    #                 n += 1
    #     fscList.append(numpy.abs(corr) / numpy.sqrt(absA * absB))
    #     nPx.append(n)
    # print("loops", time.time() - start)

    return numpy.array(fscList), numpy.array(nPx)

def fourier_shell_corr_giuseppe_vectorized(img1, img2, precomputed_masks=None):
    ''' This function computes the fourrier shell correlation from two images
    :param img1: A 2D numpy array
    :param img2: A 2D numpy array
    :returns : The fourier shell correlation
    '''
    Hm = Hamming(img1.shape[1], img1.shape[0])
    fimg1 = numpy.fft.fftshift(numpy.fft.fft2(img1 * Hm))
    fimg2 = numpy.fft.fftshift(numpy.fft.fft2(img2 * Hm))

    start = time.time()
    fscList, nPx = [], []
    h, w = fimg1.shape
    yc, xc = int((h + 1) / 2) + 1, int((w + 1) / 2) + 1
    rmax = min([w - xc, h - yc])
    corr = fimg1 * numpy.conjugate(fimg2)
    absA = numpy.abs(fimg1 ** 2)
    absB = numpy.abs(fimg2 ** 2)

    if not isinstance(precomputed_masks, type(None)):
        corr = numpy.sum(precomputed_masks * corr[numpy.newaxis], axis=(-2, -1))
        absA = numpy.sum(precomputed_masks * absA[numpy.newaxis], axis=(-2, -1))
        absB = numpy.sum(precomputed_masks * absB[numpy.newaxis], axis=(-2, -1))

        fscList = numpy.abs(corr) / numpy.sqrt(absA * absB)
        nPx = numpy.sum(precomputed_masks, axis=(-2, -1))

    # for r in range(rmax):
    #
    #     if isinstance(precomputed_masks, type(None)):
    #         x, y = numpy.ogrid[0:fimg1.shape[0], 0:fimg1.shape[1]]
    #         maskA = numpy.sqrt((x - xc)**2 + (y - yc)**2) >= r - 0.5
    #         maskB = numpy.sqrt((x - xc)**2 + (y - yc)**2) < r + 0.5
    #         mask = maskA * maskB
    #     else:
    #         mask = precomputed_masks[r]
    #
    #     corr = numpy.sum(fimg1[mask] * conj_fimg2[mask])
    #     absA = numpy.sum(numpy.abs(fimg1[mask]**2))
    #     absB = numpy.sum(numpy.abs(fimg2[mask]**2))
    #
    #     fscList.append(numpy.abs(corr) / numpy.sqrt(absA * absB))
    #     nPx.append(mask.sum())
    return fscList, nPx, time.time() - start

"""def c_frc(img1, img2):
    ''' This function computes the fourrier shell correlation from two images
    :param img1: A 2D numpy array
    :param img2: A 2D numpy array
    :returns : The fourier shell correlation
    '''
    Hm = Hamming(img1.shape[1], img1.shape[0])
    fimg1 = numpy.fft.fftshift(numpy.fft.fft2(img1 * Hm))
    fimg2 = numpy.fft.fftshift(numpy.fft.fft2(img2 * Hm))

    fscList, nPx = [], []
    h, w = fimg1.shape
    yc, xc = int((h + 1) / 2) + 1, int((w + 1) / 2) + 1
    rmax = min([w - xc, h - yc])

    start = time.time()
    fscList, nPx = fourierRingCorrelation.return_result(fimg1, fimg2, h, w, rmax)
    print("cloop", time.time() - start)


    start = time.time()
    fscList, nPx = [], []
    h, w = fimg1.shape
    yc, xc = int((h + 1) / 2) + 1, int((w + 1) / 2) + 1
    rmax = min([w - xc, h - yc])
    for r in range(rmax):
        corr = 0
        absA = 0
        absB = 0
        n = 0
        for x in range(w):
            for y in range(h):
                if (numpy.sqrt((x - xc)**2 + (y - yc)**2) >= r - 0.5) & (numpy.sqrt((x - xc)**2 + (y - yc)**2) < r + 0.5):
                    corr = corr + fimg1[y, x] * numpy.conj(fimg2[y, x])
                    absA = absA + numpy.abs(fimg1[y, x]**2)
                    absB = absB + numpy.abs(fimg2[y, x]**2)

                    n += 1
        fscList.append(numpy.abs(corr) / numpy.sqrt(absA * absB))
        nPx.append(n)
    print("loops", time.time() - start)

    return numpy.array(fscList), numpy.array(nPx)"""

def precompute(im):
    h, w = im.shape
    yc, xc = int((h + 1) / 2) + 1, int((w + 1) / 2) + 1
    rmax = min([w - xc, h - yc])

    output = []
    for r in range(rmax):

        x, y = numpy.ogrid[0:im.shape[0], 0:im.shape[1]]
        maskA = numpy.sqrt((x - xc)**2 + (y - yc)**2) >= r - 0.5
        maskB = numpy.sqrt((x - xc)**2 + (y - yc)**2) < r + 0.5
        mask = maskA * maskB

        output.append(mask)
    return numpy.array(output)


if __name__ == "__main__":

    im1, im2 = numpy.random.rand(2, 512, 512)
    precomputed_masks = precompute(im1)

    # all_times = []
    # for i in range(10):
    #     fsc, npx, elapsed = fourier_shell_corr_giuseppe(im1, im2, precomputed_masks=None)
    #     all_times.append(elapsed)
    # print("Not precomputed", numpy.mean(all_times), numpy.std(all_times))

    all_times = []
    for i in range(1):
        fsc, npx, elapsed = fourier_shell_corr_giuseppe_vectorized(im1, im2, precomputed_masks=precomputed_masks)
        print(fsc)
        all_times.append(elapsed)
    print("Precomputed", numpy.mean(all_times), numpy.std(all_times))

    all_times = []
    for i in range(1):
        fsc, npx, elapsed = fourier_shell_corr_giuseppe(im1, im2, precomputed_masks=precomputed_masks)
        all_times.append(elapsed)
        print(fsc)
    print("Precomputed", numpy.mean(all_times), numpy.std(all_times))
    # print(fsc)
    # fsc, npx = c_frc(im1, im2)
    # print(fsc)
