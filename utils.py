import os
import cv2
import numpy as np
from mahotas import center_of_mass
from mahotas.features import _zernike


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    # print(v)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # print(str(lower) + ' e ' + str(upper))
    edged = cv2.Canny(image, lower, upper, apertureSize=3, L2gradient=True)  # L2gradient True --> better results
    # cv2.waitKey(0)
    return edged


def list_images(basePath, contains=None):
    # return the set of files that are valid
    image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    if h == height or w == width:
        return image

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
                         "otherwise OpenCV changed their cv2.findContours return "
                         "signature yet again. Refer to OpenCV's documentation "
                         "in that case"))

    # return the actual contours array
    return cnts


def get_opencv_major_version(lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib

    # return the major version number
    return int(lib.__version__.split(".")[0])


def is_cv2(or_better=False):
    # grab the OpenCV major version number
    major = get_opencv_major_version()

    # check to see if we are using *at least* OpenCV 2
    if or_better:
        return major >= 2

    # otherwise we want to check for *strictly* OpenCV 2
    return major == 2


def my_zernike_moments(im, radius, degree=8, cm=None, verboseCM=False):
    """
    zvalues = zernike_moments(im, radius, degree=8, cm={center_of_mass(im)})

    Zernike moments through ``degree``. These are computed on a circle of
    radius ``radius`` centered around ``cm`` (or the center of mass of the
    image, if the ``cm`` argument is not used).

    Returns a vector of absolute Zernike moments through ``degree`` for the
    image ``im``.

    Parameters
    ----------
    im : 2-ndarray
        input image
    radius : integer
        the maximum radius for the Zernike polynomials, in pixels. Note that
        the area outside the circle (centered on center of mass) defined by
        this radius is ignored.
    degree : integer, optional
        Maximum degree to use (default: 8)
    cm : pair of floats, optional
        the centre of mass to use. By default, uses the image's centre of mass.

    Returns
    -------
    zvalues : 1-ndarray of floats
        Zernike moments

    References
    ----------
    Adapted from Teague, MR. (1980). Image Analysis via the General Theory of Moments.  J.
    Opt. Soc. Am. 70(8):920-930. I use this version whenever i want to get more info on the CM
    """
    zvalues = []
    if cm is None:
        c0, c1 = center_of_mass(im)
        if verboseCM is not False:
            circle = cv2.circle(im.copy(), (int(c0), int(c1)), int(radius), 255, 2)
            cv2.imshow('circle', resize(circle, height=300))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        c0, c1 = cm

    Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
    P = im.ravel()

    def rescale(C, centre):
        Cn = C.astype(np.double)
        Cn -= centre
        Cn /= radius
        return Cn.ravel()

    Yn = rescale(Y, c0)
    Xn = rescale(X, c1)

    Dn = Xn ** 2
    Dn += Yn ** 2
    np.sqrt(Dn, Dn)
    np.maximum(Dn, 1e-9, out=Dn)
    k = (Dn <= 1.)
    k &= (P > 0)

    frac_center = np.array(P[k], np.double)
    frac_center = frac_center.ravel()
    frac_center /= frac_center.sum()
    Yn = Yn[k]
    Xn = Xn[k]
    Dn = Dn[k]
    An = np.empty(Yn.shape, np.complex_)
    An.real = (Xn / Dn)
    An.imag = (Yn / Dn)

    Ans = [An ** p for p in range(2, degree + 2)]
    Ans.insert(0, An)  # An**1
    Ans.insert(0, np.ones_like(An))  # An**0
    for n in range(degree + 1):
        for l in range(n + 1):
            if (n - l) % 2 == 0:
                z = _zernike.znl(Dn, Ans[l], frac_center, n, l)
                zvalues.append(abs(z))
    return np.array(zvalues)
