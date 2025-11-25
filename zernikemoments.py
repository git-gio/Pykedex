# import the necessary packages
import mahotas

import utils


class ZernikeMoments:
    def __init__(self, radius, cm=None, verboseCM=False):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius
        self.center = cm
        self.verboseCM = verboseCM

    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(im=image, radius=self.radius, degree=8, cm=self.center)
        # return utils.zernike_moments(im=image, radius=self.radius, degree=8, cm=self.center, verboseCM=self.verboseCM)
        # degree means more complexity
