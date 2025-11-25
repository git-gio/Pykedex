# import the necessary packages
import pickle
import numpy as np
import cv2
import os

from zernikemoments import ZernikeMoments
from utils import list_images, grab_contours, resize

dirname = os.path.dirname(__file__)
sprites_dir = os.path.join(dirname, 'sprites/sprites_gen1/')

# initialize our descriptor (Zernike Moments with a radius of 21 used to characterize the shape of our Pokémon) and
# our index dictionary
index = {}

# loop over the sprite processed_images
for spritePath in list_images(sprites_dir):
    # parse out the Pokémon name, then load the image and
    # convert it to grayscale
    first_path = spritePath[spritePath.rfind("/") + 1:]
    pokemon_name = first_path[first_path.rfind("\\") + 1:].replace(".png", "")
    img = cv2.imread(spritePath)
    image = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # pad the image with extra white pixels to ensure the edges of the Pokémon are not up against the borders
    # of the image
    img = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    # invert the image and threshold it
    # some methods to thresh --> Otsu == line "bitwise_not + thresh[thresh > 0] = 255" (this is better for the
    # cnts passage) == simple250
    thresh = cv2.bitwise_not(img)
    thresh[thresh > 0] = 255

    template_name = sprites_dir + pokemon_name + '.png'

    # write the python object (dict) to pickle file
    # cv2.imwrite(template_name, thresh)

    # initialize the outline image, find the outermost contours (the outline) of the Pokémon, then draw it
    outline = np.zeros(img.shape, dtype="uint8")
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    outline = cv2.drawContours(outline, [cnts], -1, 255, -1)
    center, radius = cv2.minEnclosingCircle(cnts)
    circle = cv2.circle(outline.copy(), (int(center[0]), int(center[1])), int(radius), 255, 2)
    # if pokemon_name == "articuno":
    #     template_name = template_name + '_thresh.png'
    #    cv2.imwrite(template_name, thresh)
    #     print('pokemon: ' + str(pokemon_name) + ', radius: ' + str(radius))
    #     cv2.imshow('og image', resize(img, height=300))
    #     cv2.imshow('thresh', resize(thresh, height=300))
    #     cv2.imshow('outline', resize(outline, height=300))
    #     cv2.imshow('circle', resize(circle, height=300))
    #     cv2.imwrite('C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/circle.png', circle)
    #     # verboseCM = True
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # compute Zernike moments to characterize the shape
    # of Pokémon outline, then update the index
    desc = ZernikeMoments(radius=radius)#, verboseCM=verboseCM)#, cm=(int(center[0]), int(center[1])))
    moments = desc.describe(outline)
    # huMoments_start = cv2.moments(outline)
    # moments = cv2.HuMoments(huMoments_start).flatten()
    index[pokemon_name] = moments

# write the index to file
f = open("indexes/index.pkl", "wb")

# write the python object (dict) to pickle file
pickle.dump(index, f)

# close file
f.close()
quit()
