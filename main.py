# import the necessary packages
import pickle

from searcher import Searcher
from utils import *
from zernikemoments import ZernikeMoments
import numpy as np
import cv2
import glob


# load the index
index_path = '../indexes/index.pkl'
try:
    index = open(index_path, "rb").read()
except FileNotFoundError:
    raise Exception('Index not found!')
index = pickle.loads(index)

# load the image
query_path = 'C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/processed_images/squirtle/named/3.png'
try:
    img = cv2.imread(query_path)
except FileNotFoundError:
    raise Exception('Pokemon image not found!')

img = resize(img, height=56)
gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

# blurring solves some problems, adaptive thresh too sensitive
blur = cv2.bilateralFilter(gray.copy(), 11, 17, 17)
# blur = cv2.GaussianBlur(gray, (3, 3), 1.5)

# threshold the image
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 4)
# ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh = cv2.copyMakeBorder(thresh, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=0)

# initialize the outline image, find the outermost contours (the outline) of the Pokémon, then draw it
outline = np.zeros(thresh.shape, dtype="uint8")

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

outline = cv2.drawContours(outline, [cnts], -1, 255, -1)

# find the radius for the zernike moments
center, radius = cv2.minEnclosingCircle(cnts)
circle = cv2.circle(outline.copy(), (int(center[0]), int(center[1])), int(radius), 255, 2)

# compute Zernike moments to characterize the shape of Pokémon outline
desc = ZernikeMoments(radius=radius)#, cm=(int(center[0]), int(center[1])))
queryFeatures = desc.describe(outline)
# next 3 lines are for Hu Moments
#  moments = cv2.moments(outline)
# queryFeatures = cv2.HuMoments(moments).flatten()
# perform the search to identify the Pokémon
searcher = Searcher(index)
results = searcher.search(queryFeatures, verbose=True)
print("That pokemon is %s" % results[0][1].upper() + ' with a score of %s' % results[0][0])
print("Second place is %s" % results[1][1].upper() + ' with a score of %s' % results[1][0])
true_pokemon = list
for pokemon in results:
    if pokemon[1] == 'caterpie':
        true_pokemon = pokemon
        index = results.index(pokemon) + 1
print("True pokemon is %s" % true_pokemon[1].upper() + ' with a score of %s' % true_pokemon[
    0] + ' in position %s' % index)
print('radius: ' + str(radius))
# show our processed_images
cv2.imshow("image", resize(img, height=300))
cv2.imshow("gray", resize(gray, height=300))
cv2.imshow("blur", resize(blur, height=300))
cv2.imshow("thresh", resize(thresh, height=300))
cv2.imshow("outline", resize(outline, height=300))
cv2.imshow('circle', resize(circle, height=300))
cv2.waitKey(0)
cv2.destroyAllWindows()
