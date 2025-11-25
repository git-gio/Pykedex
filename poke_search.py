# import the necessary packages
import pickle

from searcher import Searcher
from utils import *
from zernikemoments import ZernikeMoments
import numpy as np
import cv2
import os


def advanced_search(thresh, template):
    # thresh = resize(thresh, height=56, width=56)
    # template1 = resize(template1, height=56, width=56)
    # template2 = resize(template2, height=56, width=56)

    # Apply template Matching
    # res1 = cv2.matchTemplate(thresh, template1, cv2.TM_CCOEFF_NORMED)
    # min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
    # top_left1 = max_loc1
    # w1, h1 = template1.shape[::-1]
    # bottom_right1 = (top_left1[0] + w1, top_left1[1] + h1)
    # rectangle1 = cv2.rectangle(thresh, top_left1, bottom_right1, 255, 2)

    res = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left2 = max_loc2
    # w2, h2 = template2.shape[::-1]
    # bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)
    # rectangle2 = cv2.rectangle(thresh, top_left2, bottom_right2, 255, 2)
    # cv2.imshow('2', resize(rectangle2, height=300))
    print(('match: ' + str(max_val)))
    #
    # # cv2.imshow('1', resize(rectangle1, height=300))
    # # cv2.imshow('2', resize(rectangle2, height=300))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # print(('dragonite: ' + str(max_val1) + 'charizard: ' + str(max_val2)))

    return max_val


def poke_search(index_folder, query_path, true_pokemon_name=None, hu=False, verbose=False, verbose_search=False):
    dirname = os.path.dirname(__file__)

    try:
        if hu:
            print('using HuIndex')
            index_path = index_folder + 'huIndex.pkl'
        else:
            index_path = index_folder + 'index.pkl'
        index = open(index_path, "rb").read()
    except FileNotFoundError:
        raise Exception('Index not found!')
    index = pickle.loads(index)
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

    if hu:
        # next 3 lines are for Hu Moments
        print('using HuMoments')
        moments = cv2.moments(outline)
        queryFeatures = cv2.HuMoments(moments).flatten()
    else:
        # compute Zernike moments to characterize the shape of Pokémon outline
        desc = ZernikeMoments(radius=radius)  # , cm=(int(center[0]), int(center[1])))
        queryFeatures = desc.describe(outline)
    # perform the search to identify the Pokémon
    searcher = Searcher(index)
    results = searcher.search(queryFeatures, verbose=verbose_search)

    if verbose:
        # show our processed_images
        cv2.imshow("image", resize(img, height=300))
        cv2.imshow("gray", resize(gray, height=300))
        cv2.imshow("blur", resize(blur, height=300))
        cv2.imshow("thresh", resize(thresh, height=300))
        cv2.imshow("outline", resize(outline, height=300))
        cv2.imshow('circle', resize(circle, height=300))

        recog_dir = os.path.join(dirname, 'utils/immagini/riconoscimento/')

        cv2.imwrite(recog_dir + 'img.png', img)
        cv2.imwrite(recog_dir + 'thresh.png', thresh)
        cv2.imwrite(recog_dir + 'outline.png', outline)
        cv2.imwrite(recog_dir + 'circle.png', circle)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("That pokemon is %s" % results[0][1].upper() + ' with a score of %s' % results[0][0])
        print("Second place is %s" % results[1][1].upper() + ' with a score of %s' % results[1][0])
        print('radius: ' + str(radius))

    true_pokemon = list
    if true_pokemon_name is not None:
        for pokemon in results:
            if pokemon[1] == true_pokemon_name:
                true_pokemon = pokemon
                index = results.index(pokemon) + 1
        if verbose:
            print("True pokemon is %s" % true_pokemon[1].upper() + ' with a score of %s' % true_pokemon[
                0] + ' in position %s' % index)
    else:
        # true_pokemon[1] = 'MissingNo'
        true_pokemon = results[0][1]

    estimated_pokemon_name = results[0][1].upper()
    difficult_pokemons = {
        'CHARIZARD': ['DRAGONITE', 'SEEL', 'TAUROS', 'GOLDUCK', 'PSYDUCK'],
        'ARTICUNO': ['BELLSPROUT', 'MEW', 'MEWTWO', 'GOLDUCK', 'PERSIAN', 'MAROWAK', 'PRIMEAPE', 'DRAGONAIR', 'KADABRA', 'LAPRAS',
                     'KOFFING'],
        'VOLTORB': ['WEEDLE', 'METAPOD', 'RHYDON', 'GOLDUCK']
    }
    for difficult_pokemon in difficult_pokemons:
        for pokemon_name in difficult_pokemons[difficult_pokemon]:
            if estimated_pokemon_name == pokemon_name:
                print('ciao!!!')
                template = cv2.imread(
                    'C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/sprites/template_gen1/{0}.png'.format(
                        difficult_pokemon.lower()), cv2.IMREAD_GRAYSCALE)
                if advanced_search(thresh, template) <= 0.55:
                    return results[0][1].upper(), true_pokemon[1].upper()
                else:
                    return difficult_pokemon, true_pokemon[1].upper()

    return results[0][1].upper(), true_pokemon[1].upper()


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)

    index_folder = os.path.join(dirname, 'indexes/')
    query_path = os.path.join(dirname, 'processed_images/cropped.png')

    poke_search(index_folder=index_folder,
                query_path=query_path, verbose=True)
    quit()
