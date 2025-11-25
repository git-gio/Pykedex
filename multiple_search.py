import os
from datetime import datetime

from poke_search import poke_search

if __name__ == "__main__":

    dirname = os.path.dirname(__file__)

    # load the index
    index_folder = os.path.join(dirname, 'indexes/')
    processed_path = os.path.join(dirname, 'processed_images/')

    results_path = os.path.join(dirname, 'results/')
    file = results_path + 'results_' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    results = file + '.txt'

    # example with caterpie
    # enter caterpie folder
    for pokemon in os.listdir(processed_path):
        name_path = os.path.join(processed_path, pokemon)
        # check if we are opening a directory
        if os.path.isdir(name_path):
            # enter named/noName folder
            for name_dir in os.listdir(name_path):
                crops_path = os.path.join(name_path, name_dir)
                # get caterpie cropped images
                i = 0

                right = 0
                wrong = 0
                for image in os.listdir(crops_path):
                    query_path = os.path.join(crops_path, image)
                    # checking if the image is a file
                    if os.path.isfile(query_path):
                        i += 1
                        # where the magic happens
                        estimate, expected = poke_search(index_folder=index_folder, query_path=query_path,
                                                         true_pokemon_name=pokemon, hu=True,
                                                         verbose=False, verbose_search=True)
                        if estimate == expected:
                            right += 1
                        else:
                            wrong += 1
                        # write the result
                        with open(results, "a") as f:
                            f.write('test on ' + pokemon + str(i) + name_dir + ': estimated: ' + estimate + '; expected: ' + expected + '\n')
                    else:
                        raise Exception('Not a file! Path: ' + query_path)
                if (right+wrong) != 0:
                    percentage = right/(right+wrong)
                    with open(results, "a") as f:
                        f.write('Percentage of passed tests: ' + str(percentage) + '\n\n')
        else:
            continue
    quit()
    # # load the image
    # name = 'noName'
    # true_pokemon_name = 'charizard'
    # directory = crops_path + '{0}/{1}'.format(
    #     true_pokemon_name, name)
    # # sys.stdout = open(results, 'w')


