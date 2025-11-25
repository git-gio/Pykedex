import os

from find_screen import find_screen

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    result_folder = os.path.join(dirname, 'processed_images/articuno/noName/')

    # assign directory
    directory = os.path.join(dirname, 'dataset/articuno/noName/')

    # iterate over files in
    # that directory
    i = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            i += 1
            save_path = result_folder + str(i) + '.png'
            result = find_screen(query_path=f, save_path=save_path, verbose=True)
            if result == False:
                print('no contour for image ' + f)
    quit()

