# import the necessary packages
from scipy.spatial import distance as dist


class Searcher:
    def __init__(self, index):
        '''
        Assumption that the index is a standard Python dictionary with the name of the Pokemon
        as the key and the shape features
        (i.e. a list of numbers used to quantify the shape and outline of the Pokemon) as the value.
        :param index: The index that the search function will be searching over.
        '''
        self.index = index

    def search(self, queryFeatures, verbose=False):
        '''
         This method takes a single parameter — our query features.
        :param queryFeatures: To be compared to every value (feature vector) in the index.
        :return: result of the comparison.
        '''
        # initialize our dictionary of results
        results = {}
        # loop over the processed_images in our index
        for (poke_name, features) in self.index.items():
            if verbose:
                print('features: ' + str(features))
                print('\n')
                print('query: ' + str(queryFeatures))
            # compute the distance between the query features
            # and features in our index, then update the results
            distance = dist.euclidean(queryFeatures, features)
            results[poke_name] = distance
        # sort results, smaller distance indicates higher similarity
        results = sorted([(v, poke_name) for (poke_name, v) in results.items()])
        if verbose:
            print('Pokemon: ' + poke_name + ', distance: ' + str(distance))
        # return the results
        return results
