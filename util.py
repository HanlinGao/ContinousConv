from grispy import GriSPy
import numpy as np


def neighbor_search(centres, data, search_radius=4):
    '''
    find neighbors for centre points, use indexes in the data to represent neighbors
    :param centres: dynamic particles
    :param data: search area
    :param search_radius: should be equal to kernel size
    :return: [[indx1, indx2, ..], [indx1, indx2, ...]]
    '''
    gsp = GriSPy(data)
    bubble_dist, bubble_ind = gsp.bubble_neighbors(
        centres, distance_upper_bound=search_radius
    )
    return bubble_ind
