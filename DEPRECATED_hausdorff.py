# encoding=utf-8

import numpy as np
from scipy.spatial.distance import cdist


def hausdorff(C1, C2):
    assert type(C2) is np.ndarray, "A is not a numpy.ndarray!"
    assert type(C1) is np.ndarray, "B is not a numpy.ndarray!"

    # C1 = C1.reshape((1, -1))
    # C2 = C2.reshape((1, -1))

    D = cdist(C1, C2, 'euclidean')
    # D = euclidean(C1, C2)

    # print "C1 is", C1
    # print "C2 is", C2
    # print "Distance is ", D
    # exit()

    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))

    # print "Min of axis 1 is", np.min(D, axis=1)
    # print "Min of axis 0 is", np.min(D, axis=0)
    #
    # print "H1 is", H1
    # print "H2 is", H2

    return max(H1, H2)


def main():
    # first case
    A = np.array([1, 2, 3])[:, np.newaxis]
    B = np.array([4, 5, 6])[:, np.newaxis]
    C = np.array([4, 5, 20])[:, np.newaxis]

    # print A
    # print B

    # print hausdorff(A, B)
    print hausdorff(A, C)

if __name__ == '__main__':
    main()
