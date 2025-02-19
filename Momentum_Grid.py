"""
This file controls the discretization of the momentum.
"""
import numpy as np
import numpy.polynomial.polynomial as Poly


def getIntegrationWeights(xCoords):
    """
    Computes the integration weights for a quadrature of maximal order for the given sampling points
    :param xCoords: The sampling points
    :return: A numpy array containing the integration weights
    """
    weigths = np.zeros(len(xCoords))
    divisors = np.prod(xCoords[:, np.newaxis] - xCoords + np.eye(len(xCoords)), axis=1)
    polys = np.array([Poly.polyfromroots([x]) for x in xCoords])
    for i in range(len(xCoords)):
        pol = Poly.Polynomial([1])
        for j in range(len(xCoords)):
            if i == j:
                continue
            pol *= polys[j]
        integral = pol.integ()
        weigths[i] = (integral(xCoords[-1]) - integral(xCoords[0])) / divisors[i]
    return weigths


def summedWeights(gridPoints, window=3):
    """
    Computes the weights for a summed quadrature function.
    Note that len(gridPoints) mod (window-1) has to be 1.
    :param gridPoints: The sampling points
    :param window: The number of points for each quadrature
    :return: An numpy array containing the corresponding weights
    """
    # grid weights for simpson rule (windowlen=3)
    # or trapezoidal rule (windowlen=2)
    assert len(gridPoints) % (window - 1) == 1

    pts = len(gridPoints)
    weigths = np.zeros(pts)

    for x in range(0, pts - window + 1, window - 1):
        weigths[x:x + window] += getIntegrationWeights(gridPoints[x: x + window])
    return weigths


def convertRegions(edges, binCount):
    """
    Compute the sampling points and edges for a segmented linear grid.
    :param edges: The starting and end points of all the linear sections
    :param binCount: A list of the number of bins in each section. len(binCount)==len(edges)-1 has to be fulfilled
    :return: A tuple containing the sampling points and weights
    """
    totalBins = np.sum(binCount)
    grid = np.zeros(totalBins)

    grid[0:binCount[0]] = np.linspace(edges[0], edges[1], binCount[0])
    runningTotal = binCount[0]

    for i in range(2, len(edges)):
        grid[runningTotal - 1:runningTotal + binCount[i - 1]] = np.linspace(edges[i - 1], edges[i], binCount[i - 1] + 1)
        runningTotal += binCount[i - 1]
    return grid, summedWeights(grid)


# If this is true the distributions will print a warning, when particles would be injected beyond the grid
debug_grid = False


# Set up the default linear grid, with y_max y_min and n_p as given in the arguments
def setupGrid(y_max_p, n_p, y_min_p=0.01):
    """
    Set up the momentum grid. This method must be called before any variable related to the grid is usable.
    The global variables set by this method are n, y_max, y_min, gridVals and gridWeights for the neutrino grid.
    Also some values for the electromagnetic grid are set: n_QED, yQED_max, yQED_min, dyQED
    :param y_max_p: Maximum comoving momentum in MeV
    :param n_p: Number of grid bins
    :param y_min_p: Minimum comoving momentum in MeV
    """
    global n, y_max, y_min, gridVals, gridWeights, n_QED, yQED_max, yQED_min, dyQED
    n = n_p
    y_max = y_max_p
    y_min = y_min_p

    gridVals = np.linspace(y_min, y_max, n)

    gridVals[0] = y_min
    gridVals[-1] = y_max  # ensure that this holds exactly
    gridWeights = summedWeights(gridVals, 3)

    # QED grid parameters, they do not depend on the LLP model chosen.
    n_QED = 81

    yQED_max = 20
    yQED_min = 0.01
    dyQED = (yQED_max - yQED_min) / (n_QED - 1)
