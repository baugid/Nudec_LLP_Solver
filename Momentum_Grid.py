import numpy as np
import numpy.polynomial.polynomial as Poly
from scipy.special import roots_legendre


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


def summedWeights(gridPoints, windowlen=3):
    """
    Computes the weights for a summed quadrature function
    :param gridPoints: The sampling points
    :param windowlen: The number of points for each quadrature
    :return: An numpy array containing the corresponding weights
    """
    # grid weights for simpson rule (windowlen=3)
    # or trapezoidal rule (windowlen=2)
    # note that n mod (windowlen-1) has to be 1
    assert len(gridPoints) % (windowlen - 1) == 1

    pts = len(gridPoints)
    weigths = np.zeros(pts)

    for x in range(0, pts - windowlen + 1, windowlen - 1):
        weigths[x:x + windowlen] += getIntegrationWeights(gridPoints[x: x + windowlen])
    return weigths


def convertRegions(edges, binCount):
    """
    Compute the sampling points and edges for a segmented linear grid.
    :param edges: The starting and end points of all the linear sections
    :param binCount: A list of the number of bins in each sections. len(binCount)==len(edges)-1 has to be fulfilled
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


#Setup the default linear grid, with y_max y_min and n_p as given in the arguments
def setupGrid(y_max_p, n_p, y_min_p=0.01):
    global n, y_max, y_min, gridVals, gridWeights, n_QED, yQED_max, yQED_min, dyQED
    n = n_p
    y_max = y_max_p
    y_min = y_min_p

    gridVals = np.linspace(y_min, y_max, n)
    # gridVals = np.logspace(np.log10(y_min), np.log10(y_max), n)

    gridVals[0] = y_min
    gridVals[-1] = y_max  # ensure that this holds exactly
    gridWeights = summedWeights(gridVals, 3)
    # QED grid parameters
    n_QED = 81

    yQED_max = 20
    yQED_min = 0.01
    dyQED = (yQED_max - yQED_min) / (n_QED - 1)

# Old test code do not blindly uncomment

# n = 81  # 201  # number of bins, n must be an odd number because we use the Simpson method
# The actual momentum grid is n-1.

# y_max = 20  # 110  # 2070
# y_min = 0.01
# dy = (y_max - y_min) / (n - 1)

# gridVals = np.linspace(y_min, y_max, n)
# gridVals = np.logspace(np.log10(y_min), np.log10(y_max), n)

# gridVals[0] = y_min
# gridVals[-1] = y_max  # ensure that this holds exactly
# gridWeights = summedWeights(gridVals, 3)

# gridVals, gridWeights = convertRegions([y_min, 10.,y_max], [41,100])
# n=len(gridVals)


# gauss legendre
# gridVals, gridWeights = roots_legendre(n)
# gridVals = (gridVals + 1) * (y_max / 2)
# gridWeights = gridWeights * y_max / 2
