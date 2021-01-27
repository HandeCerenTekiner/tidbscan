import numpy as np
import operator

# returns the point's eps-neighborhood (points preceding p in D)
from pip._vendor.msgpack.fallback import xrange


def TI_Backward_Neighborhood(D, point, Eps):
    # note that dataset D is ordered non-decreasingly
    seeds = []
    backwardThreshold = point.dist - Eps

    # index of the point in the dataset D
    index = D.index(point)
    # list of points of D immediately preceding the point (p)
    pointsAfterP = D[:index]
    pointsAfterP.reverse()  # so it starts from the closest to the point

    # going through the list
    for q in pointsAfterP:
        if q.dist < backwardThreshold:  # we directly know it's not part of the neighborhood
            break
        if Distance(q.Coords, point.Coords) <= Eps:  # it's part of the neighborhood
            seeds.append(q)

    return seeds


# returns the point's eps-neighborhood (points following p in D)
def TI_Forward_Neighborhood(D, point, Eps):
    # note that dataset D is ordered non-decreasingly
    seeds = []
    forwardThreshold = point.dist + Eps

    # index of the point in the dataset D
    index = D.index(point)
    # list of points of D immediately after the point (p)
    pointsBeforeP = D[index + 1:]

    # going through the list
    for q in pointsBeforeP:
        if q.dist > forwardThreshold:  # we directly know it's not part of the neighborhood
            break
        if Distance(q.Coords, point.Coords) <= Eps:  # it's part of the neighborhood
            seeds.append(q)

    return seeds


# returns the point's eps-neighborhood (without p itself)
def TI_Neighborhood(D, point, Eps):
    first = TI_Backward_Neighborhood(D, point, Eps)
    last = TI_Forward_Neighborhood(D, point, Eps)
    return first + last


def TI_ExpandCluster(D, checkedD, point, clusterId, Eps, MinPts):
    # distances are calculated from the reference point r

    # gets the point's eps-neighborhood without itself
    seeds = TI_Neighborhood(D, point, Eps)
    # adds that number of points to the point's neighbors number
    point.NeighborsNo += len(seeds)

    # the point p can be noise or a border point
    if point.NeighborsNo < MinPts:
        # initially declared as noise
        point.ClusterId = "NOISE"
        # goes through the list of seeds and adds p to q's border points
        for q in seeds:
            q.Border.append(point)
            q.NeighborsNo += 1

        # the point's set of border points is declared as empty
        point.Border = []
        # the point is moved from D to D' (checkedD)
        D.remove(point)
        checkedD.append(point)

        return False  # the cluster hasn't been expanded

    else:
        # assigns current cluster's id
        point.ClusterId = clusterId
        # goes through the list of seeds and assigns each one to the same cluster id
        for q in seeds:
            q.ClusterId = clusterId
            q.NeighborsNo += 1

        # goes through the list of border points of p
        for q in point.Border:
            # gets q from D' and assigns the cluster id
            checkedD[checkedD.index(q)].ClusterId = clusterId

        # the point's set of border points is declared as empty
        point.Border = []
        # the point is moved from D to D' (checkedD)
        D.remove(point)
        checkedD.append(point)

        # while there are seeds left in the list
        while seeds:
            # gets the first point from the seeds
            currentPoint = seeds[0]
            # calculates the eps-neighborhood of the seed and updates the number of neighbors
            currentSeeds = TI_Neighborhood(D, currentPoint, Eps)
            currentPoint.NeighborsNo += len(currentSeeds)

            # currentPoint is a border point
            if currentPoint.NeighborsNo < MinPts:
                # goes throught the seeds and updates each number of neighbors
                for q in currentSeeds:
                    q.NeighborsNo += 1

            # currentPoint is a core point
            else:
                # while there are seeds left
                while currentSeeds:
                    # first point of the list
                    q = currentSeeds[0]
                    # updates the number of neighbors
                    q.NeighborsNo += 1

                    # if the point hasn't been classified yet, its cluster id is assigned
                    if q.ClusterId == "UNCLASSIFIED":
                        q.ClusterId = clusterId
                        # point moved from currentSeeds to seeds
                        currentSeeds.remove(q)
                        seeds.append(q)
                    else:
                        # point gets deleted since it has already been classified
                        currentSeeds.remove(q)

                # goes through currentPoint's border points
                for q in currentPoint.Border:
                    # gets q from D' and assigns the cluster id
                    checkedD[checkedD.index(q)].ClusterId = clusterId

            # border emptied
            currentPoint.Border = []
            # point moved from D to D'
            D.remove(currentPoint)
            checkedD.append(currentPoint)
            # point deleted from seeds
            seeds.remove(currentPoint)

        return True  # the cluster has been expanded


# calculates the euclidean distance between a point and the reference point (2D)
def Distance(point, referencePoint):
    point = np.array(point[0:2])
    referencePoint = np.array(referencePoint[0:2])
    return np.sqrt(np.sum(np.power(point - referencePoint, 2)))


class pointClass:
    def __init__(self, point, referencePoint, metadata=None):
        try:
            # metadata
            self.metadata = metadata
            # coordinates
            self.Coords = point[0:2]
        except:
            pass

        self.ClusterId = "UNCLASSIFIED"
        self.dist = Distance(point[0:2], referencePoint[0:2])
        self.NeighborsNo = 1
        self.Border = []


# applies the TI-DBSCAN algorithm to the dataset D with a given value of Eps and MinPts
def TI_DBScan(D, Eps, MinPts, metadata=None):
    # the data structure is: D = [[coord1, coord2, ...], ...]
    # the first two values are the 2 dimension coordinates of each element

    try:
        referencePoint = D[0]  # we use the first point in D as the reference
    except IndexError:
        pass

    # the minimum number of points can't be 1
    MinPts = MinPts if MinPts > 1 else 2
    # D' initially empty
    checkedD = []

    # transformation and initialization of the points (distance, cluster, ...)
    try:
        D = [pointClass(D[index], referencePoint, metadata=metadata[index])
             for index in xrange(len(D))]
    except TypeError:
        D = [pointClass(D[index], referencePoint)
             for index in xrange(len(D))]

    # sorts all points in D non-decreasingly by the field dist
    D = sorted(D, key=operator.attrgetter('dist'))

    # initial label for the clusters
    i = 1
    ClusterId = "%s" % i

    # goes through the whole dataset D looking for and assigning clusters
    while D:
        point = D[0]  # first point from D
        if TI_ExpandCluster(D, checkedD, point, ClusterId, Eps, MinPts):  # cluster has been created
            i += 1
            ClusterId = "%s" % i  # moves on to the next cluster id

    # returns D', the clustered set of points from D
    return checkedD


# test data
'''testPoints = [[1.00, 1.00], [1.50, 1.00], [2.00, 1.50], [5.00, 5.00], [6.00, 5.50], [5.50, 6.00],
                      [10.00, 11.00], [10.50, 9.50], [10.00, 10.00], [8.00, 1.00], [1.00, 8.00]]'''

testPoints = [[1, 2], [3, 4], [2.5, 4], [1.5, 2.5], [3, 5], [2.8, 4.5], [2.5, 4.5], [1.2, 2.5], [1, 3],
              [1, 5], [1, 2.5], [5, 6], [4, 3]]

testPointsClustered = TI_DBScan(testPoints, 0.6, 4)

for point in testPointsClustered:
    print(point.ClusterId + ' - [' + ', '.join(map(str, point.Coords)) + ']')


