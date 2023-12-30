import numpy as np
import math

def buildStaticWorld(columnCorners, worldH, worldW):
    # build occupancy grid map
    world = np.zeros((worldH, worldW))
    # set walls to be 1
    world[0,:] = 1
    world[-1,:] = 1
    world[:,0] = 1
    world[:,-1] = 1
    # set the column area to be 1
    world[columnCorners[0][1]:columnCorners[1][1]+1, columnCorners[0][0]:columnCorners[1][0]+1] = 1
    # print("world",world)
    return world

def collisionCheck(state, staticWorld):
    gridX = np.round(state[1])
    gridY = np.round(state[0])
    return staticWorld[gridX, gridY] == 1

def collisionFreeSearch(currXY, theta, maxR, staticWorld):

    # return if state is not collision free
    if collisionCheck(currXY, staticWorld):
        return 0,0,0,0
    # return a rectangle-shape collision free area
    xMin, xMax, yMin, yMax = -maxR, maxR, -maxR, maxR
    searchDis = 1
    searchArea = True
    xbound, ybound = staticWorld.shape

    while(searchArea):
        # only search in xMIN direction if no collision has happen yet
        if (xMin == -maxR):
            searchX = -searchDis
            for iSearchY in range(max(-searchDis, yMin)+1, min(searchDis, yMax)+1):
                searchState = np.copy(currXY)
                searchState[0] = searchState[0] + searchX
                searchState[1] = searchState[1] + iSearchY
                # if searchState[0]==0 or collisionCheck(searchState, staticWorld):
                if searchState[0]==0 or getRotateOccupancy(searchState[0], searchState[1], searchX, iSearchY, theta, staticWorld):
                    print("xMin collision at ", searchState)
                    xMin = searchX
                    break

        # only search in xMax direction if no collision has happen yet
        if (xMax == maxR):
            searchX = searchDis
            for iSearchY in range(max(-searchDis, yMin)+1, min(searchDis, yMax)+1):
                searchState = np.copy(currXY)
                searchState[0] = searchState[0] + searchX
                searchState[1] = searchState[1] + iSearchY
                # if searchState[0]==xbound-1 or collisionCheck(searchState, staticWorld):
                if searchState[0]==xbound-1 or getRotateOccupancy(searchState[0], searchState[1], searchX, iSearchY, theta, staticWorld):
                    print("xMax collision at ", searchState)
                    xMax = searchX
                    break

        # only search in yMIN direction if no collision has happen yet
        if (yMin == -maxR):
            searchY = -searchDis
            for iSearchX in range(max(-searchDis, xMin)+1, min(searchDis, xMax)+1):
                searchState = np.copy(currXY)
                searchState[0] = searchState[0] + iSearchX
                searchState[1] = searchState[1] + searchY
                # if searchState[1]==0 or collisionCheck(searchState, staticWorld):
                if searchState[1]==0 or getRotateOccupancy(searchState[0], searchState[1], searchX, iSearchY, theta, staticWorld):
                    print("yMin collision at ", searchState)
                    yMin = searchY
                    break

        # only search in yMax direction if no collision has happen yet
        if (yMax == maxR):
            searchY = searchDis
            for iSearchX in range(max(-searchDis, xMin)+1, min(searchDis, xMax)+1):
                searchState = np.copy(currXY)
                searchState[0] = searchState[0] + iSearchX
                searchState[1] = searchState[1] + searchY
                # if searchState[1]==ybound-1 or collisionCheck(searchState, staticWorld):
                if searchState[1]==ybound-1 or getRotateOccupancy(searchState[0], searchState[1], searchX, iSearchY, theta, staticWorld):
                    print("yMax collision at ", searchState)
                    yMax = searchY
                    break

        # Increase search distance
        searchDis +=1
        # Determine whether the search is finished
        searchArea = (searchDis < maxR) and ( xMin == -maxR or xMax == maxR or yMin == -maxR or yMax == maxR )
    return xMin+0.5, xMax-0.5, yMin+0.5, yMax-0.5

def getRotateOccupancy(xCurr, yCurr, searchX, searchY, psi, staticWorld):
    # psi: the angle to rotate world frame to body frame
    x_searchRotate = int(math.cos(psi)*searchX - math.sin(psi)*searchY)
    y_searchRotate = int(math.sin(psi)*searchX + math.cos(psi)*searchY)
    xSearch = xCurr + x_searchRotate
    ySearch = yCurr + y_searchRotate

    # check boundary
    if (xSearch >= staticWorld.shape[1] or xSearch <= 0 or ySearch >= staticWorld.shape[0] or ySearch <= 0):
        return True
    return collisionCheck([xSearch,ySearch], staticWorld)


def goal_test(curState, target):
    curPos = curState[:2]
    if np.linalg.norm(curPos - target) < 0.2:
        return True
    return False

def near_goal_test(curState, target):
    curPos = curState[:2]
    if np.linalg.norm(curPos - target) < 2:
        return True
    return False