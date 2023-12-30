import numpy as np

# Divide reference path into small segments
def reference_segment(reference_path, refAmount):
    refSegs = np.zeros((refAmount,2))
    NrefTotal = reference_path.shape[0]
    amountIn1Sub = NrefTotal//refAmount
    for iRef in range(refSegs.shape[0]):
        refSegs[iRef] = reference_path[amountIn1Sub*iRef]
    return refSegs

# Find current reference path
def referenceGenerator(refSegments, target, reference_delV, currState, N, dt):
    currLoc = currState[:2]
    nearestRefIdx = 0
    nearestDis = np.inf
    for i in range(len(refSegments)):
        disBwt = np.linalg.norm(refSegments[i] - currLoc)
        if disBwt < nearestDis:
            nearestDis = disBwt
            nearestRefIdx = i
    startRef = refSegments[nearestRefIdx]
    if np.dot((currLoc - target), (startRef - target)) < 0:
        startRef = currLoc
    len2Target = np.linalg.norm(target - startRef)
    reference_delV = (target - startRef)/(dt*N) if len2Target < np.linalg.norm(4*reference_delV*dt*N) else currState[3]
    endRef = startRef + reference_delV*dt*N
    intermediate_points = np.linspace(startRef, endRef, N+1)
    return intermediate_points
