import numpy as np
import cv2
import geopy.distance
from geneticalgorithm import geneticalgorithm as ga
import random 

gps_points = np.array([[32.703180333, -117.233720333], [32.7031835, -117.234026], [32.702750833, -117.234147333],
                       [32.702956833, -117.2339705], [32.703253167, -117.234196667], [32.7033075, -117.234030333], 
                       [32.702981167, -117.233853167], [32.702673333, -117.233515], [32.7028145, -117.2332445]], dtype=np.float32)
distorted_points = np.array([[586, 461], [734, 474], [1209, 465], 
                            [738, 471], [160, 468], [536, 467], 
                            [1057, 455], [1143, 442], [948, 442]], dtype=np.float32)

def distance(p1, p2):
    return (((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)) ** .5
k = 5
center = 1280 // 2, 720 // 2
# f_x, f_y, k1, k2, k3, p1, p2 
varbound = np.array([[0, 1500],[0, 1500],[center[0]-200, center[0]+200],[center[1]-200, center[1]+200],[-k, k],[-k, k],[-k, k],[-k, k],[-k, k]])
best_h = None 
best_err = np.inf
def reprojection_error(X):
    global best_err, best_h
    # Camera Matrix and Distortion Coefficients
    # px, py, f, sx, sy, pan, tilt, swing, tx, ty, tz
    f_x, f_y, c_x, c_y, k1, k2, k3, p1, p2 = X
    mtx = np.array([[f_x, 0, c_x],
                    [0, f_y, c_y],
                    [0,   0,   1]])
    dist = np.array([[k1, k2, p1, p2, k3]])
    # Undistort detection points # https://stackoverflow.com/questions/22027419/bad-results-when-undistorting-points-using-opencv-in-python
    undistorted_points = cv2.undistortPoints(
        distorted_points.reshape(-1, 1, 2), mtx, dist, P=mtx)
    # Compute Homography
    sources = np.random.randint(0,len(distorted_points),4)
    h, status = cv2.findHomography(undistorted_points[sources], gps_points[sources])
    if type(h) != type(None):
        # Compute New Projections # https://stackoverflow.com/questions/55055655/how-to-use-cv2-perspectivetransform-to-apply-homography-on-a-set-of-points-in
        gps_projections = cv2.perspectiveTransform(undistorted_points, h)
        gps_projections = gps_projections.reshape(-1, 2)
        # Compute Average Projection Distance Error

        ERR = []
        for i in range(len(gps_projections)):
            ERR.append(distance(gps_projections[i], gps_points[i]))
            #ERR.append(geopy.distance.geodesic(gps_projections[i], gps_points[i]).m)
        AVG = sum(ERR)/len(gps_projections)
        if AVG < best_err:
            best_err = AVG 
            best_h = h
        return AVG
    else:
        return 99999999999999


algorithm_param = {'max_num_iteration': 1500,\
                   'population_size':100,\
                   'mutation_probability':0.2,\
                   'elit_ratio': 0.05,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'one_point',\
                   'max_iteration_without_improv':None}

model=ga(function=reprojection_error,\
            dimension=9,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)
from unittest.mock import patch
results = []
outputs = []
reports = []
runs = 2
for _ in range(runs):
    with patch('matplotlib.pyplot.show') as p: # Prevent Plot from blocking
        model.run()
        results.append(model.output_dict["variable"])
        reports.append(model.report)
        outputs.append(model.output_dict)
results = np.array(results)

import matplotlib.pyplot as plt
plt.clf() # Clear Stuff from the runs

best = min(outputs, key = lambda x: x["function"])
# Camera Matrix and Distortion Coefficients
f_x, f_y, c_x, c_y, k1, k2, k3, p1, p2 = best["variable"]
mtx = np.array([[f_x, 0, c_x],
                [0, f_y, c_y],
                [0,   0,   1]])
dist = np.array([[k1, k2, p1, p2, k3]])
print(mtx, dist, best_h)
undistorted_points = cv2.undistortPoints(
        distorted_points.reshape(-1, 1, 2), mtx, dist, P=mtx)
gps_projections = cv2.perspectiveTransform(undistorted_points, best_h)
gps_projections = gps_projections.reshape(-1, 2)
# Compute Projection Distance Error
ERR = []
for i in range(len(gps_projections)):
    ERR.append(geopy.distance.geodesic(gps_projections[i], gps_points[i]).m)
    print(gps_projections[i], gps_points[i])
print(f"""\
Average Error (m): {sum(ERR)/len(distorted_points)}
Standard Dev (+/- m): {np.std(ERR)}
""")

from scipy.stats.kde import gaussian_kde
from numpy import linspace

# these are the values over wich your kernel will be evaluated
dist_space = linspace( 0, 15, 100 )
# plot the results
try:
    for i in range(9):
        data = results[:,i]
        # this create the kernel, given an array it will estimate the probability over that values
        kde = gaussian_kde(data)
        plt.plot( dist_space, kde(dist_space) )
    #plt.show()
except:
    pass