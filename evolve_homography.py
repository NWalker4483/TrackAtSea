import numpy as np
import cv2
import geopy.distance
from geneticalgorithm import geneticalgorithm as ga
import random 
from utils.common import *
fit_videos = [7,16,13]
test_videos = fit_videos# + [14,12,10,21]

box_points, gps_points = load(fit_videos[0])
for i in range(1,len(fit_videos)):
    box_points_temp, gps_points_temp = load(fit_videos[i])
    box_points = np.concatenate((box_points, box_points_temp), axis=0)
    gps_points = np.concatenate((gps_points, gps_points_temp), axis=0)

box_points_test, gps_points_test = load(test_videos[0])
for i in range(1,len(fit_videos)):
    box_points_temp, gps_points_temp = load(fit_videos[i])
    box_points_test = np.concatenate((box_points_test, box_points_temp), axis=0)
    gps_points_test = np.concatenate((gps_points_test, gps_points_temp), axis=0)
distorted_points = []
distorted_points_test = []
for i in range(len(box_points)):
    X1, Y1, X2, Y2 = box_points[i]
    distorted_points.append([X1+((X2 - X1)//2),Y2])
for i in range(len(box_points_test)):
    X1, Y1, X2, Y2 = box_points_test[i]
    distorted_points_test.append([X1+((X2 - X1)//2),Y2])
distorted_points, distorted_points_test = np.array(distorted_points), np.array(distorted_points_test, dtype=np.float64)
center = 1280 // 2, 720 // 2
offset = 25
# f_x, f_y, k1, k2, p1, p2 
k = 1
varbound = np.array([[0, 1200],[0, 1200],[center[0]-offset, center[0]+offset],[center[1]-offset, center[1]+offset],[-k, k],[-k, k],[-k, k],[-k, k]])
best_h = None 

min_err = np.inf
best_camera_params = dict()
def distance(p1, p2):
    return (((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)) ** .5
def reprojection_error(X):
    global min_err, best_h

    # Camera Matrix and Distortion Coefficients
    # px, py, f, sx, sy, pan, tilt, swing, tx, ty, tz
    f_x, f_y, c_x, c_y, k1, k2, p1, p2 = X
    mtx = np.array([[f_x, 0, c_x],
                    [0, f_y, c_y],
                    [0,   0,   1]], dtype=np.float64)
    dist = np.array([[k1, k2, p1, p2]], dtype=np.float64)
    # Undistort detection points # https://stackoverflow.com/questions/22027419/bad-results-when-undistorting-points-using-opencv-in-python
    undistorted_points = cv2.undistortPoints(
        distorted_points.reshape(-1, 1, 2), mtx, dist, P=mtx)
    # Compute Homography
    # choices = np.random.randint(0,len(distorted_points), size=4)
    h, status = cv2.findHomography(undistorted_points, gps_points, cv2.RANSAC)
    if type(h) != type(None):
        # Compute New Projections # https://stackoverflow.com/questions/55055655/how-to-use-cv2-perspectivetransform-to-apply-homography-on-a-set-of-points-in
        gps_projections = cv2.perspectiveTransform(undistorted_points, h)
        gps_projections = gps_projections.reshape(-1, 2)
        # Compute Average Projection Distance Error

        ERR = []
        for i in range(len(gps_projections)):
            ERR.append(distance(gps_projections[i], gps_points[i]))
        RMSE = (sum([err ** 2 for err in ERR])/len(ERR))**.5
        if RMSE < min_err:
            min_err = RMSE 
            best_camera_params["Homography"] = h
            best_camera_params["Distortion Coefficients"] = dist
            best_camera_params["Intrinsic Matrix"] = mtx
        return RMSE
    else:
        return 10e10
# We definitly dont need this many
algorithm_param = {'max_num_iteration': 100, 'population_size':100,\
                   'mutation_probability':0.2,'elit_ratio': 0.3,\
                   'crossover_probability': 0.5,'parents_portion': 0.3,\
                   'crossover_type':'uniform','max_iteration_without_improv': 200}

model=ga(function=reprojection_error,\
            dimension=8,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)

f_x, f_y, c_x, c_y, k1, k2, p1, p2, median = [], [], [], [], [], [], [], [], []
from unittest.mock import patch

for _ in range(5):
    with patch('matplotlib.pyplot.show') as _: # Prevent Plot from blocking the for loop
        model.run()
        best_camera_params["Error"] = dict()
        ###################################################
        undistorted_points = cv2.undistortPoints(
                distorted_points_test.reshape(-1, 1, 2), best_camera_params["Intrinsic Matrix"], best_camera_params["Distortion Coefficients"], P=best_camera_params["Intrinsic Matrix"])
        gps_projections = cv2.perspectiveTransform(undistorted_points, best_camera_params["Homography"])
        gps_projections = gps_projections.reshape(-1, 2)
        # Compute Projection Distance Error
        ERR = []
        for i in range(len(gps_projections)):
            ERR.append(geopy.distance.geodesic(gps_projections[i], gps_points_test[i]).m)
        best_camera_params["Error"]["Mean"] = np.mean(ERR)
        best_camera_params["Error"]["Median"] = np.median(ERR)
        best_camera_params["Error"]["Std"] = np.std(ERR)
        best_camera_params["Error"]["RMSE"] = (sum([err ** 2 for err in ERR])/len(ERR))**.5

        f_x.append(best_camera_params["Intrinsic Matrix"][0][0])
        f_y.append(best_camera_params["Intrinsic Matrix"][1][1])
        c_x.append(best_camera_params["Intrinsic Matrix"][0][2])
        c_y.append(best_camera_params["Intrinsic Matrix"][1][2])
        median.append(best_camera_params["Error"]["Median"])
print("")
print(f"Average Error (m): {best_camera_params['Error']['Mean']}")
print(f"Median Error (m): {best_camera_params['Error']['Median']}")
print(f"Error Standard Deviation (+/- m): {best_camera_params['Error']['Std']}")
print(f"Root Mean Square Error: {best_camera_params['Error']['RMSE']}")


from utils.common import plot_gps
plot_gps([gps_projections, gps_points_test])

# from scipy.stats.kde import gaussian_kde
# from numpy import linspace
# import matplotlib.pyplot as plt
# plt.clf() # Clear Stuff from the runs
# def normalize_array(A):
#     A = np.array(A, copy=True)
#     A -= min(A)
#     A /= max(A)
#     return A

# # these are the values over wich your kernel will be evaluated
# dist_space = linspace( 0, 1, 10000 ) # plot the results
# for data in [median]:
#     # this create the kernel, given an array it will estimate the probability over that values
#     kde = gaussian_kde(normalize_array(data))
#     plt.plot(dist_space, kde(dist_space) )
# plt.show()