import numpy as np
import cv2
import geopy.distance
from geneticalgorithm import geneticalgorithm as ga
import random 

def load(video_num):
    det_frames = dict()
    with open(f"generated_data/tracks/manual.{video_num}.csv","r") as f:
        content = f.readlines()
        content = content[1:] # Skip Header
    for entry in content:
        Frame_No, ID, X1, Y1, X2, Y2 = [int(i) for i in entry.split(",")]
        det_frames[int(Frame_No)] = [X1+((X2 - X1)//2),Y2]

    gps_points = []
    distorted_points = []
    with open(f"generated_data/frame2gps/{video_num}.csv","r") as f:
        content = f.readlines()
        content = content[1:] # Skip Header
    for entry in content:
        Frame_No, Frame_Time, GPS_Time, Latitude, Longitude = entry.split(",")
        if int(Frame_No) in det_frames:
            gps_points.append([float(Latitude), float(Longitude)])
            distorted_points.append(det_frames[int(Frame_No)])
    return np.array(distorted_points, dtype=np.float64), np.array(gps_points, dtype=np.float64)

fit_videos = [7,16,13,14]
test_videos = [14,12,10,21]

distorted_points, gps_points = load(fit_videos[0])
for i in range(1,len(fit_videos)):
    distorted_points_temp, gps_points_temp = load(fit_videos[i])
    distorted_points = np.concatenate((distorted_points, distorted_points_temp), axis=0)
    gps_points = np.concatenate((gps_points, gps_points_temp), axis=0)

distorted_points_test, gps_points_test = load(test_videos[0])
for i in range(1,len(fit_videos)):
    distorted_points_temp, gps_points_temp = load(fit_videos[i])
    distorted_points_test = np.concatenate((distorted_points_test, distorted_points_temp), axis=0)
    gps_points_test = np.concatenate((gps_points_test, gps_points_temp), axis=0)

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
        AVG = sum(ERR)/len(gps_projections)
        if AVG < min_err:
            min_err = AVG 
            best_camera_params["Homography"] = h
            best_camera_params["Distortion Coefficients"] = dist
            best_camera_params["Intrinsic Matrix"] = mtx
        return AVG
    else:
        return 10e10
# We definitly dont need this many
algorithm_param = {'max_num_iteration': 100,\
                   'population_size':100,\
                   'mutation_probability':0.2,\
                   'elit_ratio': 0.3,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv': 200}

model=ga(function=reprojection_error,\
            dimension=8,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)

from unittest.mock import patch

with patch('matplotlib.pyplot.show') as p: # Prevent Plot from blocking
    model.run()
print(best_camera_params)
###################################################
undistorted_points = cv2.undistortPoints(
        distorted_points_test.reshape(-1, 1, 2), best_camera_params["Intrinsic Matrix"], best_camera_params["Distortion Coefficients"], P=best_camera_params["Intrinsic Matrix"])
gps_projections = cv2.perspectiveTransform(undistorted_points, best_camera_params["Homography"])
gps_projections = gps_projections.reshape(-1, 2)
# Compute Projection Distance Error
ERR = []
for i in range(len(gps_projections)):
    ERR.append(geopy.distance.geodesic(gps_projections[i], gps_points_test[i]).m)
print(f"""\
Average Error (m): {np.mean(ERR)}
Median Error (m): {np.median(ERR)}
Error Standard Dev (+/- m): {np.std(ERR)}
""")
from utils.common import plot_gps
plot_gps([gps_projections, gps_points_test])

# from scipy.stats.kde import gaussian_kde
# from numpy import linspace
# import matplotlib.pyplot as plt
# plt.clf() # Clear Stuff from the runs


# # these are the values over wich your kernel will be evaluated
# dist_space = linspace( 0, 800, 10000 )
# # plot the results
# try:
#     for i in range(8):
#         data = results[:,i]
#         # this create the kernel, given an array it will estimate the probability over that values
#         kde = gaussian_kde(data)
#         plt.plot( dist_space, kde(dist_space) )
#     plt.show()
# except:
#     pass