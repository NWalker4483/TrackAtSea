import numpy as np
import cv2
import geopy.distance
from geneticalgorithm import geneticalgorithm as ga


c_x, c_y = 1280 // 2, 720 // 2  # Center Point of the Image
gps_points = np.array([[32.703180333, -117.233720333], [32.7031835, -117.234026],
                       [32.7027545, -117.234017667], [32.702750833, -117.234147333],
                       [32.702956833, -117.2339705], [32.703253167, -117.234196667],
                       [32.7033075, -117.234030333], [32.702981167, -117.233853167],
                       [32.702673333, -117.233515], [32.7028145, -117.2332445]], dtype=np.float32)
distorted_points = np.array([[586, 461], [734, 474], [734, 474],
                             [1209, 465], [738, 471], [160, 468],
                             [536, 467], [1057, 455], [1143, 442], [948, 442]], dtype=np.float32)


def reprojection_error(X):
    # Camera Matrix and Distortion Coefficients
    f_x, f_y, k1, k2, k3, p1, p2 = X
    mtx = np.array([[f_x, 0, c_x],
                    [0, f_y, c_y],
                    [0,   0,   1]])
    dist = np.array([[k1, k2, p1, p2, k3]])
    # Undistort detection points # https://stackoverflow.com/questions/22027419/bad-results-when-undistorting-points-using-opencv-in-python
    undistorted_points = cv2.undistortPoints(
        distorted_points.reshape(-1, 1, 2), mtx, dist, P=mtx)
    # Compute Homography
    h, status = cv2.findHomography(undistorted_points, gps_points)
    if type(h) != type(None):
        # Compute New Projections # https://stackoverflow.com/questions/55055655/how-to-use-cv2-perspectivetransform-to-apply-homography-on-a-set-of-points-in
        gps_projections = cv2.perspectiveTransform(undistorted_points, h)
        gps_projections = gps_projections.reshape(-1, 2)
        # Compute Mean Squared Projection Error
        MSE = 0
        for i in range(len(gps_projections)):
            MSE += geopy.distance.geodesic(gps_projections[i], gps_points[i]).m ** 2
        return MSE
    else:
        return np.inf


varbound = np.array([[0, 10]]*7)

algorithm_param = {'max_num_iteration': 3000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=reprojection_error,\
            dimension=7,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)
model.run()
