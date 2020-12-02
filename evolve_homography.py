import numpy as np
import cv2
import geopy.distance
from geneticalgorithm import geneticalgorithm as ga
from utils.common import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

fit_videos = [7,16]
test_videos = [7,12,10,16]

box_points, gps_points = load_many(fit_videos)#, preface="sort.orb.matched", train=True) # 
box_points_test, gps_points_test = load_many(test_videos)#, preface="orb", train=True)

distorted_points = []
distorted_points_test = []
for i in range(len(box_points)):
    X1, Y1, X2, Y2 = box_points[i]
    distorted_points.append([X1+((X2 - X1)//2),Y2])
for i in range(len(box_points_test)):
    X1, Y1, X2, Y2 = box_points_test[i]
    distorted_points_test.append([X1+((X2 - X1)//2),Y2])

distorted_points, distorted_points_test = np.array(distorted_points, dtype = np.float64), np.array(distorted_points_test, dtype=np.float64)
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
    f_x, f_y, c_x, c_y, k1, k2, p1, p2 = X
    mtx = np.array([[f_x, 0, c_x],
                    [0, f_y, c_y],
                    [0,   0,   1]], dtype=np.float64)
    dist = np.array([[k1, k2, p1, p2]], dtype=np.float64)
    # Undistort detection points # https://stackoverflow.com/questions/22027419/bad-results-when-undistorting-points-using-opencv-in-python
    undistorted_points = cv2.undistortPoints(
        distorted_points.reshape(-1, 1, 2), mtx, dist, P=mtx)
    # Compute Homography
    choices = np.random.randint(0,len(distorted_points), size=10)
    h, status = cv2.findHomography(undistorted_points[choices], gps_points[choices], cv2.RANSAC)
    
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
    
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            ax.scatter(gps_projections[:,0],gps_projections[:,1],c=ERR/max(ERR),cmap="Reds")
            ax.scatter(gps_points[:,0],gps_points[:,1], marker="x")#,c=colors,cmap="bwr")

            canvas.print_figure("output.png")
            frame = plt.imread("output.png")
            image_without_alpha = frame[:,:,:3]
            frame = np.uint8(255 * image_without_alpha)
            out.write(frame)
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
shape = (480, 640, 4)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 15,
                    (shape[1],  shape[0]))
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
out.release()
print("")
print(f"Average Error (m): {best_camera_params['Error']['Mean']}")
print(f"Median Error (m): {best_camera_params['Error']['Median']}")
print(f"Error Standard Deviation (+/- m): {best_camera_params['Error']['Std']}")
print(f"Root Mean Square Error: {best_camera_params['Error']['RMSE']}")

print(best_camera_params["Homography"])
import pickle
pickle.dump(best_camera_params, open( "generated_data/best_camera_params.pkl", "wb" ))
from utils.common import plot_gps
img = plot_gps([ gps_projections, gps_points_test], colors = [(255,0,0),(0,0,255)])


# from scipy.stats.kde import gaussian_kde
# from numpy import linspace
# import matplotlib.pyplot as plt
# plt.clf() # Clear Stuff from the runs
# def normalize_array(A):

# Joe Adams A & E Technologies Rapid Autonomy Integration Labs Automated Testing and Evaluation Software
# Matt Veneda Operations Research Modeling and Simulation 
# Mark been there 20 years Unnmaned Underwater Vehicles. Creating Standards for autonomous systems, UUV  
# David From Raytheon EMA Kinda works under Joe 
# 
# Look Up 
# ICD ICL 
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