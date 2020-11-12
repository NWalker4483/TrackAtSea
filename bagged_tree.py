from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from utils.common import load
import numpy as np
import geopy.distance

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

lat_regressor = DecisionTreeRegressor(random_state=0)
lon_regressor = DecisionTreeRegressor(random_state=0)
lat_regressor.fit(distorted_points, gps_points[:,0])
lon_regressor.fit(distorted_points, gps_points[:,1])

# cross_val_score(lat_regressor, distorted_points, gps_points[:,0], cv=10) # doctest: +SKIP
# cross_val_score(lon_regressor, distorted_points, gps_points[:,1], cv=10) # doctest: +SKIP

gps_projections = list(zip(lat_regressor.predict(distorted_points_test), lon_regressor.predict(distorted_points_test)))
# Compute Projection Distance Error
ERR = []
for i in range(len(gps_projections)):
    ERR.append(geopy.distance.geodesic(gps_projections[i], gps_points_test[i]).m)
print(f"Average Error (m) {np.mean(ERR)}")
print(f"Average Error (m) {np.median(ERR)}")
print(f"Error Standard Deviation (+/- m) {np.std(ERR)}")