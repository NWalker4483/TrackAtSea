from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from utils.common import load
import numpy as np
import geopy.distance

fit_videos = [7,13,16,21]
test_videos = [7,13,16,21] #[14,12,10,21]

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

lat_regressor = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                        n_estimators=10, random_state=0).fit(distorted_points, gps_points[:,0])
lon_regressor = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                        n_estimators=10, random_state=0).fit(distorted_points, gps_points[:,1])

gps_projections = list(zip(lat_regressor.predict(distorted_points_test), lon_regressor.predict(distorted_points_test)))
# Compute Projection Distance Error
ERR = []
for i in range(len(gps_projections)):
    ERR.append(geopy.distance.geodesic(gps_projections[i], gps_points_test[i]).m)
print(f"Average Error (m) {np.mean(ERR)}")
print(f"Median Error (m) {np.median(ERR)}")
print(f"Error Standard Deviation (+/- m) {np.std(ERR)}")