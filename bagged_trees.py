from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from utils.common import load_many
import numpy as np
import geopy.distance

fit_videos = [7,13,16,21]
test_videos = [14,12,10,21]

distorted_points, gps_points = load_many(fit_videos)

distorted_points_test, gps_points_test = load_many(test_videos)

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
print(f"Root Mean Square Error {(sum([err ** 2 for err in ERR])/len(ERR))**.5}")