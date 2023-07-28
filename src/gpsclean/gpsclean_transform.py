import numpy as np
from pyproj import Transformer
import sys

#setting max int used for masking
max_int = sys.maxsize

ecef = {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}
lla = {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'}

transformer = Transformer.from_crs(lla, ecef)

def create_deltas(points, times):
    
    deltas = []
    features = []
    datetimes = []
    previousElement = -1
    
    for i in range(points.shape[0]):
        point = points[i]
        lon, lat, alt = point[0], point[1], point[2]
        
        if alt is not None:
            
            x, y, z = transformer.transform(lon, lat, alt, radians=False)
            
            #append
            features.append([x,y,z])

            #TIMESTAMPS
            #append parsed datetime object
            datetimes.append(times[i])
            
            if i > 0 and previousElement >= 0:
                time_difference = (datetimes[previousElement + 1] - datetimes[previousElement]).total_seconds()
                lon_diff = x - features[previousElement][0]
                lat_diff = y - features[previousElement][1]
                alt_diff = z - features[previousElement][2]
                
                deltas.append(np.array([lon_diff, lat_diff, alt_diff, time_difference]))
                
            previousElement += 1
            
            
            
    return np.array(deltas)
            
            
