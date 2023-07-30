import numpy as np
from pyproj import Transformer
import sys

#setting max int used for masking
max_int = sys.maxsize

ecef = {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}
lla = {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'}

lla_to_ecef_transformer = Transformer.from_crs(lla, ecef)
ecef_to_lla_transform = Transformer.from_crs(ecef, lla)

def create_deltas(points, times):
    
    deltas = []
    features = []
    datetimes = []
    previousElement = -1
    
    for i in range(points.shape[0]):
        point = points[i]
        lon, lat, alt = point[0], point[1], point[2]
        
        if alt is not None:
            
            x, y, z = lla_to_ecef_transformer.transform(lon, lat, alt, radians=False)
            
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
            
            
def lla_to_ecef(lon, lat, alt):
    x, y, z = lla_to_ecef_transformer.transform(lon, lat, alt, radians=False)

    return x, y, z

def ecef_to_lla(x, y, z):
    lon, lat, alt = ecef_to_lla_transform.transform(x, y, z, radians=False)

    return lon, lat, alt

def convert_lla_points_to_ecef(points):
    
    #convert everything to ecef
    ecef_points = []
    
    for i in range(points.shape[0]):
        #get point
        point = points[i]
        #convert to ECEF
        lon, lat, alt = point[0], point[1], point[2]
        x, y, z = lla_to_ecef(lon, lat, alt)
        #append to data
        ecef_points.append(np.array([x,y,z]))
    
    return np.array(ecef_points)

def convert_ecef_points_to_lla(points):
    lla_points = []
    
    for i in range(points.shape[0]): #for each point
        #get data
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        #convert to lat long and append
        lon, lat, alt = ecef_to_lla(x, y, z)
        lla_points.append(np.array([lon, lat, alt]))
   
    return np.array(lla_points)