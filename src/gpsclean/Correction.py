#this file contains the functions used to correct a trace based on the generated predictions
import numpy as np
import pyproj
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import pandas as pd
from datetime import datetime
from tensorflow import clip_by_value, GradientTape, convert_to_tensor, zeros_like
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


#defining conversions between the two used coordinates systems
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

#value used for clipping the deltas correction of each epoch applied to outliers
EPS = 0.1

#function used to remove pauses from the original trace based on the predictions, 
#since in case of pauses no extra correction is needed besides removing them
def remove_pauses(all_coords, all_coordtimes, predictions, deltas):
      
    #prepare variables that will host the final trace
    points = []
    original_points = []
    reduced_deltas = []
    reduced_predictions = []
    
    coordTimes = []
    original_times = []
    
    cur_point = 0
    
    #print("Coords: ", all_coords.shape[0], ", Preds: ", predictions.shape)
    
    #if we have subtraces
    if len(np.array(all_coords[0]).shape) > 1:
        #loop over them and over their points
        for i in range(all_coords.shape[0]):
            for j in range(all_coords[i]):
                #get current point
                point = np.array(all_coords[i][j])
                
                #if it is valid, namely it has an altitude value
                if point[2] is not None:
                    #append to original trace
                    original_points.append(point)
                    original_times.append(all_coordtimes[i][j])
                    #then check the prediction and if it is not a pause let's keep it along with its data (time and prediction)
                    if cur_point == 0 or not predictions[cur_point] == 1:
                        points.append(point)
                        coordTimes.append(all_coordtimes[i][j])
                        reduced_predictions.append(predictions[cur_point])
                        
                        if deltas is not None:
                            reduced_deltas.append(deltas[cur_point])
                        
                    cur_point += 1
    else: #if it is a single trace do the same but with one loop only
        for j in range(all_coords.shape[0]):
            point = np.array(all_coords[j]) #get the point
            
            if point[2] is not None: #check it is valid
                    
                    original_points.append(point) #append to original trace
                    original_times.append(all_coordtimes[j])
                    
                    if cur_point == 0 or not predictions[cur_point] == 1: #check the prediction is not a pause and keep it
                        points.append(point)
                        coordTimes.append(all_coordtimes[j])
                        reduced_predictions.append(predictions[cur_point])
                        
                        if deltas is not None:
                            reduced_deltas.append(deltas[cur_point])
                    
                    #else:
                        #print("Deleting pause point at index: ", cur_point, " with prediction: ", predictions[cur_point])
                        
                    cur_point += 1
   
    #return the results as numpy arrays
    return np.array(points), np.array(coordTimes), np.array(reduced_deltas), np.array(reduced_predictions), np.array(original_points), np.array(original_times)
        
    
#function used to clip a delta tensor using the specified EPS value
def clip_eps(tensor, eps):
    # clip the values of the tensor to a given range and return it
    return clip_by_value(tensor, clip_value_min=-eps,clip_value_max=eps)


#function used to generate a delta correction for a segment provided as input, using the specified model for
#the learning process
def generate_adversaries(model, example, delta, prediction, classIdx, steps=30):
    # iterate over the number of steps
    optimizer = Adam()
    sccLoss = SparseCategoricalCrossentropy()
    prediction = prediction.squeeze()
    for step in range(0, steps):
        # record our gradients
        with GradientTape() as tape:
            
            # explicitly indicate that our perturbation vector should
            # be tracked for gradient updates
            tape.watch(delta)
            #print(tf.trainable_variables())
            adversary =  example + delta
           
            predictions = model(adversary, training=False)
            #print(predictions)
            
            loss = sccLoss(convert_to_tensor([classIdx]),predictions)
            
            #if step % 5 == 0:
            #    print("step: {}, loss: {}...".format(step,loss.numpy()))
                
            gradients = tape.gradient(loss, delta)
          
            # update the weights, clip the perturbation vector, and
            # update its values....
            optimizer.apply_gradients([(gradients, delta)])
            delta.assign_add(clip_eps(delta, EPS))
            
            #but keep the time column to 0..
            delta[:,:,3].assign(zeros_like(delta[:,:,3]))
            
            #as well as the rows corresponding to already correct points
            for i in range(len(prediction)):
                if prediction[i]:
                    delta[:,i, :].assign(zeros_like(delta[:,i, :]))
            
            #delta.assign_add(delta)
    
    # return the perturbation vector
    adversary =  example + delta
           
    predictions = model.predict_classes(adversary)
    prediction_or = model.predict_classes(np.array([example]))
    print("New pred: ", predictions, " was ", prediction_or)
    return delta

#method used to correct outliers using customly learned deltas, applied after we have removed the pauses
#if a point was predicted as incorrect but not a pause,
#convert to ECEF coordinates and applicate delta
def apply_deltas(points, times, deltas, predictions):
    
    #array that will host the corrected points
    corrected_points = []        
    #go over each point
    for i in range(points.shape[0]):
        
        point = points[i]
        # if point is incorrect
        if predictions[i] > 1:
            #convert to ecef and apply delta
            cur_lon, cur_lat, cur_alt = point[0], point[1], point[2]
            x, y, z = pyproj.transform(lla, ecef, cur_lon, cur_lat, cur_alt, radians=False)
            x += deltas[i][0]
            y += deltas[i][1]
            z += deltas[i][2]
            #transform back to lat lon and add to list
            lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
            
            corrected_points.append(np.array([lon,lat,alt]))
        else: #keep the original point if it is correct
             corrected_points.append(point)
                  
    #convert to a numpy array and return        
    corrected_points = np.array(corrected_points)
        
    return corrected_points, times
    
    
    
#method used to correct outliers using linear interpolation (based on time)
#here we need all points in the ECEF system to apply interpolation
#using the whole history of the trace
def correct_using_interpolation(points, times, predictions):
    
    #list that will host converted data
    data = []
    
    #convert all points to ECEF
    for i in range(points.shape[0]):
        #get current points
        point = points[i]
        #if it is an outlier add it to the list with NaN as value for the properties
        if predictions[i] > 1:
            
            data.append(np.array([times[i], np.nan, np.nan, np.nan]))
            
        else: #otherwise convert to ECEF and add
            lon, lat, alt = point[0], point[1], point[2]
            x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
            
            data.append(np.array([times[i], x, y, z]))
                  
    #convert to numpy array and create a Pandas dataframe    
    data = np.array(data)
    points_df = pd.DataFrame(data = data, columns=['t', 'x', 'y', 'z'])
    
    #convert timestamps column to date time format
    points_df['t'] = pd.to_datetime(points_df['t'])
    
    #removing timezone infos
    points_df['t'] = points_df['t'].dt.tz_localize(None)
    
    #convert to numeric
    points_df['x'] = pd.to_numeric(points_df['x'], errors='coerce')
    points_df['y'] = pd.to_numeric(points_df['y'], errors='coerce')
    points_df['z'] = pd.to_numeric(points_df['z'], errors='coerce')
    
    #set index
    points_df = points_df.set_index('t')
    
    #interpolate using time (linear interpolation based on time)
    points_df = points_df.interpolate(method='time')
    
    #array that will host the corrected points converted back to lat long
    points = []
    
    
    #create final array by converting back each point to lat/long
    for index, row in points_df.iterrows():
        lon, lat, alt = pyproj.transform(ecef, lla, row['x'], row['y'], row['z'], radians=False)
        points.append(np.array([lon, lat, alt]))
        
    #return results and 
    return np.array(points), np.array(times)
    
#correct the trace using spline interpolation
def correct_using_spline(points, times, predictions, order):
    data = []

    #convert all points to ECEF
    for i in range(points.shape[0]):

        point = points[i]
        
        #but if the point is an outlier use NaN instead of the values
        if predictions[i] > 1:

            data.append(np.array([times[i], np.nan, np.nan, np.nan]))

        else: #otherwise the real values
            lon, lat, alt = point[0], point[1], point[2]
            x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)

            data.append(np.array([times[i], x, y, z]))
    
    #add data to dataframe
    data = np.array(data)
    points_df = pd.DataFrame(data = data, columns=['t', 'x', 'y', 'z'])

    #convert timestamps column to date time format
    points_df['t'] = pd.to_datetime(points_df['t'])

    #removing timezone infos
    points_df['t'] = points_df['t'].dt.tz_localize(None)

    #convert time to timedelta, to keep correct distance between points during interpolation
    points_df['t_i'] = pd.to_timedelta(points_df.t).dt.total_seconds().astype(int)

    #convert to numeric
    points_df['x'] = pd.to_numeric(points_df['x'], errors='coerce')
    points_df['y'] = pd.to_numeric(points_df['y'], errors='coerce')
    points_df['z'] = pd.to_numeric(points_df['z'], errors='coerce')
    
    #set index to our time delta column
    points_df = points_df.set_index('t_i')

    #interpolate using spline and the desired order
    points_df = points_df.interpolate(method='spline', order=order, limit_direction='both')
    
    #convert back to lat long
    final_points = []
    
    
    #create final array converting back to lat/long
    for index, row in points_df.iterrows():
        if not row.isna().any():
            lon, lat, alt = pyproj.transform(ecef, lla, row['x'], row['y'], row['z'], radians=False)
            final_points.append(np.array([lon, lat, alt]))
            
    final_points = np.array(final_points)
    
    return final_points, times
    
    
#correct the trace by applying a Kalman Filter (https://en.wikipedia.org/wiki/Kalman_filter) over the entire trace
#Kalman filters are created and manages using the Filterpy library (https://filterpy.readthedocs.io/en/latest/)
def full_kalman_smoothing(points, times):
    
    print("Points shape: ", points.shape)
    
    #convert everything to ecef
    data = []
    
    for i in range(points.shape[0]):
        #get point
        point = points[i]
        #convert to ECEF
        lon, lat, alt = point[0], point[1], point[2]
        x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
        #append to data
        data.append(np.array([x,y,z]))
    
    data = np.array(data)
     
    
    #calculate also time deltas to adjust state transition matrix of the Kalman Filter based on the measured delta
    datetimes = []
    if not (type(times[0]) == datetime):
        for dt in times:
            datetimes.append(datetime.fromisoformat(dt.replace("Z", "")))
    else:
        datetimes = times
        
    
    time_deltas = []
    for i in range(1, len(datetimes)):
        time_deltas.append((datetimes[i] - datetimes[i-1]).total_seconds())
        
    
    
    # create kalman filter
    dt = 2. #time delta

    f1 = KalmanFilter(dim_x=6, dim_z=3)# 6 state variables, 3 observed

    f1.F = np.array ([[1, dt, 0, 0, 0, 0], #kalman transition matrix
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]], dtype=float)

    f1.R *= 4.9
    f1.Q *= .1

    f1.x = np.array([[data[0][0], 0, data[0][1], 0, data[0][2], 0]], dtype=float).T #starting position, assuming it is correct
    f1.P = np.eye(6) * 500.
    f1.H = np.array([[1, 0, 0, 0, 0, 0], #how do we pass from measurement to position?
                     [0, 0, 1, 0, 0, 0], 
                     [0, 0, 0, 0, 1, 0]])
    
    q = Q_discrete_white_noise(dim=3, dt=dt, var=0.001) #white noise
    f1.Q = block_diag(q, q)
    
    #correction process
    filterpy_data = []

    filterpy_data.append(np.array([data[0][0], data[0][1], data[0][2]])) #first position is assumed to be correct
    
    #for every other position but the first
    for i in range(1, data.shape[0]):
        #take ecef lat, long and alt
        long = data[i][0]
        lat = data[i][1]
        alt = data[i][2]
        #get time delta
        cur_dt = time_deltas[i-1]
    
        #update time delta in state transition matrix with the measured one
        f1.F[0][1] = cur_dt
        f1.F[2][3] = cur_dt
        f1.F[4][5] = cur_dt
        
        #predict next point
        f1.predict()
        #update prediction based on measured value
        f1.update(np.array([long, lat, alt]).T)

        #print(f1.x)
        #add updated version to array
        filterpy_data.append(np.array([f1.x[0][0], f1.x[2][0], f1.x[4][0]]))
    
    filterpy_data = np.array(filterpy_data)

    #now convert adjusted points back to lat lang
    corrected_points = []
    
    for i in range(filterpy_data.shape[0]): #for each point
        #get data
        x = filterpy_data[i][0]
        y = filterpy_data[i][1]
        z = filterpy_data[i][2]
        #convert to lat long and append
        lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
        corrected_points.append(np.array([lon, lat, alt]))
   
    corrected_points = np.array(corrected_points)

    #print("Filtered points: ", corrected_points.shape)
    
    return corrected_points, times
    
    
#correct the trace by applying a Kalman Filter on outliers only
def kalman_smoothing(points, times, predictions):
    
    #print("Points shape: ", points.shape)
    
    #convert everything to ecef
    data = []
    
    for i in range(points.shape[0]): #for each point
        
        point = points[i] #get it
        
        #convert values to ECEF
        lon, lat, alt = point[0], point[1], point[2]
        x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
        
        data.append(np.array([x,y,z]))
    
    data = np.array(data)
    
    #calculate also time deltas to adjust state transition matrix of the Kalman Filter
    datetimes = []
    if not (type(times[0]) == datetime):
        for dt in times:
            datetimes.append(datetime.fromisoformat(dt.replace("Z", "")))
    else:
        datetimes = times
        
    time_deltas = []
    for i in range(1, len(datetimes)):
        time_deltas.append((datetimes[i] - datetimes[i-1]).total_seconds())
    
    # create kalman filter
    dt = 2. #time delta

    f1 = KalmanFilter(dim_x=6, dim_z=3)# 6 state variables, 3 observed

    f1.F = np.array ([[1, dt, 0, 0, 0, 0], #kalman transition matrix
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]], dtype=float)

    f1.R *= 4.9
    f1.Q *= .1

    f1.x = np.array([[data[0][0], 0, data[0][1], 0, data[0][2], 0]], dtype=float).T #starting position, assuming it is correct
    f1.P = np.eye(6) * 500.
    f1.H = np.array([[1, 0, 0, 0, 0, 0], #how do we pass from measurement to position?
                     [0, 0, 1, 0, 0, 0], 
                     [0, 0, 0, 0, 1, 0]])
    
    q = Q_discrete_white_noise(dim=3, dt=dt, var=0.001) #white noise
    f1.Q = block_diag(q, q)
    
    #correction
    filterpy_data = []

    filterpy_data.append(np.array([data[0][0], data[0][1], data[0][2]])) #first position is assumed to be correct
    
    #for each point but the first one
    for i in range(1, data.shape[0]):
        long = data[i][0] #get data
        lat = data[i][1]
        alt = data[i][2]
        
        #get time delta
        cur_dt = time_deltas[i-1]
    
        #update time delta in state transition matrix
        f1.F[0][1] = cur_dt
        f1.F[2][3] = cur_dt
        f1.F[4][5] = cur_dt
    
        #predict next point
        f1.predict()
        
        #update predicted value using the measurement
        f1.update(np.array([long, lat, alt]).T)
        
        #if it is an outlier
        if predictions[i] >= 2:
            #append the corrected version
        #print(f1.x)
            filterpy_data.append(np.array([f1.x[0][0], f1.x[2][0], f1.x[4][0]]))
        else:
            #otherwise keep the original one
            filterpy_data.append(np.array([data[i][0], data[i][1], data[i][2]]))
            #f1.x = np.array([[long, f1.x[0][1], lat, f1.x[0][3], alt, f1.x[0][4]]], dtype=float).T
            f1.x = np.array([[long, 0, lat, 0, alt, 0]], dtype=float).T
    
    filterpy_data = np.array(filterpy_data)

    #now convert back to lat lang
    corrected_points = []
    
    for i in range(filterpy_data.shape[0]): 
        #convert each point back to lat long
        x = filterpy_data[i][0]
        y = filterpy_data[i][1]
        #z = filtered_state_means[i][2]
        z = filterpy_data[i][2]
        lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
        corrected_points.append(np.array([lon, lat, alt]))
   
    corrected_points = np.array(corrected_points)

    #print("Filtered points: ", corrected_points.shape)
    
    return corrected_points, times



#correct the trace by applying a separate Kalman Filter on each subtrace containing only outliers
def separate_kalman_smoothing(points, times, predictions):
    
    print("Points shape: ", points.shape)
    
    #convert everything to ecef
    data = []
    
    for i in range(points.shape[0]): #for each point
        
        point = points[i] #get it
        
        #convert values to ECEF
        lon, lat, alt = point[0], point[1], point[2]
        x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
        
        data.append(np.array([x,y,z]))
    
    data = np.array(data)
    
    #calculate also time deltas to adjust state transition matrix of the Kalman Filter
    datetimes = []
    if not (type(times[0]) == datetime):
        for dt in times:
            datetimes.append(datetime.fromisoformat(dt.replace("Z", "")))
    else:
        datetimes = times
        
    time_deltas = []
    for i in range(1, len(datetimes)):
        time_deltas.append((datetimes[i] - datetimes[i-1]).total_seconds())



    #control variables to be used
    isIncorrect = False
    indexStart = 0
    
    #final data
    filterpy_data = []
    filterpy_data.append(np.array([data[0][0], data[0][1], data[0][2]])) #first position is assumed to be correct
    
    #for each point but the first one
    for i in range(1, data.shape[0]):
        
        long = data[i][0] #get data
        lat = data[i][1]
        alt = data[i][2]
        
        #get time delta
        cur_dt = time_deltas[i-1]
        
        #if the point is wrong
        if predictions[i] >= 2:
            if not isIncorrect: #if the previous point was correct we need to instantiate a Kalman filter
                f1 = KalmanFilter(dim_x=6, dim_z=3)# 6 state variables, 3 observed

                f1.F = np.array ([[1, cur_dt, 0, 0, 0, 0], #kalman transition matrix
                                  [0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, cur_dt, 0, 0],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, cur_dt],
                                  [0, 0, 0, 0, 0, 1]], dtype=float)

                f1.R *= 4.9
                f1.Q *= .1     
            
                f1.x = np.array([[filterpy_data[indexStart][0], 0, filterpy_data[indexStart][1], 0, filterpy_data[indexStart][2], 0]], dtype=float).T #starting position, assuming it is correct
                
                
                f1.P = np.eye(6) * 500.
                f1.H = np.array([[1, 0, 0, 0, 0, 0], #how do we pass from measurement to position?
                                 [0, 0, 1, 0, 0, 0], 
                                 [0, 0, 0, 0, 1, 0]])

                q = Q_discrete_white_noise(dim=3, dt=cur_dt, var=0.001) #white noise
                f1.Q = block_diag(q, q)
                
                #do some predict and update to learn parameters
                for j in range(indexStart + 1, i):
                    f1.predict()
                    
                    f1.update(np.array([filterpy_data[j][0], filterpy_data[j][1], filterpy_data[j][2]]).T)
                    
                    f1.x = np.array([[filterpy_data[j][0], 0, filterpy_data[j][1], 0, filterpy_data[j][2], 0]], dtype=float).T
                
            #update time delta
            f1.F[0][1] = cur_dt
            f1.F[2][3] = cur_dt
            f1.F[4][5] = cur_dt
            
            #predict next point
            f1.predict()     

            #update predicted value using the measurement
            f1.update(np.array([long, lat, alt]).T)

            filterpy_data.append(np.array([f1.x[0][0], f1.x[2][0], f1.x[4][0]])) #append filtered data

            isIncorrect = True
        
        else:
            filterpy_data.append(np.array([data[i][0], data[i][1], data[i][2]])) #append current data
            
            if isIncorrect:  #if previous point was incorrect the start for next kalman is the curren one
                indextStart = i
            elif i - indexStart == 4: #otherwise if we have more than 5 points correct in a row increment index by one
                indexStart += 1
                
            isIncorrect = False #set current point as correct
            
    filterpy_data = np.array(filterpy_data)

    #now convert back to lat lang
    corrected_points = []
    
    for i in range(filterpy_data.shape[0]): 
        #convert each point back to lat long
        x = filterpy_data[i][0]
        y = filterpy_data[i][1]
        #z = filtered_state_means[i][2]
        z = filterpy_data[i][2]
        lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
        corrected_points.append(np.array([lon, lat, alt]))
   
    corrected_points = np.array(corrected_points)

    #print("Filtered points: ", corrected_points.shape)
    
    return corrected_points, times


#correct the trace by applying a separate Kalman Filter on each subtrace containing only outliers
def separate_bidirectional_kalman_smoothing(points, times, predictions, R = 4.9):
    
    #print("Points shape: ", points.shape)
    
    #convert everything to ecef
    data = []
    
    for i in range(points.shape[0]): #for each point
        
        point = points[i] #get it
        
        #convert values to ECEF
        lon, lat, alt = point[0], point[1], point[2]
        x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
        
        data.append(np.array([x,y,z]))
    
    data = np.array(data)
    
    #calculate also time deltas to adjust state transition matrix of the Kalman Filter
    datetimes = []
    if not (type(times[0]) == datetime):
        for dt in times:
            datetimes.append(datetime.fromisoformat(dt.replace("Z", "")))
    else:
        datetimes = times
        
    time_deltas = []
    for i in range(1, len(datetimes)):
        time_deltas.append((datetimes[i] - datetimes[i-1]).total_seconds())



    
    #for each point but the first one
    outliers = []
    cur_start = -1
    cur_end = -1
    isOutlier = False
    
    #identify all areas which need a correction
    for i in range(1, data.shape[0]):
        
        long = data[i][0] #get data
        lat = data[i][1]
        alt = data[i][2]
        
        #get time delta
        cur_dt = time_deltas[i-1]
        
        #if the point is wrong
        if predictions[i] >= 2:
            if isOutlier:
                cur_end = i
            else:
                cur_start = i
                cur_end = i
            
            isOutlier = True
        else:
            if isOutlier: #append current outlying area
                outliers.append({'start' : cur_start, 'end' : cur_end})
            
            isOutlier = False
                
    #check we do not have outlying areas at the end
    if isOutlier: #append current outlying area
        outliers.append({'start' : cur_start, 'end' : cur_end})
            
    #print("Number of outliers area: ", len(outliers))
    
    #for each outlying area
    corrected_areas = []
    for cur_outlier in outliers: 
            
            
        #print("Cur outlier area: ", cur_outlier['start'], " - ", cur_outlier['end'])     
        cur_outlier_corrected = []

        #create Kalman filter and go over trace from at most 5 points before
        cur_start = cur_outlier['start']

        #check how many points can we go back
        for i in range(cur_start, cur_start - 6, -1):
            if predictions[i] < 2:
                cur_start -= 1
            else:
                break

        #initialise filter    
        cur_dt = 2.

        f1 = KalmanFilter(dim_x=6, dim_z=3)# 6 state variables, 3 observed
        f1.F = np.array ([[1, cur_dt, 0, 0, 0, 0], #kalman transition matrix
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, cur_dt, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, cur_dt],
                          [0, 0, 0, 0, 0, 1]], dtype=float)
        f1.R *= R
        f1.Q *= .1     
        f1.x = np.array([[data[cur_start][0], 0, data[cur_start][1], 0, data[cur_start][2], 0]], dtype=float).T #starting position
        f1.P = np.eye(6) * 500.
        f1.H = np.array([[1, 0, 0, 0, 0, 0], #how do we pass from measurement to position?
                         [0, 0, 1, 0, 0, 0], 
                         [0, 0, 0, 0, 1, 0]])
        q = Q_discrete_white_noise(dim=3, dt=cur_dt, var=0.001) #white noise
        f1.Q = block_diag(q, q)

        #apply to points and store correction
        for i in range(cur_start, cur_outlier['end'] + 1):

            #update time delta
            #get time delta
            cur_dt = time_deltas[i-1]

            #update time delta
            f1.F[0][1] = cur_dt
            f1.F[2][3] = cur_dt
            f1.F[4][5] = cur_dt

            #predict
            f1.predict()

            #update
            f1.update(np.array([data[i][0], data[i][1], data[i][2]]).T)

            #append to list if point is incorrect
            if predictions[i] >= 2:
                cur_outlier_corrected.append(np.array([f1.x[0][0], f1.x[2][0], f1.x[4][0]]))
            else:
                f1.x = np.array([[data[i][0], 0, data[i][1], 0, data[i][2], 0]], dtype=float).T

       #now initialise filter again and 

        cur_end = cur_outlier['end']

        #check how many points can we go back
        for i in range(cur_end, cur_end + 6):
            if predictions[i] < 2:
                cur_end += 1
            else:
                break


       #initialise filter
        cur_dt = 2.
        f1 = KalmanFilter(dim_x=6, dim_z=3)# 6 state variables, 3 observed
        f1.F = np.array ([[1, cur_dt, 0, 0, 0, 0], #kalman transition matrix
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, cur_dt, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, cur_dt],
                          [0, 0, 0, 0, 0, 1]], dtype=float)
        f1.R *= R
        f1.Q *= .1     
        f1.x = np.array([[data[cur_end][0], 0, data[cur_end][1], 0, data[cur_end][2], 0]], dtype=float).T #starting position
        f1.P = np.eye(6) * 500.
        f1.H = np.array([[1, 0, 0, 0, 0, 0], #how do we pass from measurement to position?
                         [0, 0, 1, 0, 0, 0], 
                         [0, 0, 0, 0, 1, 0]])
        q = Q_discrete_white_noise(dim=3, dt=cur_dt, var=0.001) #white noise
        f1.Q = block_diag(q, q)

        #now apply it backward
        cur_item = len(cur_outlier_corrected) - 1
        for i in range(cur_end, cur_outlier['start'] - 1, -1):
            #update time delta
            #get time delta
            cur_dt = time_deltas[i-1]

            #update time delta
            f1.F[0][1] = cur_dt
            f1.F[2][3] = cur_dt
            f1.F[4][5] = cur_dt

            #predict
            f1.predict()

            #update
            f1.update(np.array([data[i][0], data[i][1], data[i][2]]).T)

            #append to list if point is incorrect
            if predictions[i] >= 2:
                #print("\tCorrecting point: ", i)
                #print("\t\tBefore sum: ", cur_outlier_corrected[cur_item]) 
                #print("\t\tSumming: ",  np.array([f1.x[0][0], f1.x[2][0], f1.x[4][0]]))
                cur_outlier_corrected[cur_item] += np.array([f1.x[0][0], f1.x[2][0], f1.x[4][0]])
                #print("\t\tAfter sum: ", cur_outlier_corrected[cur_item]) 
                cur_outlier_corrected[cur_item] /= 2
                #print("\t\tMean: ", cur_outlier_corrected[cur_item]) 
                
                cur_item -= 1
            else:
                f1.x = np.array([[data[i][0], 0, data[i][1], 0, data[i][2], 0]], dtype=float).T

        #now apply correction
        cur_outlier_corrected = np.array(cur_outlier_corrected)
        #print("\tCur outlier correction shape: ", cur_outlier_corrected.shape)
        data[cur_outlier['start'] : cur_outlier['end'] + 1][:] = cur_outlier_corrected
        
    #now convert back to lat lang
    corrected_points = []
    
    for i in range(data.shape[0]): 
        #convert each point back to lat long
        x = data[i][0]
        y = data[i][1]
        #z = filtered_state_means[i][2]
        z = data[i][2]
        lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
        corrected_points.append(np.array([lon, lat, alt]))
   
    corrected_points = np.array(corrected_points)

    #print("Filtered points: ", corrected_points.shape)
    
    return corrected_points, times