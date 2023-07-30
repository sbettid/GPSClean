#this file contains the functions used to correct a trace based on the generated predictions
import numpy as np
from . import PointsUtil as gt
from . import DateTimeUtil as date_util
from . import KalmanFilterFactory as kalman_factory

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
                        
                    cur_point += 1
   
    #return the results as numpy arrays
    return np.array(points), np.array(coordTimes), np.array(reduced_deltas), np.array(reduced_predictions), np.array(original_points), np.array(original_times)  
    
#correct the trace by applying a Kalman Filter (https://en.wikipedia.org/wiki/Kalman_filter) over the entire trace
#Kalman filters are created and manages using the Filterpy library (https://filterpy.readthedocs.io/en/latest/)
def full_kalman_smoothing(points, times):
        
    #convert everything to ecef
    data = gt.convert_lla_points_to_ecef(points)
    
    #calculate also time deltas to adjust state transition matrix of the Kalman Filter based on the measured delta
    time_deltas = date_util.calculate_timedeltas(times)
    
    # create kalman filter
    kalman_filter = kalman_factory.get_kalman_filter_config_from_data(data)
    
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
        kalman_filter.F[0][1] = cur_dt
        kalman_filter.F[2][3] = cur_dt
        kalman_filter.F[4][5] = cur_dt
        
        #predict next point
        kalman_filter.predict()
        #update prediction based on measured value
        kalman_filter.update(np.array([long, lat, alt]).T)

        #add updated version to array
        filterpy_data.append(np.array([kalman_filter.x[0][0], kalman_filter.x[2][0], kalman_filter.x[4][0]]))
    
    filterpy_data = np.array(filterpy_data)

    #now convert adjusted points back to lat lang
    corrected_points = gt.convert_ecef_points_to_lla(filterpy_data)
    
    return corrected_points, times
    
    
#correct the trace by applying a Kalman Filter on outliers only
def kalman_smoothing(points, times, predictions):
        
    #convert everything to ecef
    data = gt.convert_lla_points_to_ecef(points)
    
    #calculate also time deltas to adjust state transition matrix of the Kalman Filter
    time_deltas = date_util.calculate_timedeltas(times)
    
    # create kalman filter
    kalman_filter = kalman_factory.get_kalman_filter_config_from_data(data)
    
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
        kalman_filter.F[0][1] = cur_dt
        kalman_filter.F[2][3] = cur_dt
        kalman_filter.F[4][5] = cur_dt
    
        #predict next point
        kalman_filter.predict()
        
        #update predicted value using the measurement
        kalman_filter.update(np.array([long, lat, alt]).T)
        
        #if it is an outlier
        if predictions[i] >= 2:
            #append the corrected version
            filterpy_data.append(np.array([kalman_filter.x[0][0], kalman_filter.x[2][0], kalman_filter.x[4][0]]))
        else:
            #otherwise keep the original one
            filterpy_data.append(np.array([data[i][0], data[i][1], data[i][2]]))
            kalman_filter.x = np.array([[long, 0, lat, 0, alt, 0]], dtype=float).T
    
    filterpy_data = np.array(filterpy_data)

    #now convert back to lat lang
    corrected_points = gt.convert_ecef_points_to_lla(filterpy_data)
    
    return corrected_points, times



#correct the trace by applying a separate Kalman Filter on each subtrace containing only outliers
def separate_kalman_smoothing(points, times, predictions):
        
    #convert everything to ecef
    data = gt.convert_lla_points_to_ecef(points)
    
    #calculate also time deltas to adjust state transition matrix of the Kalman Filter
    time_deltas = date_util.calculate_timedeltas(times)

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
                kalman_filter = kalman_factory.get_kalman_filter_config_from_data(data, dt = cur_dt, starting_index= indexStart)
                
                #do some predict and update to learn parameters
                for j in range(indexStart + 1, i):
                    kalman_filter.predict()
                    
                    kalman_filter.update(np.array([filterpy_data[j][0], filterpy_data[j][1], filterpy_data[j][2]]).T)
                    
                    kalman_filter.x = np.array([[filterpy_data[j][0], 0, filterpy_data[j][1], 0, filterpy_data[j][2], 0]], dtype=float).T
                
            #update time delta
            kalman_filter.F[0][1] = cur_dt
            kalman_filter.F[2][3] = cur_dt
            kalman_filter.F[4][5] = cur_dt
            
            #predict next point
            kalman_filter.predict()     

            #update predicted value using the measurement
            kalman_filter.update(np.array([long, lat, alt]).T)

            filterpy_data.append(np.array([kalman_filter.x[0][0], kalman_filter.x[2][0], kalman_filter.x[4][0]])) #append filtered data

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
    corrected_points = gt.convert_ecef_points_to_lla(filterpy_data)
    
    return corrected_points, times


#correct the trace by applying a separate Kalman Filter on each subtrace containing only outliers
def separate_bidirectional_kalman_smoothing(points, times, predictions, R = 4.9):
        
    #convert everything to ecef
    data = gt.convert_lla_points_to_ecef(points)
    
    #calculate also time deltas to adjust state transition matrix of the Kalman Filter
    time_deltas = date_util.calculate_timedeltas(times)
    
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
                
    #for each outlying area
    for cur_outlier in outliers: 
            
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
        kalman_filter = kalman_factory.get_kalman_filter_config_from_data(data, starting_index = cur_start, R=R)

        #apply to points and store correction
        for i in range(cur_start, cur_outlier['end'] + 1):

            #update time delta
            #get time delta
            cur_dt = time_deltas[i-1]

            #update time delta
            kalman_filter.F[0][1] = cur_dt
            kalman_filter.F[2][3] = cur_dt
            kalman_filter.F[4][5] = cur_dt

            #predict
            kalman_filter.predict()

            #update
            kalman_filter.update(np.array([data[i][0], data[i][1], data[i][2]]).T)

            #append to list if point is incorrect
            if predictions[i] >= 2:
                cur_outlier_corrected.append(np.array([kalman_filter.x[0][0], kalman_filter.x[2][0], kalman_filter.x[4][0]]))
            else:
                kalman_filter.x = np.array([[data[i][0], 0, data[i][1], 0, data[i][2], 0]], dtype=float).T

       #now initialise filter again and 

        cur_end = cur_outlier['end']

        #check how many points can we go back
        for i in range(cur_end, cur_end + 6):
            if predictions[i] < 2:
                cur_end += 1
            else:
                break


        #initialise filter
        kalman_filter = kalman_factory.get_kalman_filter_config_from_data(data, starting_index = cur_end, R=R)

        #now apply it backward
        cur_item = len(cur_outlier_corrected) - 1
        for i in range(cur_end, cur_outlier['start'] - 1, -1):
            #update time delta
            #get time delta
            cur_dt = time_deltas[i-1]

            #update time delta
            kalman_filter.F[0][1] = cur_dt
            kalman_filter.F[2][3] = cur_dt
            kalman_filter.F[4][5] = cur_dt

            #predict
            kalman_filter.predict()

            #update
            kalman_filter.update(np.array([data[i][0], data[i][1], data[i][2]]).T)

            #append to list if point is incorrect
            if predictions[i] >= 2:
                cur_outlier_corrected[cur_item] += np.array([kalman_filter.x[0][0], kalman_filter.x[2][0], kalman_filter.x[4][0]])
                cur_outlier_corrected[cur_item] /= 2
                
                cur_item -= 1
            else:
                kalman_filter.x = np.array([[data[i][0], 0, data[i][1], 0, data[i][2], 0]], dtype=float).T

        #now apply correction
        cur_outlier_corrected = np.array(cur_outlier_corrected)
        data[cur_outlier['start'] : cur_outlier['end'] + 1][:] = cur_outlier_corrected
        
    #now convert back to lat lang
    corrected_points = gt.convert_ecef_points_to_lla(data)
    
    return corrected_points, times