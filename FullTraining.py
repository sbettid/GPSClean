#This file contains functions used to perform the training on an entire dataset
#and to predict a newly acquired trace
import numpy as np
import pyproj
import sys
from tensorflow import squeeze
import pandas as pd

#setting max int used for masking
max_int = sys.maxsize

ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        

#this function takes the generated segments and keeps those that
#contain at least an annotated point. Points annotated as correct are then
#unified to those not annotated. 
def get_train_data(X,y, indices, ids):
    
    trainset = []
    labels = []
    #for each segment
    for i in range(X.shape[0]):
        index = indices[i]

        intersection = set.intersection(set(ids), set(index))
        #if the intersection between the segments indices and the annotated ones is not empty
        if len(intersection) > 0:
            trainset.append(X[i])#add segment
            cur_label = np.where(y[i] == 99, 0, y[i])#add label unifying points
            labels.append(cur_label)
            
    trainset = np.array(trainset)
    labels = np.array(labels)

    print("Train data shape: ", trainset.shape)
    
    
    return trainset, labels


#this function is used to predict a newly acquired trace, given an already trained model 
def predict_trace(model, trace, window_size, step):
    
    segments = []
    indices = []
    
    annotatedIDs = set([])
    notAnnotatedIDs = set([])
    
    act_counter = 1
    
    #first, we need to generate segments
    for point_index in range(0,len(trace),step):
            
            segmentAdded = False
            #as usual, if we have room for an entire segment just take it
            if point_index + window_size < len(trace):
                
                #if sum(label[act_counter - 1][point_index:point_index+window_size]) > 0:
                
                #append segment and indices
                segments.append(trace[point_index:point_index+window_size])


                cur_indices = np.arange(point_index,point_index+window_size)
                indices.append(cur_indices)

                #segmentAdded = True
                
            else: #otherwise we need first to pad
                #pad first segment and label
                cur_segment = trace[point_index:]
                
                pad = np.full((window_size - len(cur_segment), 4), max_int)
                ind_pad = np.full(window_size - len(cur_segment), max_int)
                
                #append padded version
                cur_segment = np.append(cur_segment, pad, axis=0)

                cur_ind = np.concatenate((np.arange(point_index,point_index+len(trace[point_index:])), ind_pad))

                segments.append(cur_segment)
                indices.append(cur_ind)    

                      
    
    
    #convert to Numpy arrays
    segments = np.array(segments)
    indices = np.array(indices)
    
   
    
    #print(segments.shape)
    #print(labels.shape)
    #print(indices.shape)
    #print(ids.shape)
    
    #predictions = model.predict_classes(segments)
    #get prediction for each segment
    #use the model to predict the segments
    predictions_raw = model.predict(segments)
    #take the class with the largest probability
    predictions=np.argmax(predictions_raw,axis=2)
    
    #print("Predictions shape: ", predictions.shape)
    
    #return everything
    return segments, indices, predictions


#this function is used to compress the delta corrections learned based on points indices
def compress_delta_corrections(segments, indices, predictions):
    
    #get unique indices
    unique_indices = np.unique(indices.flatten())
    
    #create dictionary
    points = {}
    
    #initialise coordinates and count to 0s for each point
    for point in unique_indices:
        points[point] = {'coords' : np.zeros(4), 'count' : 0}
    
    #sum the valuse we got by the deltas of different segments
    for i in range(indices.shape[0]):
        for j in range(len(indices[i])):
            if not indices[i][j] == max_int and not predictions[indices[i][j]] < 2:
                #print("Point at indices", i, " ", j, " : ")
                points[indices[i][j]]['coords'] += squeeze(segments[i])[j]
                points[indices[i][j]]['count'] += 1
           
            
    #take the average of the corrections
    for point in points:
        if points[point]['count'] > 0:
            points[point]['coords'] /= points[point]['count']
        
    #rebuild trace sorting the points by their index
    sorted_keys = sorted(points)
    
    new_trace = []
    
    for key in sorted_keys: #sort the points...
        if not key == max_int: 
            new_trace.append(points[key]['coords']) #... appending the coords
            
    return np.array(new_trace)


#this function is used to compress the predictions by the previous function 
#using the points indices
def compress_trace_predictions(predictions, indices, n_classes):
    
    
    #crate the points labels matrix
    unique_indices = np.unique(indices.flatten())
    points = { key : np.zeros(n_classes) for key in unique_indices}
    
    #loop over predictions and update the correspondent label
    for k in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            points[indices[k][j]][predictions[k][j]] += 1
                
            
    #apply majority voting and record multiple classifications for statistics use
    for point in points: 
        points[point] = np.argmax(points[point])
        
    #rebuild the trace sorting the points
    sorted_keys = sorted(points)
    
    trace_predictions = []
    
    for key in sorted_keys: #the for each point
        if not key == max_int:
            trace_predictions.append(points[key]) #we append to the list the final prediction
            
    return np.array(trace_predictions)


def convert_to_ECEF(points):
    
    ecef_points = []
    
    for point in points:
        cur_lon, cur_lat, cur_alt = point[0], point[1], point[2]
        x, y, z = pyproj.transform(lla, ecef, cur_lon, cur_lat, cur_alt, radians=False)
        
        ecef_points.append(np.array([x, y, z]))
        
    
    return np.array(ecef_points)
