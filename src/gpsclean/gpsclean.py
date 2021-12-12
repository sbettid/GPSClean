#prevent tensorflow warnings to appear 
import os
from pathlib import Path
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #or any {'0','1','2'} 
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

#import libraries 
import gpxpy
import gpxpy.gpx
import numpy as np
from gpsclean import gpsclean_transform as gt
from gpsclean import FullTraining as ft
from tensorflow import keras
from tensorflow.autograph import set_verbosity
from gpsclean import Correction as tc
from geojson import Feature, LineString, FeatureCollection, dump
import argparse
from argparse import RawTextHelpFormatter
from art import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#deactivate autograph warnings in tensorflow
set_verbosity(0)

#current version of the program
__VERSION__ = "0.2.0"

def main():

    #add description of the program in the arguments parser 
    parser=argparse.ArgumentParser(description='Applies a machine learning model to recognise errors in your GPX input trace.\nThe output is represented by the corrected version of your trace, always in the GPX format (Kalman filters are applied on outliers at this stage).\n Optionally, you can have as a second output the original trace with the predicted errors in the GeoJSON format (you can view and inspect it on https://api.dawnets.unibz.it/ ).\nFor more info please visit: https://gitlab.inf.unibz.it/gps-clean/transform-and-model', formatter_class=RawTextHelpFormatter)

    #add argument: input trace in GPX format (mandatory)
    parser.add_argument("input_trace",help="Your input GPS trace in the GPX format. Example: gps_trace.gpx")

    #add argument: boolean to output also the predictions (optional)
    parser.add_argument("-op","--outputPredictions",help="Output predicted points in a GeoJSON file", action="store_true")
    #add argument: integer to represent the chosen R parameters for the measurement nois (optional)
    parser.add_argument("-r","--RMeasurementNoise",type=float, default=4.9, help="R parameter representing the measurement noise for the Kalman Filter")
    #add argument: print program version and exit
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __VERSION__)
    #parse the arguments 
    args = parser.parse_args()

    #print program name using ascii art
    tprint("GPSClean")
    print("Version: ", __VERSION__, "\n\n")

    #define dictionary of labels 
    labels_dict ={ 0: 'Correct', 1 : 'Pause', 2 : 'Outlier', 3 : 'Indoor' }

    #read input trace 
    print("Reading your trace...")
    gpx_file = open(args.input_trace,'r')
    gpx = gpxpy.parse(gpx_file)

    #get points and times from the trace as numpy arrays 
    points = []
    times = []
    for track in gpx.tracks: 
        for segment in track.segments:
            for point in segment.points:
                points.append(np.array([point.longitude, point.latitude, point.elevation, point.time]))
                times.append(point.time)

    points = np.array(points)
    times = np.array(times)
    #create the deltas using the associated function 
    deltas = gt.create_deltas(points, times)

    #load the already trained model
    print("Loading the model...")
    dirPath = Path(__file__).absolute().parent
    model_path = dirPath.joinpath("data/model_42t_traces.h5")

    model = keras.models.load_model(model_path)

    #predict the trace, creating segments using a window of 15 points and a step of 2 
    print("Predicting points...")
    segments, indices, predictions = ft.predict_trace(model, deltas, 15, 2)

    #compress the predictions obtaining a point to point prediction
    pred_points = ft.compress_trace_predictions(predictions, indices, 4)

    #insert prediction for starting point, assumed to be correct 
    full_pred_points=np.insert(pred_points, 0, 0)

    #remove the pauses 
    reduced_points, reduced_times, reduced_delta, reduced_predictions, original_trace, original_times = tc.remove_pauses(points[:,:3], times, full_pred_points, None)

    #now we can correct all other points using Kalman Filters only on them
    print("Correcting outliers...")
    kalman_trace, kalman_times = tc.separate_bidirectional_kalman_smoothing(reduced_points, reduced_times,
                                                                            reduced_predictions, args.RMeasurementNoise)

    #recreate a GPX file containing the cleaned trace 
    #create empty trace 
    print("Exporting corrected trace...")
    gpx = gpxpy.gpx.GPX()
    #Create first track in our GPX:
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    #Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    #loop over points and add to segment
    for i in range(kalman_trace.shape[0]):
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(longitude=kalman_trace[i][0], latitude=kalman_trace[i][1], elevation=kalman_trace[i][2], time=kalman_times[i]))

    #get original filename
    filename = args.input_trace.rsplit(".",1)[0]

    #write gpx trace as file
    with open(filename + "_cleaned.gpx",'w')as f:
        f.write(gpx.to_xml())

    #check if we need to output also the predictions 
    if args.outputPredictions:
        print("Exporting also predicted trace...")
        #create geojson file 
        #first create and populate lists of original points
        geojson_coords=[]
        geojson_times=[]
        for i in range(points.shape[0]):
            geojson_coords.append((points[i][0], points[i][1], points[i][2]))
            geojson_times.append(times[i].isoformat())

        #then create list of annotations, with a one pass on the points 
        annotations=[]
        cur_start=-1
        cur_end=-1
        cur_type=0
        cur_started=False

        for i in range(full_pred_points.shape[0]):
            if full_pred_points[i] > 0:
                if cur_started:
                    if full_pred_points[i] == cur_type:
                        cur_end = i
                    else: 
                        annotations.append({'start' : cur_start, 'end' : cur_end, 'annotation' : labels_dict[cur_type]})   
                        cur_start = i
                        cur_end = i 
                        cur_type = full_pred_points[i]
                        cur_started = True
                else:
                    cur_start = i
                    cur_end = i
                    cur_type = full_pred_points[i]
                    cur_started = True
            else:
                if cur_started:     
                    annotations.append({'start' : cur_start, 'end' : cur_end, 'annotation' : labels_dict[cur_type]})
                cur_started = False
        if cur_started:
            annotations.append({'start' : cur_start, 'end' : cur_end - 1, 'annotation' : labels_dict[cur_type]})


        #now set times and annotations as properties
        properties = {'coordTimes' : geojson_times, 'annotations' : annotations}

        #create line string from points
        line = LineString(geojson_coords)

        #create a feature starting from line string and properties
        feature = Feature(geometry = line, properties = properties)

        #create a feature collection containing our feature
        feature_collection = FeatureCollection([feature])

        #write on file
        with open(filename + "_predicted.geojson", 'w') as f:
            dump(feature_collection, f)

if __name__ == '__main__':
    main()