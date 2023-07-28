#prevent tensorflow warnings to appear 
import os
from pathlib import Path
import sys

#import libraries 
import gpxpy
import gpxpy.gpx
import numpy as np
from gpsclean import gpsclean_transform as gt
from gpsclean import FullTraining as ft
import tflite_runtime.interpreter as tflite
import tflite_runtime
from gpsclean import Correction as tc
from geojson import Feature, LineString, FeatureCollection, Point, dump
import argparse
from argparse import RawTextHelpFormatter
from art import *
import matplotlib

#current version of the program
__VERSION__ = "1.0.1"

def main(args=None):

    #add description of the program in the arguments parser 
    parser=argparse.ArgumentParser(description='Applies a machine learning model to recognise errors in your GPX input trace.\nThe output is represented by the corrected version of your trace, always in the GPX format (Kalman filters are applied on outliers at this stage).\nOptionally, you can have as a second output the original trace with the predicted errors in the GeoJSON format (you can view and inspect it on https://api.dawnets.unibz.it/ ).\nMoreover, a third option is to have the mean of the predictions for each point as output, represented by a continuous color (correct = green, pause = yellow, outlier = red, indoor = gray). The output GeoJSON can be visually inspected at: https://geojson.io. \n\nFor more info please visit: https://gitlab.inf.unibz.it/gps-clean/transform-and-model', formatter_class=RawTextHelpFormatter)

    #add argument: input trace in GPX format (mandatory)
    parser.add_argument("input_trace",help="Your input GPS trace in the GPX format. Example: gps_trace.gpx")

    #add argument: boolean to output also the predictions (optional)
    parser.add_argument("-op","--outputPredictions",help="Output predicted points in a GeoJSON file (_predicted)", action="store_true")
    #add argument: boolean to output also the mean predictions colored (optional)
    parser.add_argument("-mpc","--meanPredictionColored",help="Output the mean prediction of each point with its continuous color (correct = green, pause = yellow, outlier = red, indoor = gray) in a GeoJSON file (_predictedColors)", action="store_true")
    #add argument: integer to represent the chosen R parameters for the measurement nois (optional)
    parser.add_argument("-r","--RMeasurementNoise",type=float, default=4.9, help="R parameter representing the measurement noise for the Kalman Filter")
    #add argument: print program version and exit
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __VERSION__)
    #parse the arguments 
    args = parser.parse_args(args)

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

    # check we have at least a point
    if points.size == 0:
        print("""TRACE ERROR:
No points have been detected in your GPS trace. Please note the tool expects
at least a track with a segment containing a point to be considered valid and
waypoints are excluded.
You can consult https://wiki.openstreetmap.org/wiki/GPX for more.
              """)
        return

    #create the deltas using the associated function 
    deltas = gt.create_deltas(points, times)

    #load the already trained model
    print("Loading the model...")
    dirPath = Path(__file__).absolute().parent
    modelPath = dirPath.joinpath("data/model.tflite")
    interpreter = tflite.Interpreter(model_path=str(modelPath))
    
    #predict the trace, creating segments using a window of 15 points and a step of 2 
    print("Predicting points...")
    segments, indices, predictions = ft.predict_trace(interpreter, deltas, 15, 2)

    #compress the predictions obtaining a point to point prediction
    pred_points, mean_pred_points = ft.compress_trace_predictions(predictions, indices, 4)
    
    #insert prediction for starting point, assumed to be correct 
    full_pred_points=np.insert(pred_points, 0, 0)
    full_mean_pred_points=np.insert(mean_pred_points, 0, 0)

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
    
    #check if we need to output also the mean colored predictions 
    if args.meanPredictionColored:
        print("Exporting also mean predictions colored...")
        colorMap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', ['green', 'yellow', 'red', 'gray'])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=3)
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colorMap)
                
        features = []
        for i in range(points.shape[0]):
            cur_point = Point((points[i][0], points[i][1], points[i][2]))
            rgba = mappable.to_rgba(full_mean_pred_points[i])
            color = matplotlib.colors.rgb2hex(rgba)
            cur_properties = {'marker-color': color, 'marker-size': 'small', 'meanPrediction': float(full_mean_pred_points[i]), 'prediction': int(full_pred_points[i]), 'id': int(i)}
            cur_feature = Feature(geometry = cur_point, properties = cur_properties)
            features.append(cur_feature)
        
        feature_collection = FeatureCollection(features)
        
        #write on file
        with open(filename + "_predictedColors.geojson", 'w') as f:
            dump(feature_collection, f)
        
if __name__ == '__main__':
    main()