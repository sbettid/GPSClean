# GPSClean

## What is GPSClean?

### The goal

GPSClean is an Open Source application developed with the goal of automatically detect and correct errors in GPS traces, exported in the widely adopted GPX format, using machine learning techniques and without prior geographical knowledge. It is the result of the research undertaken for my thesis project during the Master in Computational Data Science (University of Bolzano). 

Feel free to try the application and report any feedback on the project's [Github page](https://github.com/sbettid/GPSClean). 

### How does it work?

The activity undertaken by GPSClean can be divided in three main steps: 

- Data Preprocessing
- Error detection
- Error correction

The following sections briefly explain the three activities, for further details I invite you to consult the jupyter notebooks available at the [Transform and Model](https://gitlab.inf.unibz.it/gps-clean/transform-and-model) repository. 

### Data preprocessing

The first operations performed by GPSClean are related to data pre-processing. In particular, the machine learning model adopted by the application considers the differences (deltas) between the registered points, since it does not use any geographical knowledge abou the region in which the trace was registered. 

For this reason, the latitude, longitude and altitude original coordinates are converted to ECEF (Earth Centered Earth Fixed) coordinates, enabling so a simpler calculation of the deltas given their cartesian nature. 

The second pre-processing step is the calculation of the deltas between two consecutive points, with the result being the input that will be provided to the machine learning model. 

### Error detection

The error detection step is based on a previously trained machine learning model. The model, based on a neural network composed by Bidirectional LSTM (Long-Short Term Memory) cells, was trained on a set of annotated GPS traces, namely traces on which the errors were explicitly marked using an ad-hoc developed web-based tool: [Track annotation](https://api.dawnets.unibz.it/). 

The considered errors are the following: 

- Pauses: when the selected movements were a pause and are the result of GPS reception errors.
- Outliers: when the point, due to a reception error, is in an incorrect position.
- Indoor: a point collected while being indoor in a building.

The result of the application of the model is an annotated trace, used then in the following step. 

### Error correction

#### A special type of error: pauses

Firstly, it should be noted how one of the considered error categories, namely pauses, differs fromthe others, when it comes to correction. In fact, points representing pauses should appear, in the ideal case, stacked one upon the other, given that the position of the receiver did not change in the considered time frame.  However, often pauses are represented, in the GPS trace, by a cloud of points resulting from a measurement error bythe GPS sensor. For the aforementioned reasons, points identified by the model as belonging to a pause can bedirectly removed, resulting so in a more smoothed trace with no unnecessary points

#### Correcting misplaced points

Pauses, as explained in the previous section, can be directly removed, step that can not be performed on misplaced points, which need then to be corrected.

In order to obtain the most appropriate error correction, various techniques were investigated during the thesis research. The selected approach was denoted Bidirectional Separate Kalman Filters, a variation of the widely adopted Kalman Filter. 

A Kalman Filter is a structure which predicts the following observation (in our case position) in time series and joins it together with the observed value to obtain the final observation. The predicted value of the next observation, represented by a state vector, is based on the covariance and relation of the different components, combined with an external influence and uncertainty. The values for the covariance and uncertainty were derived from previous researches in the field. 

The proposed variation applies two separate Kalman Filters for each detected error area, one in each direction, and takes the mean of the resulting corrected positions as the final ones, smoothing so the initial corrections exploiting the availability of the entire time series. 

The corrected version of the trace is then exported in the GPX format. 

## Running the application

In order to run the application, please follow the subsequent steps: 

1) Install the package by executing the following command: `pip install -i https://test.pypi.org/simple/ --extra-index-url https://google-coral.github.io/py-repo/ gpsclean`
2) Now you are ready to clean your first GPS trace by executing the following command: `gpsclean path/to/your_trace.gpx`
3) In the same folder where the original trace resides, you will find a cleaned version of the trace with the suffix "_cleaned"

## Downloading a pre-built all-in-one executable

On the other hand, if you prefer to download a all-in-one executable which does not require any prior software installed (not even Python), you can find pre-built all-in-one packages in the [Release section](https://github.com/sbettid/GPSClean/releases) of the repository. 


## How to manually create the all-in-one executable

1) Clone the [repository](https://github.com/sbettid/GPSClean) and open a shell in the `src/gpsclean` folder
2) Execute the following command: `pyinstaller -F --add-data "data/model_42t_traces.h5;data" gpsclean.py`
3) At the end of the process, a dist folder can be found inside the `src/gpsclean` folder, containing the packaged executable. 