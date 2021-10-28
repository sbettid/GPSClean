# GPSClean

## How to create the executable

1) Clone the repository and open a shell in the main folder
2) Execute the following command: `pyinstaller -F --add-data "model_42t_traces.h5;." gpsclean.py`
3) At the end of the process, a dist folder can be found inside the main folder, containing the packaged executable. 