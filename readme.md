# FaceMask WebCam Detection

This was created as part of the module Sensors networks in Université de Technologie de Troyes (UTT) in autumn 2020. It has been created by Thomas Simon, Pierre Sinnave and myself.  
The goal was to create a model allowing through deeplearning to recognize the wearing of a mask by the persons in front of the camera. 

## Requirements

* Install Python 3 apt-get install python3
* Install the required python libs `pip install matplotlib numpy opencv_python scikit_learn tensorflow`

## How to use
1. Create the trained model to recognize the mask on a picture `python3 createmaskedmodel.py`
2. Execute the script to access the webcam and evaluate the webcam flow `python3 maskedcam.py`