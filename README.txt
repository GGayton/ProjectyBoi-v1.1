### PROJECTYBOI 1.1 ###

OVERVIEW:
Currently, this holds two sections: measurement and conversion.

MEASUREMENT:
The measurement folder holds the modules and scripts required to run the fringe projection system.
Particularly, \measurement\ProjectyBoi2001.py is the script that initialises and runs all the code necessary, i.e.
run ProjectyBoi2001.py to begin taking measurements.

In this readme - macrosync means taking a picture of the correct projection image, microsync is the synchronisation
of camera exposure time and projector projection time.

The measurement modules and their functions are:
 - camera              | control the camera, macrosync
 - projector           | feed images to projector, macrosync
 - saver               | save images, check for image failures
 - streamer            | seperate sub-window to view and check images (for aligning objects, checking over exposure etc.)
 - measurement         | macrosync all components together
 - console_outputs     | loading bar
 - common_functions    | common functions 
 - common_qt_functions | numpy->QT array conversion 





CONVERSION:
The conversion folder holds the modules and scripts that converts the images captured during the measurement to a 
point-cloud. The script for this is \conversion\convertToPointCloud.py. Currently, the parameters that define the
fringe projection system are held in parameters \conversion\parameters\modelParameters.hdf5. These are found during
the calibration and the calibration code is not yet added.