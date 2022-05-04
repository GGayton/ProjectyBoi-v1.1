### PROJECTYBOI 1.1 ###

===============================================================
OVERVIEW:
===============================================================
Currently, this holds three sections: measurement, calibration and conversion.
The file requirements.txt labels all necessary libraries required to run every script.
In this readme - macrosync means taking a picture of the correct projection image, microsync is the synchronisation
of camera exposure time and projector projection time. Macrosync is in code, microsync is from hardware.
The file Operation.txt holds the information needed to run this specific FP system.

Measurement takes the images and serves to sync/run the FP system. It returns only images.
Conversion converts the images to a point-cloud.
Calibration quantifies the parameters that define the system e.g. focal length etc.

===============================================================
MEASUREMENT:
===============================================================
The measurement folder holds the modules and scripts required to run the fringe projection system.
The measurement is completed entirelyu using the QT library, which allows both multithreading, pixel writing and UI.

\measurement\ProjectyBoi2001.py 
	initialises and runs all the code necessary, i.e. run ProjectyBoi2001.py 
	to begin taking measurements.

\measurement\ShowAllMeasurements.py 
	shows all the images taken during a particular measurement.

\measurement\projection regimes
	Holds all the series of images required to take a measurement. Each "regime" is a .hdf5 file, each image is
	labelled with a two-digit number, i.e., "00", "01", "02" ..., and will take a camera image (or multiple camera images
	if chosen) for each digit in that order. 

The measurement modules and their functions are:
 - camera              | control the camera, macrosync
 - projector           | feed images to projector, macrosync
 - saver               | save images, check for image failures
 - streamer            | seperate sub-window to view and check images (for aligning objects, checking over exposure etc.)
 - measurement         | macrosync all components together
 - console_outputs     | loading bar
 - common_functions    | common functions 
 - common_qt_functions | numpy->QT array conversion 

===============================================================
CONVERSION:
===============================================================
The conversion folder holds the modules and scripts that converts the images captured during the measurement to a 
point-cloud. The script for this is \conversion\convertToPointCloud.py. Currently, the parameters that define the
fringe projection system are held in parameters \conversion\parameters\modelParameters.hdf5. These are found during
the calibration and the calibration code is not yet functional.

===============================================================
CALIBRATION:
===============================================================
The calibration is compeleted using tensorflow, which serves to compute the gradient.

First, the inputs to the calibration must be known. \dot localisation houses a method for completing
this on a dot grid. Other methods not written.

The calibration supports arbitrary numbers of camera/projectors/artefacts, with differing points per pose if need.
A pose is defined as a single artefact measured once by any number of cameras/projectors. I.e., a pose
shares a single rotation and translation estimation of the artefact. Even if you measure 2 dot grids in one image,
that would equate to 2 poses, since they wouldn't share a rotation and translation.

The input hdf5 file hierarchy looks like:

0->inputs
1 ->{2 digit num} (indicate what pose it is)
2  ->{keyword} (single seperate unique keyword for each component, a component shares distortion/camera matrix parameters)
3   ->points (The 2D measured points)
3   ->artefact (The 3D artefact points)

keywords must remain the same throughout dataset
example
"inputs\04\left camera\points"
would access the 5th pose (index from 0) measured 2d points on the component labelled "left camera".

"inputs\03\upside-down rubbish projector\artefact"
would access the 4th pose 3d artefact points on the component labelled "upside-down rubbish projector".

The Levenberg-Marquardt algorithm (serial calibration) requires an estimation, which is provided using
calibration\analytical_calibrate.py.
===============================================================
HARDWARE CONSIDERATIONS
===============================================================
This section is for considering replacing hardware (camera, projector etc).

Currently, the projector and camera are microsynced using the HDMI signal output (~60hz). The signal is faster
than the projector allows - its not truly projecting at this speed. The camera is microsynced using an option that 
summarised by "Wait for signal and then take picture". Replacing hardware will likely require a new solution to this microsync
problem (good thing - this can be improved).

The camera is controlled using an API called vimba - replacing the camera will require an entirely new
"camera" class.

A new projector that is not sent images using HDMI (or another HD digital output) will require an 
entirely new "projector" class.