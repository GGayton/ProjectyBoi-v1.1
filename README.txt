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
The measurement is completed entirely using the QT library, which allows both multithreading, pixel writing and UI.

\measurement\ProjectyBoi2001.py 
	initialises and runs all the code necessary, i.e. run ProjectyBoi2001.py 
	to begin taking measurements.

\measurement\show_all_measurements.py 
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

The entire measurement works using QSemaphore to track when to project an image and when to acquire one, and using queue, 
which is a first in first out queue to pass information between threads.

Each *******_cond object is a QSemaphore object which works using .acquire(), which acquires on token
from the object. If a token is not there to be taken, then the thread pauses until there is a token
available.

The behaviour of the UI in taking a measurement immediately after a measurement is not 100%
tested. For critical measurements, would suggest closing projecyboi and rerunning the script.
===============================================================
CONVERSION:
===============================================================
The conversion folder holds the modules and scripts that converts the images captured during the measurement to a 
point-cloud. The script for this is \conversion\convert_to_point_cloud.py. Currently, the parameters that define the
fringe projection system are held in parameters \conversion\parameters\model_parameters.hdf5. These are found during
the calibration (given in \calibration folder).

The conversion modules and their functions are:
 - decoding               | decode the images to give camera->projector correspondence
 - input_measurement      | class to hold the decoded images/perform some filtering
 - input_parameter        | class to hold the parameters of the system
 - model                  | object that performs the decoded image-> point-cloud conversion
 - nonlinear_correction   | performs the distortion correction

===============================================================
CALIBRATION:
===============================================================
The calibration is compeleted using tensorflow, which serves to compute the gradient.

The calibration modules and their functions are:
 - analytical_calib       | Analytical calibration used to produce an estimate for the nonlinear regression
 - base_calib             | class to common calibration functions
 - input_data             | class to hold the input data to a calibration - reads the hdf5 file holding the data and
                            prepares data for calibration
 - serial                 | object that performs the decoded image-> point-cloud conversion

The calibration supports arbitrary numbers of camera/projectors/artefacts, with differing points per pose if needed.
A pose is defined as a single artefact measured once by any number of cameras/projectors. I.e., a pose
shares a single rotation and translation estimation of the artefact. Iif you measure 2 dot grids in one image,
that would equate to 2 poses, since they wouldn't share a rotation and translation.

First, the inputs to the calibration must be known. \dot localisation houses a method for completing
this on a dot grid. Other methods for differing artefact are not written. Any feature localisation
method must produce a set of 2D image points, that match the input hdf5 file hierarchy:

(a {} implies a user designated phrase, otherwise the key must match, i.e. the first level
of the .hdf5 file must be "inputs", which allows other data to be stored on the file)

level | key
0     | ->inputs
1     |   ->{2 digit num} (indicate what pose it is)
2     |     ->{keyword} (single seperate unique keyword for each component, a component is defined as an object that
      |          shares distortion/camera matrix/artefcat rotation/artefact translation parameters)
3     |       ->points (The 2D measured points)
3     |       ->artefact (The 3D artefact points)

keywords must remain the same throughout dataset
example
"inputs\04\left camera\points"
would access the 5th pose (index from 0) measured 2d points on the component labelled "left camera".

"inputs\03\upside-down rubbish projector\artefact"
would access the 4th pose 3d artefact points on the component labelled "upside-down rubbish projector".

Theoretically, you could label poses as you like, but I suggest using double digits 00 upwards, anything else
could mix up poses and their estimations.

The Levenberg-Marquardt algorithm (serial calibration) requires an estimation, which is provided using
calibration\analytical_calibrate.py. The rotation/translation between cameras/projectors can be 
found using the individual estimation of the artefact in each common pose. As of yet, the parallel
method and uncertainties are not yet implemented.

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

When changing the camera or projector and the connecion between them must be changed, ensure to 
GROUND the signal cable - there is a lot of EM interference in the lab.

===============================================================
KNOWN ISSUES/MISSING FUNCTIONALITY
===============================================================
CAMERA IMAGE     |The camera is known to occasionally return either a blank image - or a repeated image.
                 |The error seems to happen between 1 in 100 - 1 in 200 images taken. These measurements
                 |are not recoverable and will have to be taken again. It's believed to be a camera issue - 
                 |after taking 1000s of images - the error rate will increase dramatically. This may be improved
_________________|by repairing the vimba install.
CAMERA IMAGE     |The camera image error described above can potentially be circumvented when taking multiple 
ERROR HANDLING   |images per projector image by check for discrepancies between each repeated camera image.
_________________|
IMPERFECT        |The camera-projector synchronisation could be improved - it is synchronised as much as is 
SYNCHRONISATION  |possible. To improve the synchronisation, a complete system rehaul would be required since the
_________________|HDMI signal is driving these issues.
NO EXCEPTION     |Any exceptions will lead to the screen hanging and requiring a restart of the 
HANDLING         |kernel.
_________________|
NO PHASE MAP     |Currently, no filtering is applied to the phase map, which would greatly improve the 
FILTERING        |rate at detecting phase unwrapping errors. The calculation is essentially
_________________|completed but no filtering is being done.
NO 3D POSITION   |Checking the feinal 3D points would be a cheap method to clear out any unwrapping
FILTERING        |error outliers.
_________________|
PROJECTOR        |There is a way to automatically set projector options, there may even already be a library
OPTIONS          |available to handle this. Otherwise, the commands can be found in the proejctor
_________________|handbook.
UNCHECKED REPEAT |The repeat measurement section is not check - and may not work completely well. This function
MEASUREMENT FCN  |should be checked before conducting any long-lengt measurements.
_________________|
CUDA             |CUDA implementation in the conversion (replacing numpy with cupy) will massively speed up
IMPLEMENTATION   |measurements - but requires a GPU with ~>4GB of memory.
_________________|
CAMERA TEMP      |Camera has a built-in temperature sensor that should be utilised.
SENSOR           |
_________________|
GENERIC          |The conversion is not currently generic to multiple cameras and projectors, while the calibration
CONVERSION		 |is. This is missing functionality - which will require overhauling the saver class too, so that
				 |images are saved to their respective components, instead of just saving the single image.