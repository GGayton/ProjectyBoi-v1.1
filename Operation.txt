1. Turn on camera/projector
2. Make sure the camera is plugged in
3. Set up projector
4. Run projecty boi

Set up projector:

1. Turn on projector GUI (DLPLCR4500 GUI 4.0.0)
2. set to "pattern mode"
3. Set pattern mode to "white"
3. Set input to vync/video port
4. Set pattern exposure (9000)
5. Set pattern bit depth to 8 bit
6. Add 8 bit green to pattern
7. Send
8. Validate
9. Play

First time set up:
1. Make sure environment is ProjectyBoi (should say bottom right - "conda: ProjectyBoi (Python 3.9.2))
2. Otherwise click that, change environment to "C:/ProgramData/Anaconda3/envs/ProjectyBoi/python.exe"

Projectyboi2001:

1. Load images from drop down list (3-step XY Modified Heterodyne) (i have included other regimes in extras - these will not appear unless you put them in \\Projection Regimes)
2. Set repeats (sets number of repeated camera images for each projected image) - for official measurements recommend at least 3 repeats
3. Click Measure
4. Close in between measurements and reopen everything (I havent double checked behaviour for repeating measurements in one session)
5. Sometimes camera returns corrupted measurements: check first measurement if about to do many repeated measurements. Check measurements using showMeasurements.py

For aligning/checking/optimising measurements:
1. Project 0th image (which should be blank) (after completing (1) from list above)
2. click "start streaming"
3. click capture image

#### SETTING UP ENV ####

This is only if the environment is not available and you need to build it

If using conda:
1. Change directory to directory holdnig ProjectyBoiEnv.txt
2. conda create --name <env> --file ProjectyBoiEnv.txt

vimba must be set up manually
1. in anaconda prompt, activate environment
2. cd <directory of \VimbaPython-master>
3.run "python -m pip install .[numpy-export,opencv-export]"