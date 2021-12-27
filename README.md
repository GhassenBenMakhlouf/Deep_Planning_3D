# Deep_Planning_3D

### Data Generation
* Install Matlab
* Install comsol with the option livelink for Matlab
* In Windows run "COMSOL Multiphysics with MATLAB "
  In Mac OS X run "COMSOL with MATLAB"
  In Linux enter the command "comsol mphserver matlab" in teminal
* Add the function cubeint, using the " Add-On Explorer" in Matlab. It will be needed in the function "generate_valid_config_3d.m"
* in the Matlab script "generate_dataset_3d.m":
  - edit line 5 to set the path where the Comsol Mph file will be saved. This file can be used to open the model with "Comsol Multiphysics".
  - edit line 6 to set the path where the simulated data will be saved.
  - edit line 7 to set the wished number of the generated data. 
* Run the Matlab script "generate_dataset_3d.m"

### Preprocess Data
The Python script "preprocess_data.py" transforms the simulated data from Comsol to Numpy Matrices
Each output matrix is a 101x101x101x4 Matrix (4 channels), where:
* channel 1 --> data distance to the goal
* channel 2 --> data conductivity
* channel 3 --> data labels: angles between x and y axes of the ground truth EMF gradients
* channel 4 --> data labels: angles between the xy plan and z axis of the ground truth EMF gradients
Use lines 12, 13 and 14 to set the paths of the Comsol output data for train, val and test.
Use lines 25, 27 and 29 to set the conductivities of the environment, obstacles and goal respectively.
PS: The Python script "preprocess_data.py" was written in Tensorflow 1.15 and not yet updated to 2.7.

### Train the 3D CNN
* Use requirements.txt to install the needed packages to use the "train.py" Python script
* Use lines 20, 21 and 22 to set the paths of the numpy data for train, val and test.
* Use line 23 to set the batch size
* Use line 24 to set the maximum number of epochs. The script uses an early stopping with patience 100 epochs.
* The model checkpoints will be saved in folder "checkpoints"
* The training logs can be displayed using Tensorboard and the folder "logs/fit"
* The Python script "cnn_3d.py" defines the architecture of the used 3D CNN. 

