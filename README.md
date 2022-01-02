# Deep_Planning_3D

Use requirements.txt to install the needed packages (Python 3.9 was used)

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
The Python function "preprocess_data" transforms the simulated data from Comsol to Numpy Matrices
Each output matrix is a 101x101x101x2 Matrix (2 channels), where:
* channel 1 --> data conductivity
* channel 2 --> data labels: the ground truth EMF magnitudes

Use config.yaml("comsol_data" dictionary) to set the paths of the Comsol output data for train, val and test.

Use config.yaml("comsol_data" dictionary) also to set the conductivities of the environment, obstacles and goal respectively.

### Train the 3D CNN
* Use config.yaml("learning" dictionary) to set the paths of the numpy data for train, val and test and the batch size.
* Use config.yaml("learning" dictionary) also to set the maximum number of epochs. The script uses an early stopping with patience 100 epochs.
* The model checkpoints will be saved in folder "checkpoints"
* The training logs can be displayed using Tensorboard and the folder "logs/fit"
* The Python script "cnn_3d.py" defines the architecture of the used 3D CNN. It provides two functions 
"build_cnn_3d" and "build_cnn_3d_cheap", which is the same architecture with cheaper computation. 
Set the variable "cheap" in config.yaml("learning" dictionary) to True to use the cheaper network.  

