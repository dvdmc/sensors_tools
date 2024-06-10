# Sensors Tools

This repository offers a generic sensor interface for RGB-D semantic ROS messages.
It mainly tackles the problem of bridging semantic measurements from simulators, datasets, other ROS frameworks.

<p float="center" align="middle">
  <img src="media/rgb.png" width="20%" hspace="20"/>
  <img src="media/depth.png" width="20%" hspace="20"/> 
  <img src="media/semantic.png" width="20%" hspace="20"/>
</p>

## Installation

Clone the repository with:

```
git clone https://github.com/dvdmc/sensors_tools
```

Then, you can install the required dependencies and the python package with:

```
pip install . (use -e for installing an editable version in case you want to modify / debug)
```

## Structure
- `sensors_tools/`:
  - `base/`: general classes.
  - `utils/`: general functions or tool classes.
  - `bridges/`: classes to interface with simulators, datasets, ROS...
  - `inference/`: module to store inference models (currently mainly aimed at semantics), loading them for inference with a general interface, and performing inference.
  - `sensor.py`: a generic sensor that loads: a bridge as a data interface, (optionally) an inference module to obtain data from Neural Networks.
- `sensor_tools_ros/`:
  -  `semantic_ros.py`: main node for using the `sensor.py` within ROS.

## Configuration

For the configuration, this repository follows the approach of keeping `{Class}Config` dataclasses that are used for typing and defaults. 
The config objects are used to configure an instantiated class. In ROS, this configuration is input using a `.yaml` file (check `semantic_ros.py`).

## Usage

The intended usage is by running the `sensor` node with `rosrun` or using a launch file:

```
roslaunch sensors_tools_ros semantic_sensor.launch
```
