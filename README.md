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
**If you want to use Trident:** Install mmcv 2.1.0 following their [installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This version is currently required by [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). The installation of [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/introduction.html) and [Pytorch](https://pytorch.org/get-started/locally/) depends on your CUDA version. It is recommended to install a torch version which allows to install pre-built mmcv binaries. You can check the compatibility in this [section](https://github.com/open-mmlab/mmsegmentation). Then install mmsegmentation and mmengine.

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

## Acknowledgements

**Trident** is based on the original [repository (commit 5803ae)](https://github.com/YuHengsss/Trident/commit/5803ae4b4e1251d782298a7a426707ae55360c9f). The configuration for the detected classes was changed to use the `sensor` config interface. The `dataset_type` changed to pascal_voc and cityscapes to align with our naming. `model_type` changed to `clip_model_type`.