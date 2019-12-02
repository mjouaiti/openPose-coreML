# openPose-coreML
Pose estimation for Mac using CoreML

Installation:
pip install tfcoreml
pip install coremltools
cd pafprocess && swig -python -c++ pafprocess.i && python2 setup.py build_ext --inplace

CoreML does not support variable resolution so you have to generate a new model if you want to change it. The conversion code is provided in tfcoreML.py, just change the dim_x and dim_y values.

Inference Time on a Radeon Pro Vega 20 4 GB:
320 x 320: 30 ms
640 x 640: 55 ms

We also provide some example code on how to use the model. 

The code handles multiple person detection thanks to pafprocess (https://github.com/ildoonet/tf-pose-estimation/).
