#coding=utf-8


import tfcoreml as tf_converter

## Resolution has to be specified for the conversion to coreML
dim_x, dim_y = 320, 320

tf_converter.convert(tf_model_path="models/graph/mobilenet_thin/graph_freeze.pb",
                     mlmodel_path="models/graph/mobilenet_thin/graph" + str(dim_x
                     ) + "x" + str(dim_y) + ".mlmodel",
                     output_feature_names=["Openpose/concat_stage7:0"],
                     image_input_names="image:0",
                     input_name_shape_dict={"image:0": [1, dim_x, dim_y, 3]})
