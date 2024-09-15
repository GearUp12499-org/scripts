#!/usr/bin/python3

# Packages
import os
import numpy

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startwith('2')

tf.get_logger().setlevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Global Constants
path_to_train = 'samples/train'
path_to_test = 'samples/test'
detection_elements = ['specimen_block_red','specimen_block_blue','specimen_block_yellow']

# Declare Training Dataset (what the model will train upon)
train_dataset = object_detector.DataLoader.from_pascal_voc(
	path_to_train,
	path_to_train,
	detection_elements
)

# Declare Testing Dataset (what the model will use to predict)
test_dataset = object_detector.DataLoader.from_pascal_voc(
	path_to_test,
	path_to_test,
	detection_elements
)

# Create Model via Model Maker (if model maker does not a good job, we will need to make our own :P )
# spec is set to 'efficient_lite0' in order to optimize performance on SDK when using TFLite Java
spec = model_spec.get('efficient_lite0')
model = object_detector.create(
	train_dataset,
	batch_size=4,
	train_whole_model=True,
	epochs=20,
	validation_data=test_dataset
)

# Evaluate the Model and Export it as a  .TFLite file for use on the SDK
model.evaluate(test_dataset)
model.export(export_dir='.', tflite_filename='specimens.tflite')

# End of Model Construction
