from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import tempfile

import pandas as pd
from six.moves import urllib
import tensorflow as tf
import numpy as np

CSV_COLUMNS = [
  "h0", "h1", "h2", "label"
]

# Continuous base columns.
h0 = tf.feature_column.numeric_column("h0")
h1 = tf.feature_column.numeric_column("h1")
h2 = tf.feature_column.numeric_column("h2")

base_columns = [h0, h1, h2]

def build_estimator(model_dir, model_type):
  """Build an estimator."""
  if model_type == "wide":
    m = tf.estimator.DNNClassifier(feature_columns=base_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=2,
                                          model_dir=model_dir)
  return m


def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
   def decode_csv(line):
       parsed_line = tf.decode_csv(line, [[0],[0],[0],[]], field_delim=' ', na_value=' ')
       label = parsed_line[-1:] # Last element is the label
       del parsed_line[-1] # Delete last element
       features = parsed_line # Everything (but last element) are the features
       d = dict(zip(CSV_COLUMNS, features)), label
       print(features)
       return d

   dataset = (tf.data.TextLineDataset(file_path) # Read text file
       .skip(1) # Skip header row
       .map(decode_csv)) # Transform each elem by applying decode_csv fn
   if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=256)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(32)  # Batch size to use
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels

def serving_input_receiver_fn():
  """Build the serving inputs."""
  # The outer dimension (None) allows us to batch up inputs for
  # efficiency. However, it also means that if we want a prediction
  # for a single instance, we'll need to wrap it in an outer list.
  #inputs = {"x": tf.placeholder(shape=[None, 3], dtype=tf.float32)}
  # inputs = base_columns
  inputs = {"h0": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h1": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h2": tf.placeholder(shape=[None, 1], dtype=tf.float32)}

  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  if train_data:
    train_file_name = train_data
  if test_data:
    test_file_name = test_data

  # Specify file path below if want to find the output easily
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir

  m = build_estimator(model_dir, model_type)
  m.train(
      input_fn=lambda: my_input_fn(train_file_name, perform_shuffle=True),
      steps=train_steps)

  # export_dir = m.export_savedmodel(
  #   export_dir_base="/home/ammar/data/tflow/drivable/est",
  #   serving_input_receiver_fn=serving_input_receiver_fn,
  #   as_text=True)


  predict_fn = tf.contrib.predictor.from_saved_model("/home/ammar/data/tflow/drivable/est/1511829651/", signature_def_key='predict')
  predictions = predict_fn({"h0" : np.array([[45],[67]]), "h1" : np.array([[2],[43]]), "h2" : np.array([[3],[1]])})
  #predictions = list(m.predict(input_fn=lambda: my_input_fn(test_file_name, perform_shuffle=False)))
  print(predictions)

  for i, p in enumerate(predictions):
    print(p)
    #print("Prediction %s: %s %s" % (i, p["classes"], p["probabilities"]))

  # Manual cleanup
  shutil.rmtree(model_dir)

  FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="~/data/tflow/",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=20,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="/home/ammar/data/tflow/drivable/tf_dnn/dnn_est.train",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="/home/ammar/data/tflow/drivable/tf_dnn/dnn_est.test",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
