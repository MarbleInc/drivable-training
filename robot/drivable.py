# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
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
from tensorflow.contrib import predictor

CSV_COLUMNS = [
  "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11", "h12", "label"
]

# Continuous base columns.
h0 = tf.feature_column.numeric_column("h0")
h1 = tf.feature_column.numeric_column("h1")
h2 = tf.feature_column.numeric_column("h2")
h3 = tf.feature_column.numeric_column("h3")
h4 = tf.feature_column.numeric_column("h4")
h5 = tf.feature_column.numeric_column("h5")
h6 = tf.feature_column.numeric_column("h6")
h7 = tf.feature_column.numeric_column("h7")
h8 = tf.feature_column.numeric_column("h8")
h9 = tf.feature_column.numeric_column("h9")
h10 = tf.feature_column.numeric_column("h10")
h11 = tf.feature_column.numeric_column("h11")
h12 = tf.feature_column.numeric_column("h12")

# Wide columns and deep columns.
base_columns = [
    h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12
]

def build_estimator(model_dir, model_type):
  """Build an estimator."""
  if model_type == "wide":
    m = tf.estimator.DNNClassifier(feature_columns=base_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=2,
                                          model_dir=model_dir)
  return m

def serving_input_receiver_fn():
  """Build the serving inputs."""
  # The outer dimension (None) allows us to batch up inputs for
  # efficiency. However, it also means that if we want a prediction
  # for a single instance, we'll need to wrap it in an outer list.
  #inputs = {"x": tf.placeholder(shape=[None, 3], dtype=tf.float32)}
  # inputs = base_columns
  inputs = {
            "h0": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h1": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h2": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h3": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h4": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h5": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h6": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h7": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h8": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h9": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h10": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h11": tf.placeholder(shape=[None, 1], dtype=tf.float32),
            "h12": tf.placeholder(shape=[None, 1], dtype=tf.float32),
  }
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def input_fn(data_file, num_epochs, shuffle, num_threads=5):
  """Input builder function."""
  print(data_file)
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      sep=' ',
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0).astype(float)
  print(df_data)
  labels = df_data["label"].astype(int)
  print ("len df_data %d" % len(df_data))
  l_mat =labels.as_matrix()
  print(l_mat)
  print(np.bincount(l_mat))
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      #batch_size=100,
      batch_size=20,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=num_threads)

def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  if train_data:
    train_file_name = train_data
  if test_data:
    test_file_name = test_data

  # Specify file path below if want to find the output easily
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir

  m = build_estimator(model_dir, model_type)
  # set num_epochs to None to get infinite stream of data.
  m.train(
      input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
  #   input_fn=lambda: my_input_fn(train_file_name, perform_shuffle=True),
      steps=train_steps)

  # Evaluation
  # set steps to None to run evaluation until all data consumed.
  # results = m.evaluate(
  #     input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
  #     steps=None)
  # print("model directory = %s" % model_dir)
  # for key in sorted(results):
  #   print("%s: %s" % (key, results[key]))
  # Export
  export_dir = m.export_savedmodel(
    export_dir_base="/home/ammar/data/tflow/drivable/robot/models/",
    serving_input_receiver_fn=serving_input_receiver_fn,
    as_text=True)
  # Predict
  # predict_fn = predictor.from_saved_model("/home/ammar/data/tflow/drivable/robot/models/1512612563/", signature_def_key='predict')

  # i_data, i_labels = lambda: input_fn(test_file_name, num_epochs=1, shuffle=False, num_threads=1)

  #predcitions_2 = predict_fn({"h0": h0})
  predictions = list(m.predict(input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False, num_threads=1)))
  #print (predictions)
  for i, p in enumerate(predictions):
    print("Prediction %s: %s %s" % (i, p["classes"], p["probabilities"]))

  # predictions = list(classifier.predict(input_fn=predict_input_fn))
  # predicted_classes = [p["classes"] for p in predictions]
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
      default="/home/ammar/data/tflow/drivable/robot/driv5_z.train",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="/home/ammar/data/tflow/drivable/robot/driv5_z.test",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
