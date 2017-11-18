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


CSV_COLUMNS = [
  "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11", "h12", "h13", "h14", "h15", "h16", "h17","h18", "h19", "h20", "h21", "h22", "h23", "h24", "h25", "h26", "h27", "h28", "h29", "h30", "h31", "h32", "label"
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
h13 = tf.feature_column.numeric_column("h13")
h14 = tf.feature_column.numeric_column("h14")
h15 = tf.feature_column.numeric_column("h15")
h16 = tf.feature_column.numeric_column("h16")
h17 = tf.feature_column.numeric_column("h17")
h18 = tf.feature_column.numeric_column("h18")
h19 = tf.feature_column.numeric_column("h19")
h20 = tf.feature_column.numeric_column("h20")
h21 = tf.feature_column.numeric_column("h21")
h22 = tf.feature_column.numeric_column("h22")
h23 = tf.feature_column.numeric_column("h23")
h24 = tf.feature_column.numeric_column("h24")
h25 = tf.feature_column.numeric_column("h25")
h26 = tf.feature_column.numeric_column("h26")
h27 = tf.feature_column.numeric_column("h27")
h28 = tf.feature_column.numeric_column("h28")
h29 = tf.feature_column.numeric_column("h29")
h30 = tf.feature_column.numeric_column("h30")
h31 = tf.feature_column.numeric_column("h31")
h32 = tf.feature_column.numeric_column("h32")

# Wide columns and deep columns.
base_columns = [
   h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17,h18, h19, h20, h21, h22, h23, h24, h25, h26, h27,
  h28, h29, h30, h31, #h32,
]

def build_estimator(model_dir, model_type):
  """Build an estimator."""
  if model_type == "wide":
    # optimizer=tf.train.FtrlOptimizer(
    #   learning_rate=5.0,
    #   l1_regularization_strength=1.0,
    #   l2_regularization_strength=1.0)
    # kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
    #   input_dim=24, output_dim=200, stddev=5.0, name='rffm')
    # kernel_mappers = {base_columns: [kernel_mapper]}
    # m = tf.contrib.kernel_methods.KernelLinearClassifier(
    #   n_classes=2, optimizer=optimizer, kernel_mappers=kernel_mappers)
    # m = tf.estimator.LinearClassifier(
    #   model_dir=model_dir, feature_columns=base_columns, optimizer=optimizer, n_classes=2)
    # m = tf.estimator.LinearClassifier(
    #   model_dir=model_dir, feature_columns=base_columns, n_classes=2)
    m = tf.estimator.DNNClassifier(feature_columns=base_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir=model_dir)
  return m


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

  df_data = df_data.dropna(how="any", axis=0)
  # convert label
  def conv_label(x):
    if (x =='l'):
      return 1
    elif (x == 'w'):
      return 2
    else:
      return 0

  #labels = df_data["label"].apply(lambda x: "l" in x).astype(int)
  labels = df_data["label"].apply(lambda x: conv_label(x)).astype(int)
  print ("len df_data %d" % len(df_data))
  print(labels)
  #my_dict = dict((i, labels.count(i)) for i in labels)#{i:labels.count(i) for i in labels}
  #print(my_dict)
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
      steps=train_steps)
  # set steps to None to run evaluation until all data consumed.
  # results = m.evaluate(
  #     input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
  #     steps=None)
  # print("model directory = %s" % model_dir)
  # for key in sorted(results):
  #   print("%s: %s" % (key, results[key]))

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
      default="/home/ammar/data/tflow/drivable/driv3.train",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="/home/ammar/data/tflow/drivable/driv3.test",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
