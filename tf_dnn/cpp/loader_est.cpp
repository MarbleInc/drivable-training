#include <unordered_set>
#include <utility>
#include <vector>
#include <string>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

int main(int argc, char** argv) {
  const std::string export_dir = argv[1];

  tf::SavedModelBundle bundle;
  std::unordered_set<std::string> tmp_set({"serve"});
  tf::Status load_status
  = tf::LoadSavedModel(
                       tf::SessionOptions(), tf::RunOptions(), export_dir, tmp_set, &bundle);
  if (!load_status.ok()) {
    std::cout << "Error loading model: " << load_status << std::endl;
    return -1;
  }

  // We should get the signature out of MetaGraphDef, but that's a bit
  // involved. We'll take a shortcut like we did in the Java example.
  const std::string scores_name = "dnn/head/predictions/probabilities:0";

  auto h0 = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({1, 1}));
  {
    auto matrix = h0.matrix<float>();
    matrix(0, 0) = 6.4;
  }

    auto h1 = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({1, 1}));
  {
    auto matrix = h1.matrix<float>();
    matrix(0, 0) = 4;
  }

    auto h2 = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({1, 1}));
  {
    auto matrix = h2.matrix<float>();
    matrix(0, 0) = 1;
  }

  std::vector<std::pair<std::string, tf::Tensor>> inputs = {{"Placeholder:0", h0}, {"Placeholder_1:0", h1}, {"Placeholder_2:0", h2}};
  std::vector<tf::Tensor> outputs;

  tf::Status run_status =
      bundle.session->Run(inputs, {scores_name}, {}, &outputs);
  if (!run_status.ok()) {
    std::cout << "Error running session: " << run_status << std::endl;
    return -1;
  }

  for (const auto& tensor : outputs) {
    std::cout << tensor.matrix<float>() << std::endl;
  }
}
