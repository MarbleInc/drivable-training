#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include <chrono>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tf = tensorflow;

int main(int argc, char **argv) {
  const std::string export_dir = argv[1];

  tf::SavedModelBundle bundle;
  std::unordered_set<std::string> tmp_set({"serve"});
  tf::Status load_status = tf::LoadSavedModel(
      tf::SessionOptions(), tf::RunOptions(), export_dir, tmp_set, &bundle);
  if (!load_status.ok()) {
    std::cout << "Error loading model: " << load_status << std::endl;
    return -1;
  }
  std::cout << "model loaded" << std::endl;

  std::vector<std::vector<float>> feat =
    {{0.301587, 0, 0, 0.174603, 0, 0, 0.52381, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.992063, 0.00793651, 0, 0, 0.166667, 0.166667, 0.468254, 0.198413, 0, 0, 0, 0.0206936, -6.75091e-05, 0.00951792, 0.0111757, 1},
{ 0.81746, 0, 0, 0.174603, 0, 0, 0.00793651, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0238095, 0.952381, 0.0238095, 0, 0, 0.166667, 0.174603, 0.47619, 0.18254, 0, 0, 0, 0.0206936, -0.00556574, 0.0147159, 0.00597775, 1}};/*,,
{ 0.685039, 0, 0, 0.173228, 0, 0, 0.141732, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.00787402, 0.984252, 0.00787402, 0, 0, 0.165354, 0.173228, 0.456693, 0.204724, 0, 0, 0, 0.0206936, -0.00292745, 0.0119281, 0.00876555, 1},
{ 0.126984, 0, 0, 0.174603, 0, 0, 0.698413, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.00793651, 0.968254, 0.0238095, 0, 0, 0.166667, 0.166667, 0.452381, 0.214286, 0, 0, 0, 0.0206936, 0.00135241, 0.00755753, 0.0131361, 1},
{ 0.357143, 0, 0, 0.174603, 0, 0, 0.468254, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.015873, 0.97619, 0.00793651, 0, 0, 0.166667, 0.174603, 0.460317, 0.198413, 0, 0, 0, 0.0206936, -0.000737572, 0.00950269, 0.0111909, 1},
{ 0.456693, 0, 0, 0.181102, 0, 0, 0.362205, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.015748, 0.984252, 0, 0, 0, 0.165354, 0.165354, 0.464567, 0.204724, 0, 0, 0, 0.0225702, -0.00140866, 0.0117968, 0.0107733, 1},
{ 0.787402, 0, 0, 0.19685, 0, 0, 0.015748, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0314961, 0.952756, 0.015748, 0, 0, 0.165354, 0.173228, 0.456693, 0.204724, 0, 0, 0, 0.0234003, -0.00527689, 0.0161268, 0.00727346, 1},
{ 0.0952381, 0, 0, 0.190476, 0, 0, 0.714286, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.00793651, 0.984127, 0.00793651, 0, 0, 0.166667, 0.166667, 0.436508, 0.230159, 0, 0, 0, 0.0228811, 0.00225034, 0.008074, 0.0148071, 1},
{ 0.119048, 0, 0, 0.198413, 0, 0, 0.68254, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.00793651, 0.984127, 0.00793651, 0, 0, 0.166667, 0.166667, 0.436508, 0.230159, 0, 0, 0, 0.0234003, 0.00177922, 0.00886729, 0.014533, 1},
{ 0.269841, 0, 0, 0.206349, 0, 0, 0.52381, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.00793651, 0.984127, 0.00793651, 0, 0, 0.166667, 0.166667, 0.452381, 0.214286, 0, 0, 0, 0.0223577, -0.000362369, 0.0109138, 0.0114439, 1},
{ 0.344, 0, 0, 0.216, 0, 0, 0.44, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.008, 0.992, 0, 0, 0, 0.168, 0.168, 0.456, 0.208, 0, 0, 0, 0.0225165, -0.000915185, 0.0113899, 0.0111265, 1},
{ 0.634921, 0, 0, 0.230159, 0, 0, 0.134921, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.00793651, 0.984127, 0.00793651, 0, 0, 0.166667, 0.166667, 0.452381, 0.214286, 0, 0, 0, 0.0253732, -0.00321626, 0.0161867, 0.00918655, 1},
{0.504, 0, 0, 0.232, 0, 0, 0.264, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.992, 0.008, 0, 0, 0.168, 0.176, 0.44, 0.216, 0, 0, 0, 0.0253732, -0.00208295, 0.0149195, 0.0104537, 1},
{ 0.145161, 0, 0, 0.233871, 0, 0, 0.620968, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.975806, 0.0241935, 0, 0, 0.169355, 0.169355, 0.443548, 0.217742, 0, 0, 0, 0.0253732, 0.000601533, 0.0120838, 0.0132895, 1}};*/

    // {
    //   0.016129,  0,         0.290323,  0.596774,  0.0967742, 0,
    //   0,         0.0483871, 0.0645161, 0.016129,  0.548387,  0.209677,
    //   0.0322581, 0.0806452, 0.0322581, 0,         0.016129,  0.548387,
    //   0.016129,  0.193548,  0.193548,  0.0967742, 0.0806452, 0.0645161,
    //   0.0806452, 0.225806,  0.16129,   0.290323,  0.422826,  -0.00891484,
    //   0.285487,  0.137339};
    size_t num_feat = 29;//feat.size();
  size_t inp_size = feat.size();
  auto start = std::chrono::system_clock::now();

  // We should get the signature out of MetaGraphDef, but that's a bit
  // involved. We'll take a shortcut like we did in the Java example.
  const std::string scores_name = "dnn/head/predictions/probabilities:0";
  std::vector<std::pair<std::string, tf::Tensor>> inputs;
  inputs.reserve(num_feat);
  for (size_t i = 0; i < num_feat; ++i) {
    auto t = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({inp_size, 1}));
    auto matrix = t.matrix<float>();
    for (size_t j = 0; j < inp_size; ++j)
      matrix(j, 0) = feat[j][i];
    std::string placeholder;
    if (i == 0)
      placeholder = "Placeholder:0";
    else
      placeholder = "Placeholder_" + std::to_string(i) + ":0";
    inputs.push_back({placeholder, t});
  }

  auto end = std::chrono::system_clock::now();

  std::cout << "Tensor prep: "
            << std::chrono::duration<double>(end - start).count() << std::endl;

  start = std::chrono::system_clock::now();
  std::vector<tf::Tensor> outputs;
  tf::Status run_status =
      bundle.session->Run(inputs, {scores_name}, {}, &outputs);
  if (!run_status.ok()) {
    std::cout << "Error running session: " << run_status << std::endl;
    return -1;
  }
  end = std::chrono::system_clock::now();
  std::cout << "Inference: "
            << std::chrono::duration<double>(end - start).count() << std::endl;

  for (const auto &tensor : outputs) {
    std::cout << "ne " << tensor.shape().num_elements() << std::endl;
    std::cout << tensor.matrix<float>() << std::endl;
  }
}
