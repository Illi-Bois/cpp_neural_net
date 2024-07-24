#ifndef CPP_NN_LAYER_LINEAR
#define CPP_NN_LAYER_LINEAR

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/layer.h"

namespace cpp_nn {


class LinearLayer : public Layer {
public:
  LinearLayer(int input_size, int output_size);
  util::Tensor<float> forward(const util::Tensor<float>& input) override;
  util::Tensor<float> backward(const util::Tensor<float>& gradient) override;
  void update_parameters(float learning_rate) override;
  void save_gradients() override;
  void clear_gradients() override;
private:
  util::Tensor<float> weights;
  util::Tensor<float> biases;
  util::Tensor<float> input;
  util::Tensor<float> output;
  util::Tensor<float> weight_gradients;
  util::Tensor<float> bias_gradients;
};


} // cpp_nn

#endif // CPP_NN_LAYER_LINEAR
