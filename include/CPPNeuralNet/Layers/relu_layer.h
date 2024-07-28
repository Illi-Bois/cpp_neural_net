#ifndef CPP_NN_LAYER_RELU
#define CPP_NN_LAYER_RELU

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/layer.h"

namespace cpp_nn {

class ReLU : public Layer {
private:
  util::Tensor<float> last_input;

public:
  ReLU() = default;
  ~ReLU() override = default;

  util::Tensor<float> forward(const util::Tensor<float>& input) override;
  util::Tensor<float> backward(const util::Tensor<float>& gradient) override;
  void update_parameters(float learning_rate) override;
  void save_gradients() override;
  void clear_gradients() override;
};

} // cpp_nn

#endif // CPP_NN_LAYER_RELU