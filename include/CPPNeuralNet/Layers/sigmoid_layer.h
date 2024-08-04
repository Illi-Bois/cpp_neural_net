#ifndef CPP_NN_LAYER_SIGMOID
#define CPP_NN_LAYER_SIGMOID

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/layer.h"
#include <cmath>

namespace cpp_nn {

class Sigmoid : public Layer {
private:
  typedef util::Tensor<float> Tensor;

  util::Tensor<float> last_output_;

public:
  Tensor forward(const Tensor& input)     override;
  Tensor backward(const Tensor& gradient) override;
};

/**
 *  Background.
 * 
 *  Sigmoid is motivated by trying to 'normalize' or contain
 *    the possibly infinite input within 0 and 1.
 *  Sigmoid is element-wise function defined
 *    f(x) = 1 / (1 + e^-x)
 *  This is chosen as its derivitive is simply defined in terms of itself:
 *   f'(x) = f(x) * (1 - f(x))
 *  
 *  with y = L(x) = f(x)
 *      dy/dx = y * (1 - y)
 *  so element-wise
 *      dC/dx = dC/dy * y * (1 - y)
 *  
 *  Note that unlike ReLU the last Output needs to be stored
 *    for fast gradient comptation, rather than last input.
 */
} // namespace cpp_nn

#endif // CPP_NN_LAYER_SIGMOID