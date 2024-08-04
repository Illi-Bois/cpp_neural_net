#include "CPPNeuralNet/Layers/relu_layer.h"
#include <cstdlib>
#include <algorithm>

namespace cpp_nn {


/**
 * Relu function is defined as : f(x) = max(0, x)
 * so its if x > 0, f(x) = x else f(x) = 0
 * Relu basically "activaes" only for positive inputs
 */

util::Tensor<float> ReLU::forward(const util::Tensor<float>& input) {
  last_input_ = input;
  return Tensor(last_input_.getShape(),
                [last_input_iter = last_input_.begin()] () mutable {
                  float res = (*last_input_iter > 0) ? *last_input_iter : 0;
                  ++last_input_iter;
                  return res;
                });
}
/**
 * derivative of ReLU is f'(x) = 1 if x > 0 else f'(x) = 0 
 * If x > 0, dL/dx = dL/dy * 1 = dL/dy 
 * else , dL/dx = dL/dy * 0 = 0
 */
ReLU::Tensor ReLU::backward(const util::Tensor<float>& gradient) {
  return Tensor(gradient.getShape(),
                [last_input_iter = last_input_.begin(),
                 gradient_iter   = gradient.begin()] () mutable {
                  float res = (*last_input_iter > 0) ? *gradient_iter : 0;
                  ++last_input_iter;
                  ++gradient_iter;
                  return res;
                });
}


} // namespace cpp_nn