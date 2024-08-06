#include "CPPNeuralNet/Layers/relu_layer.h"
#include <cstdlib>
#include <algorithm>

namespace cpp_nn {
// Constructor -----------------------------------------
ReLU::ReLU()
    : last_input_({1}) /* default smallest tensor */ 
  {}
// End of Constructor ----------------------------------

/**
 * Relu function is defined as : f(x) = max(0, x)
 * so its if x > 0, f(x) = x else f(x) = 0
 * Relu basically "activaes" only for positive inputs
 */
ReLU::Tensor ReLU::forward(const Tensor& input) {
  last_input_ = input;
  return util::Apply<float, 
                     Tensor>(input,  
                             [] (float input)->float {
                               return input > 0 ? input : 0;
                             });
}
/**
 * derivative of ReLU is f'(x) = 1 if x > 0 else f'(x) = 0 
 * If x > 0, dL/dx = dL/dy * 1 = dL/dy 
 * else , dL/dx = dL/dy * 0 = 0
 */
ReLU::Tensor ReLU::backward(const Tensor& gradient) {
  return util::Apply<float, 
                     Tensor, 
                     Tensor>(last_input_,
                             gradient,
                             [] (float input, float gradient)->float {
                               return ((input > 0) ? gradient : 0);
                             });
}


} // namespace cpp_nn