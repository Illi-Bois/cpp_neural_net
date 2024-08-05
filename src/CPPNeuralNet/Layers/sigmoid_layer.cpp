#include "CPPNeuralNet/Layers/sigmoid_layer.h"

namespace cpp_nn {
// Constructor -----------------------------------------
Sigmoid::Sigmoid()
    : last_output_({1}) /* default smallest tensor */ 
  {}
// End of Constructor ----------------------------------


Sigmoid::Tensor Sigmoid::forward(const Tensor& input) {
  last_output_ = util::Apply<float,
                             Tensor>(input,
                                     *util::SigmoidFucntion);
  return last_output_;
}

Sigmoid::Tensor Sigmoid::backward(const Tensor& gradient) {
  return util::Apply<float,
                     Tensor,
                     Tensor>(gradient,
                             last_output_,
                             [] (float grad, float sigmoid)->float {
                               return grad * sigmoid * (1 - sigmoid);
                             });
}

} // namespace cpp_nn