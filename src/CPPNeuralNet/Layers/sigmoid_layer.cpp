#include "CPPNeuralNet/Layers/sigmoid_layer.h"

namespace cpp_nn {

Sigmoid::Tensor Sigmoid::forward(const Tensor& input) {
  last_output_ = Tensor(input.getShape(),
                       [input_iter = input.begin()] () mutable {
                         ++input_iter;
                         return 1 / (1 + std::exp(-(*input_iter)));
                       });
  return last_output_;
}

Sigmoid::Tensor Sigmoid::backward(const Tensor& gradient) {
  return Tensor(gradient.getShape(),
                [last_output_iter = last_output_.begin(),
                 gradient_iter   = gradient.begin()] () mutable {
                  float sigmoid = *last_output_iter;
                  float res = *gradient_iter * sigmoid * (1 - sigmoid);
                  ++last_output_iter;
                  ++gradient_iter;
                  return res;
                });
}

} // namespace cpp_nn