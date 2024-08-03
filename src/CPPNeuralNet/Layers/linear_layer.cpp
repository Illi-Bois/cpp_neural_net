#include "CPPNeuralNet/Layers/linear_layer.h"
#include <cstdlib>

namespace cpp_nn {

LinearLayer::LinearLayer(int input_size, int output_size)
    : weights_({input_size, output_size}), // TODO: Initializer Generator as random?
      biases_({output_size}),
      last_input_({1}) /*TODO: maybe a better place holder exists? maybe even make it a pointer?*/ {
  // TODO: Initialzation shold be put in as Generator
}

LinearLayer::Tensor LinearLayer::forward(const LinearLayer::Tensor& input) {
    last_input_ = input;
    // With broadcasted Multiplication no need to reshape
    return biases_ + weights_ * input;
}

util::Tensor<float> LinearLayer::backward(const util::Tensor<float>& gradient) {
  // TODO: AVERGAE NEEDS TO BE IMPLEMENTED
  // Biase Update
  biases_ = biases_ - gradient.AverageVectors(); // TODO NEED TO IMPLEMENT AVEAGE

  // Weight Update
  weights_ = weights_ - (gradient.VectorAsMatrix() * last_input_.VectorTranspose()).AverageMatrices();
  
  // pass gradient
  // TODO: should the gradient be relative to new or old layer? right now, using new Weights
  return weights_ * gradient;
}

} // namespace cpp_nn

