#include "CPPNeuralNet/Layers/linear_layer.h"

namespace cpp_nn {

/**
 * Used Box Muller to generate gaussian random numbers 
 * Z1 = sqrt( -2 * ln(U1)) * sin(2 * pi * U2)
 */
LinearLayer::LinearLayer(int input_size, int output_size)
    : weights_({output_size, input_size}, [input_size]() {
          float u1 = static_cast<float>(std::rand()) / RAND_MAX;
          float u2 = static_cast<float>(std::rand()) / RAND_MAX;
          return std::sqrt(-2.0f * std::log(u1)) * std::sin(2.0f * M_PI * u2) * std::sqrt(2.0f / input_size);

    }), // TODO: Initializer Generator as random?
      biases_({output_size}),
      last_input_({1}) /*TODO: maybe a better place holder exists? maybe even make it a pointer?*/ {
  // TODO: Initialzation shold be put in as Generator
}

LinearLayer::Tensor LinearLayer::forward(const LinearLayer::Tensor& input) {
    last_input_ = input;
    // With broadcasted Multiplication no need to reshape
    return biases_ + CollapseEnd(weights_ * AsVector(input));
}

util::Tensor<float> LinearLayer::backward(const util::Tensor<float>& gradient) {
  if (gradient.getOrder() == 1) {
    // no need to sum and average to update
    biases_ = biases_ - gradient;

    weights_ = weights_ - (AsVector(gradient) * TransposeAsVector(last_input_));
  } else {
    // then assumes order is 2, where first is associated with mult-array for batch
    biases_ = biases_ - Average(gradient,
                                0);

    weights_ = weights_ - Average(AsVector(gradient) * TransposeAsVector(last_input_),
                                  0);
  }

  // pass gradient
  // TODO: should the gradient be relative to new or old layer? right now, using new Weights
  return CollapseEnd(weights_.Transpose() * AsVector(gradient));
}

} // namespace cpp_nn

