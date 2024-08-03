#include "CPPNeuralNet/Layers/linear_layer.h"
#include <cstdlib>

namespace cpp_nn {

LinearLayer::LinearLayer(int input_size, int output_size)
    : weights(util::Tensor<float>({input_size, output_size})),
      biases(util::Tensor<float>({output_size})),
      input(util::Tensor<float>({1})),  // Initialize input & output with a dummy shape
      output(util::Tensor<float>({1})), 
      weight_gradients(util::Tensor<float>({input_size, output_size})),
      bias_gradients(util::Tensor<float>({output_size}))
{
    // Initialize weights and biases
    for (auto& w : weights) w = (std::rand() / (RAND_MAX + 1.0f)) - 0.5f;
    for (auto& b : biases) b = (std::rand() / (RAND_MAX + 1.0f)) - 0.5f;
}

util::Tensor<float> LinearLayer::forward(const util::Tensor<float>& input) {
    this->input = input;
    // write some test cases on if -1 works?? I dont remember too well if it does 
    output = (input * weights.Reshape({input.getDimension(-1), biases.getDimension(0)})) + biases;
    return output;
}

util::Tensor<float> LinearLayer::backward(const util::Tensor<float>& gradient) {
    weight_gradients = input.Transpose() * gradient;
    bias_gradients = gradient.Reshape({biases.getShape()});
    return gradient * weights.Transpose();
}

void LinearLayer::update_parameters(float learning_rate) {
    weights = weights + (weight_gradients * -learning_rate);
    biases = biases + (bias_gradients * -learning_rate);
}

void LinearLayer::save_gradients() {
    // might or might not need 
}

void LinearLayer::clear_gradients() {
    weight_gradients = util::Tensor<float>(weight_gradients.getShape());
    bias_gradients = util::Tensor<float>(bias_gradients.getShape());
}

} // namespace cpp_nn

