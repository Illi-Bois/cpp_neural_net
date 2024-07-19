#include "CPPNeuralNet/layer.h"
#include <cstdlib>

namespace cpp_nn {

LinearLayer::LinearLayer(int input_size, int output_size) {
    weights = util::Tensor<double>({input_size, output_size}).Reshape({input_size * output_size});
    biases = util::Tensor<double>({output_size});
    
    for (auto& w : weights) w = (std::rand() / (RAND_MAX + 1.0)) - 0.5;
    for (auto& b : biases) b = (std::rand() / (RAND_MAX + 1.0)) - 0.5;
}

util::Tensor<double> LinearLayer::forward(const util::Tensor<double>& input) {
    this->input = input;
    // write some test cases on if -1 works?? I dont remember too well if it does 
    output = (input * weights.Reshape({input.getDimension(-1), biases.getDimension(0)})) + biases;
    return output;
}

util::Tensor<double> LinearLayer::backward(const util::Tensor<double>& gradient) {
    weight_gradients = input.Transpose() * gradient;
    bias_gradients = gradient.Reshape({biases.getShape()});
    return gradient * weights.Transpose();
}

void LinearLayer::update_parameters(double learning_rate) {
    // weights = weights - (weight_gradients * learning_rate);
    // weights = weights - weight_gradients (Tensor) * lr number
    // biases = biases - (bias_gradients * learning_rate);
    // biases = biases - bias_gradients (Tensor) * lr (number)
}

void LinearLayer::save_gradients() {
    // might or might not need 
}

void LinearLayer::clear_gradients() {
    weight_gradients = util::Tensor<double>(weight_gradients.getShape());
    bias_gradients = util::Tensor<double>(bias_gradients.getShape());
}



} // namespace cpp_nn

