#include "CPPNeuralNet/Layers/sigmoid_layer.h"

namespace cpp_nn {

util::Tensor<float> Sigmoid::forward(const util::Tensor<float>& input) {
    last_input = input;
    util::Tensor<float> output(input.getShape());
    
    auto input_it = input.begin();
    auto output_it = output.begin();
    const auto end_it = input.end();
    //applies sigmoid function  1 / (1 + e^(-x)) to each of the element of the output tensor 
    while (input_it != end_it) {
        *output_it = 1.0f / (1.0f + std::exp(-(*input_it)));
        ++input_it;
        ++output_it;
    }
    
    last_output = output;
    return output;
}

util::Tensor<float> Sigmoid::backward(const util::Tensor<float>& gradient) {
    util::Tensor<float> output(gradient.getShape());
    
    auto grad_it = gradient.begin();
    auto last_output_it = last_output.begin();
    auto output_it = output.begin();
    const auto end_it = gradient.end();

    while (grad_it != end_it) {
        float sigmoid_x = *last_output_it;
        //compute derivative & multiply derivative my incoming gradient 
        *output_it = (*grad_it) * sigmoid_x * (1.0f - sigmoid_x);
        ++grad_it;
        ++last_output_it;
        ++output_it;
    }
    
    return output;
}

void Sigmoid::update_parameters(float learning_rate) {
}

void Sigmoid::save_gradients() {
}

void Sigmoid::clear_gradients() {
}

} // namespace cpp_nn