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
    last_input = input;
    util::Tensor<float> output(input.getShape());
    
    std::vector<int> indices(input.getOrder(), 0);
    for (int i = 0; i < input.getCapacity(); ++i) {
        // Convert linear index i to multi-dimensional indices
        int temp = i;
        for (int j = input.getOrder() - 1; j >= 0; --j) {
            indices[j] = temp % input.getDimension(j);
            temp /= input.getDimension(j);
        }
        output.getElement(indices) = std::max(0.0f, input.getElement(indices));
    }
    
    return output;
}
/**
 * derivative of ReLU is f'(x) = 1 if x > 0 else f'(x) = 0 
 * If x > 0, dL/dx = dL/dy * 1 = dL/dy 
 * else , dL/dx = dL/dy * 0 = 0
 */
util::Tensor<float> ReLU::backward(const util::Tensor<float>& gradient) {
    util::Tensor<float> output(gradient.getShape());
    
    std::vector<int> indices(gradient.getOrder(), 0);
    for (size_t i = 0; i < gradient.getCapacity(); ++i) {
        // Convert linear index i to multi-dimensional indices
        size_t temp = i;
        for (int j = gradient.getOrder() - 1; j >= 0; --j) {
            indices[j] = temp % gradient.getDimension(j);
            temp /= gradient.getDimension(j);
        }
        output.getElement(indices) = last_input.getElement(indices) > 0 ? gradient.getElement(indices) : 0.0f;
    }
    
    return output;
}

void ReLU::update_parameters(float learning_rate) {
}

void ReLU::save_gradients() {
}

void ReLU::clear_gradients() {
}

} // namespace cpp_nn