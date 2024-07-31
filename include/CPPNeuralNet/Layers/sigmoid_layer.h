#ifndef CPP_NN_LAYER_SIGMOID
#define CPP_NN_LAYER_SIGMOID

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/layer.h"
#include <cmath>

namespace cpp_nn {

class Sigmoid : public Layer {
private:
    util::Tensor<float> last_input;
    util::Tensor<float> last_output;

public:
    Sigmoid() = default;
    ~Sigmoid() override = default;

    util::Tensor<float> forward(const util::Tensor<float>& input) override;
    util::Tensor<float> backward(const util::Tensor<float>& gradient) override;
    void update_parameters(float learning_rate) override;
    void save_gradients() override;
    void clear_gradients() override;
};

} // namespace cpp_nn

#endif // CPP_NN_LAYER_SIGMOID