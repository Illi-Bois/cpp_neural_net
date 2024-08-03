#ifndef CPP_NN_LOSS_FUNCTION
#define CPP_NN_LOSS_FUNCTION

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor.h"

namespace cpp_nn {

class LossFunction {
private:
    typedef util::Tensor<float> Tensor;
public:
    virtual float compute_loss(const Tensor& predictions, const Tensor& targets) const = 0;
    virtual Tensor compute_gradient(const Tensor& predictions, const Tensor& targets) const = 0;
    virtual ~LossFunction() = default;
};

}

#endif // CPP_NN_LOSS_FUNCTION