#ifndef CPP_NN_BINARY_CROSS_ENTROPY_LOSS_H
#define CPP_NN_BINARY_CROSS_ENTROPY_LOSS_H

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/loss_function.h"

namespace cpp_nn {

class BinaryCrossEntropyLoss : public LossFunction {
private:
  typedef util::Tensor<float> Tensor;
  static Tensor tensor_log(const Tensor& input);
public:
  float compute_loss(const Tensor& predictions, const Tensor& targets) const override;
  Tensor compute_gradient(const Tensor& predictions, const Tensor& targets) const override;
};

} // namespace cpp_nn

#endif // CPP_NN_BINARY_CROSS_ENTROPY_LOSS_H