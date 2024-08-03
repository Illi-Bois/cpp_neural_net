#include "CPPNeuralNet/Loss_Functions/bce_loss.h"

namespace cpp_nn {

BinaryCrossEntropyLoss::Tensor BinaryCrossEntropyLoss::tensor_log(const Tensor& input) {
  Tensor result(input.getShape());
  auto it_result = result.begin();
  for (auto it = input.begin(); it != input.end(); ++it, ++it_result) {
      *it_result = std::log(*it);
  }
  return result;
}

float BinaryCrossEntropyLoss::compute_loss(const Tensor& predictions, const Tensor& targets) const {
  Tensor log_pred = tensor_log(predictions);
  Tensor b(predictions.getShape(), 1.0f);
  Tensor log_1_minus_pred = tensor_log(predictions * (-1.0f) + b);
  Tensor loss = ((targets * log_pred) + ((targets * (-1.0f) + b) * log_1_minus_pred)) * (-1.0f);
  
  float sum = 0.0f;
  for (auto it = loss.begin(); it != loss.end(); ++it) {
      sum += *it;
  }
  return sum / predictions.getCapacity();
}

BinaryCrossEntropyLoss::Tensor BinaryCrossEntropyLoss::compute_gradient(const Tensor& predictions, const Tensor& targets) const {
    /**
     * would be (predictions - targets) / (predicitions * (1-predictions)) 
     * divided by predictions.getCapacity? implement after element wise subtraction
     */
}

} // namespace cpp_nn