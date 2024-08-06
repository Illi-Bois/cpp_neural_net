#ifndef CPP_NN_LAYER_LINEAR
#define CPP_NN_LAYER_LINEAR

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/layer.h"

namespace cpp_nn {


class LinearLayer : public Layer {
 private:
  typedef util::Tensor<float> Tensor;

 public:
  LinearLayer(int input_size, int output_size);
  Tensor forward(const Tensor& input)     override;
  Tensor backward(const Tensor& gradient) override;

 private:
  Tensor weights_;
  Tensor biases_;
  Tensor last_input_;

/**
 * BACKGROUND FOR LAYERS
 * 
 * The Linear layer has two updatable Paramaters W and B. 
 *  with relation being
 *    y = L(x, W, B) = Wx + B
 *  
 *  dy/dB = I
 *  -> dC/dB = I * dC/dy  = dC/dy
 *  
 *  
 *  * Derivative for weights involve some magical tensor, which simplifies after product with previous gradient.
 *    Therefore it is beneficial to handle Weight updating as a special case.
 *    Given that dC/dy is a vector, 
 *  dC/dW = dC/dy * x^T
 *    We will not go into details to why this is case here.
 * 
 *  dy/dx = W^T
 *  -> dC/dx = W^T * dC/dy
 * 
 *  Needed intermediate members are:
 *    W, B
 *    x := previous input
 */
};


} // cpp_nn

#endif // CPP_NN_LAYER_LINEAR
