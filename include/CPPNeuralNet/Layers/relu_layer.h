#ifndef CPP_NN_LAYER_RELU
#define CPP_NN_LAYER_RELU

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/layer.h"

namespace cpp_nn {

class ReLU : public Layer {
private:
  typedef util::Tensor<float> Tensor;

  util::Tensor<float> last_input_;

public:
// Constructor -----------------------------------------
  ReLU();
// End of Constructor ----------------------------------

  Tensor forward(const Tensor& input)     override;
  Tensor backward(const Tensor& gradient) override;
};

/** 
 *  background in ReLU.
 *  ReLU can be thought of as clamping the lower bound of input at 0,
 *    which can be implemened either as max(0, input) or as ternary 
 *    input > 0 ? input : 0
 *  Regardless of the implementation, the motiation is to encourage 
 *    'strong signal' (indicated by it being greater than 0) and 
 *    discourage 'weak signals'.
 *  ReLU is the simplest such activation that adds non-linearity thorugh its piecewise nature.
 * 
 *  The layer then is with no Layer Paramater and is written as piecewise
 *    y = L(x) =  { x, x > 0
 *                  0, x <= 0
 *    
 *  The graidient is, for each element
 *    dy/dx = { 1, x > 0
 *              0, x <= 0
 *  
 *  The passing gradient dC/dx is then computed element-wise as
 *    dC/dx = { dC/dy, x > 0
 *              0,     x <= 0
 *  
 */
} // cpp_nn

#endif // CPP_NN_LAYER_RELU