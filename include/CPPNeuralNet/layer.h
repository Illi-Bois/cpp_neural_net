#ifndef CPP_NN_LAYER
#define CPP_NN_LAYER

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor.h"

namespace cpp_nn {

/***
 * Layer is interface for any object that defines a layer in a Model. 
 * Both linear weight-linkages and activation functions will inherit from Layer. 
 * Running and updating weights of layers will be done through its methods and not externally. 
 * 
 * Ideally, layers will also maintain last run deriviates for faster training.
*/

/***
 * Layer will be an abstract base class with pure virtual functions 
 * 
 */
class Layer {
 private:
  typedef util::Tensor<float> Tensor;

 public:
  // Virtual destructor of the Layer class.
  virtual ~Layer() = default;

  // Forward pass
  virtual Tensor forward(const Tensor& input) = 0;

  // Backward pass
  virtual Tensor backward(const Tensor& gradient) = 0;

  // Update parameters
  virtual void update_parameters(float learning_rate) = 0;

  // Save gradients for optimization algorithms
  virtual void save_gradients() = 0;

  // Clear gradients after parameter update
  virtual void clear_gradients() = 0;

/** WAIT TILL TENSOR IS COMPLETE ---------------------------------------------------
  // Compute forward pass of input through the layer.
  virtual util::Vector<double> forward(const util::Vector<double>& input); 
  // Update the layer according to last run input. 
  // Gradient from net layer is passed. Current gradient is returned.
  virtual util::Vector<double> backward(const util::Vector<double>& gradient); 
---------------------------------------------------------------------------------- */


  /**
   * Each Layer is a function where input is passed and forwarded.
   * Therefore, each layer will have their own local gradient.
   * 
   * y = L(x)
   * 
   * In backward, we will be passed dC/dy where C is cost function.
   * Let the current layer be Linear, meaning it has weights W and biases b.
   * We will compute gradiatents for W and b as 
   *   dC/dy * dy/dW and dC/dy * dy/db
   * respectively and use such to update the paramaters. 
   * At the end, we will however return 
   *   dC/dy * dy/dx 
   * So that it may propagate backwards and continue updating on previous layers.
   * 
   * Activation Layers, such as sigmoid and ReLU will not need to update any terms. 
   * It will however be forced still to return correct dC/dy * dy/dx
   * 
   * Therefore, it may be good implementation to store dy/dx at each layer.
   */
};

} // cpp_nn

#endif  // CPP_NN_LAYER