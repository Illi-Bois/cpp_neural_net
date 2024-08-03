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

/**
 * BACKGROUND FOR LAYERS
 * 
 * The current Layer L will have some Layer Paramaters (in case of Linear, it will be W and B,  in case of activation, it might be called A. There may be more than one or two)
 *  The Layer Parameters will be named P1, P2.... 
 * The input paramater will be called x. The output y. 
 * 
 * In effect, the relation becomes:
 *  y = L(x, P1, P2, ...)
 * 
 * Then backpropagating, we aim to minize certain Cost function. In the context of layer, it becomes irrelevant what that cost function is, or even how far removed from the final cost the 
 *  current layer is.
 * That means we are attempting to minize gradient for y alone, meaning we are following dC/dy only. Therefore dC/dy is the gradient that must be passed to 'backwards' function.
 * 
 * It should also be mentioned that not every Layer Parameter needs udating. In activation functions specifically, parameters for ReLU exsist as P1 but does not get updated. 
 *  In Linear Layer, all Paramaters W=P1 and B=P2 will need to be updated. 
 * 
 * To minimize gradient following dC/dy for the Updatable Paramaters, we will need to compute dC/dPn where Pn are all the paramaters that needs to be updated. 
 *  By chain rule dC/dPn = dC/dy * dy/dPn. Note that dC/dy is given as paramater and only dy/dPn needs to be computed in a layer. 
 *  Therefore the members of the Layers will need to store only the values that facilitate computation of dy/dPn (and dy/dx for reasons soon to be explained). 
 * 
 * After each Updatable Paramaters are updated, we must return gradient on the current layer to propagate to previous layer. The previous layer will need gradient of cost relative to its output,
 *  which precisely is x, the input of current layer. To compute dC/dx = dC/dy * dy/dx, the layer will need to compute dy/dx as well. This might necessitate storing of last input even for Ativation layers.
 * The dC/dx then gets passed as return and is used to propagate previous layers and so on. 
 */
};

} // cpp_nn

#endif  // CPP_NN_LAYER