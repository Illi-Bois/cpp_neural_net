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
template <typename DerivedLayer>
class Layer {
 private:
  typedef util::Tensor<float> Tensor;
  
 public:
// CRTP ------------------------------------------------
  DerivedLayer& getRef() {
    return static_cast<DerivedLayer&>(*this);
  }
// End of CRTP -----------------------------------------

// NN Functions ----------------------------------------
/** Forward Pass
 *  computes and returns forward pass on the layer.
 *  Stores necesssary values needed to compute gradient. 
 */
  Tensor forward(const Tensor& input) {
    // CRTP interface
    return getRef().forward(input);
  }

/** Backwards Pass
 *  Updates updatable paramaters based on last forward passed values.
 *  Returns gradient of cost relative to last input, ie. dC/dx where x is last input.
 */
  Tensor backward(const Tensor& gradient) {
    // CRTP interface
    return getRef().backward(gradient);
  }
// End of NN Functions ---------------------------------

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