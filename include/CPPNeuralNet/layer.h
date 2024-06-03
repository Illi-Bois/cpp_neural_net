#ifndef CPP_NN_LAYER
#define CPP_NN_LAYER

#include "include/CPPNeuralNet/Utils/Utils.h"

namespace cpp_nn {

/***
 * Layer is interface for any object that defines a layer in a Model. 
 * Both linear weight-linkages and activation functions will inherit from Layer. 
 * Running and updating weights of layers will be done through its methods and not externally. 
 * 
 * Ideally, layers will also maintain last run deriviates for faster training.
*/
class Layer {
 private:

 public:

  // Compute forward pass of input through the layer.
  virtual util::Vector<double> forward(util::Vector<double> input); 
  // Update the layer according to last run input. 
  virtual void backward(); 
};

class Linear : public Layer{
  private:
  public:
};


} // cpp_nn

#endif  // CPP_NN_LAYER