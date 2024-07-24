#ifndef CPP_NN_MODEL
#define CPP_NN_MODEL

#include "CPPNeuralNet/layer.h"
#include "CPPNeuralNet/Utils/utils.h"

#include "vector" 

namespace cpp_nn {

/***
 * Model is top-level wrapper for all neural net models. 
 * Layers can be added to Models to define behaviour such as input and output dimensions along with activation functions to be used. 
 * Training and testing will also be called through Model's methods.
 * 
 * Specific types of neural net models, such as Perceptrons, will inherit from Model. 
*/
class Model {
  typedef util::Tensor<float> Tensor;
 public:
// Constructors -----------------------------------------
  Model() noexcept; 
// End of Constructors ----------------------------------

// Layer Modification -----------------------------------
  Model& addLayer(Layer* layer);
// End of Layer Modification ----------------------------

// Propagations -----------------------------------------
  // TODO: formulate alternative to minimize Tensor movement
  Tensor forward(const Tensor& input);
  Tensor bakcward(const Tensor& input);
// End of Propagations ----------------------------------

 private:
// Members ----------------------------------------------
  std::vector<Layer*> layers_; 
// End of Members ---------------------------------------
};

} // cpp_nn


// Model DEFINITION =============================================
namespace cpp_nn {

// Constructors -----------------------------------------
Model::Model() noexcept {}
// End of Constructors ----------------------------------

// Layer Modification -----------------------------------
Model& Model::addLayer(Layer* layer) {
  layers_.push_back(layer);
}
// End of Layer Modification ----------------------------

// Propagations -----------------------------------------
// TODO: formulate alternative to minimize Tensor movement
Model::Tensor Model::forward(const Tensor& input) {
  Tensor previous = input;
  for (Layer* layer : layers_) {
    previous = layer->forward(previous);
  }
  return previous;
}
Model::Tensor Model::bakcward(const Tensor& input) {
  Tensor previous = input;
  
  auto iter = layers_.end();
  while (iter != layers_.begin()) {
    --iter;

    previous = (*iter)->backward(previous);
  }

  return previous;
}
// End of Propagations ----------------------------------

} // cpp_nn
// End of Model DEFINITION ======================================

#endif  // CPP_NN_MODEL
