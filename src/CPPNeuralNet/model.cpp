#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor.h"

#include "CPPNeuralNet/model.h"

// Model DEFINITION =============================================
namespace cpp_nn {

// Constructors -----------------------------------------
Model::Model() noexcept {}
// End of Constructors ----------------------------------

// Layer Modification -----------------------------------
Model& Model::addLayer(Layer* layer) {
  layers_.push_back(layer);
  
  return *this;
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
