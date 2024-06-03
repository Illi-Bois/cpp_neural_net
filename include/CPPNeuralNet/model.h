#ifndef CPP_NN_MODEL
#define CPP_NN_MODEL

// TODO preferably find better lower level alternative. Consider array of pointers. 
#include "vector" 

#include "include/CPPNeuralNet/layer.h"
#include "include/CPPNeuralNet/Utils/utils.h"

namespace cpp_nn {

/***
 * Model is top-level wrapper for all neural net models. 
 * Layers can be added to Models to define behaviour such as input and output dimensions along with activation functions to be used. 
 * Training and testing will also be called through Model's methods.
 * 
 * Specific types of neural net models, such as Perceptrons, will inherit from Model. 
*/
class Model {
 private:
  std::vector<Layer> layers;
 public: 

  Model();
  
  Model& const addLayer(Layer layer);

  // Compute forward pass of input through all the layer.
  util::Vector<> forward(util::Vector<> input); 
  // Update all the layer according to last run input. 
  void backward(); // TODO see if return should be current gradient.
};

}

#endif  // CPP_NN_MODEL
