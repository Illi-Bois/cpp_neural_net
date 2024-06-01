#ifndef CPP_NN_MODEL
#define CPP_NN_MODEL

namespace cpp_nn {

/***
 * Model is top-level wrapper for all neural net models. 
 * Layers can be added to Models to define behaviour such as input and output dimensions along with activation functions to be used. 
 * Training and testing will also be called through Model's methods.
 * 
 * Specific types of neural net models, such as Perceptrons, will inherit from Model. 
*/
class Model {

};

}

#endif  // CPP_NN_MODEL
