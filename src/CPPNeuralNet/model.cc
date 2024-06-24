
// #include "include/CPPNeuralNet/model.h"
// #include "include/CPPNeuralNet/Utils/utils.h"

// namespace cpp_nn {

// util::Vector<> Model::forward(util::Vector<> input) {
//   util::Vector current_values = input; // TODO fix name

//   for (auto& layer : this->layers) {
//     current_values = layer.forward(current_values);
//   }

//   return current_values;
// }

// void Model::backward() {
//   for (auto& layer : this->layers) {
//     layer.backward();
//   }
// }

// } // cpp_nn