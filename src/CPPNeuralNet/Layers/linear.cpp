#include "include/CPPNeuralNet/layer.h"
#include "include/CPPNeuralNet/Utils/utils.h"
#include <random>

namespace cpp_nn {

Linear::Linear(int in_features, int out_features, double lr)
    : weights_(out_features, in_features, 0.0),
      biases_(out_features, 0.0),
      learning_rate_(lr) {
    std::random_device rand_num;
    std::mt19937 generate_rand(rand_num);
    // got random number generate code online. random = object of std::random, gets random number from hardware random number generator
    // random (creates seed number) -> std::mt19937 -> better random number apparantly
    std::normal_distribution<double> dis(0.0,0.01);
    for(int i = 0; i < out_features; ++i){
      for(int j = 0; j < in_features; ++j){
        weights_(i,j) = dis(generate_rand);
      }
      biases_(i) = dis(generate_rand);
    }


        //Initialize weights and biases with random values 

} 


util::Vector<double> Linear::forward(const util::Vector<double>& input){
    //y = MX + b
    return  weights_ * input + biases_;
}


util::Vector<double> Linear::backward(const util::Vector<double>& gradient){
  //gradient with respect to input and parameter : dL/dx = W^T * dL/dy
  //gradient with respect to weights and bias : dL/dW = dL/dy * x^T

}

} // cpp_nn