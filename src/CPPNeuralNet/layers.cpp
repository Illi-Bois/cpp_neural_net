#include "include/CPPNeuralNet/layer.h"
#include "include/CPPNeuralNet/Utils/utils.h"
#include "../../include/CPPNeuralNet/layer.h"

Linear::Linear(int in_features, int out_features, double lr)
    : weights_(out_features, in_features, 0.0),
      biases_(out_features, 0.0),
      learning_rate(lr) {
        //Initialize weights and biases with random values 
        //my include is acting weird again so there might be typos ;( will fix later)

} 


util::Vector<double> Linear::forward(util::Vector<double> input){
    //y = MX + b
    //would it be better to use matadd or +
    util::Vector<double> output = this->weights * input + this->biases;
    return output;
}