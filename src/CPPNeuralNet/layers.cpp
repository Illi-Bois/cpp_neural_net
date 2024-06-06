#include "include/CPPNeuralNet/layers.h"
#include "include/CPPNeuralNet/Utils/utils.h"



util::Vector<double> Linear::forward(util::Vector<double> input){
    //y = MX + b
    //would it be better to use matadd or +
    util::Vector<double> output = this->weights * input + this->biases;
    return output;
}