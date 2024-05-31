#include "../include/Perceptron.h"

#include <cstdlib>
#include <ctime>

Perceptron::Perceptron(int numfeatures, float rate){
    learningrate = rate;
    bias = 0;
    srand(time(NULL));
    for (int i = 0; i < numfeatures; ++i) {
        weights.push_back(((float)rand() / RAND_MAX * 2 - 1));
    }
}

int Perceptron::activationfunction(float z) {
    return (z>0) ? 1: 0;
}

void Perceptron::train(const std::vector<std::vector<float>>& input, const std::vector<int>& label, int epoch) {
    
    for(int i = 0;i < epoch; i++){ //epoch
        for(int j = 0; j < input.size();j++){ // each input
            
            // predict
            int output = this->predict(input[j]);

            // update
            if(output != label[j]){
                float error = label[j]-output;
                for(int k = 0 ; k < weights.size();k++){
                    weights[k] += learningrate * error * input[j][k];
                }
                bias+= learningrate * error;
            }
        }
    }
}

int Perceptron::predict(const std::vector<float>& input){
    float z = bias;
    for(int i = 0 ; i < input.size();i++){
        z+=weights[i] * input[i];
    }
    return activationfunction(z);
}




