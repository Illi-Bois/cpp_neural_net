#include "../include/Perceptron.h"
//#include "NeuralNet/Perceptron/Perceptron.cpp"
#include <iostream>
#include <vector>

int main() {
    Perceptron myPerceptron(2, 0.1);

    // Example training data - 3 features per input
    std::vector<std::vector<float>> trainingInputs = {
      {0, 0},
      {1, 0},
      {0, 1},
      {1, 1},
    };

    // Corresponding labels for the training data
    std::vector<int> trainingLabels = {0, 1, 1, 1};

    // Train the perceptron for 10 epochs
    myPerceptron.train(trainingInputs, trainingLabels, 100);


    // Print the prediction result
    for (auto& input : trainingInputs) {
      std::cout << "Predicted output: " << myPerceptron.predict(input) << std::endl;    
    }

    return 0;
}