#include "../include/Perceptron.h"
//#include "NeuralNet/Perceptron/Perceptron.cpp"
#include <iostream>
#include <vector>

int main() {
    Perceptron myPerceptron(3, 0.1);

    // Example training data - 3 features per input
    std::vector<std::vector<float>> trainingInputs = {
        {0.5, -1.2, 0.3},
        {-1.5, 2.3, -0.8},
        {0.4, -1.4, 0.9}
    };

    // Corresponding labels for the training data
    std::vector<int> trainingLabels = {1, 0, 1};

    // Train the perceptron for 10 epochs
    myPerceptron.train(trainingInputs, trainingLabels, 10);

    // Example input for prediction
    std::vector<float> input = {0.5, -1.0, 0.2};

    // Predict the output using the trained Perceptron
    int output = myPerceptron.predict(input);

    // Print the prediction result
    std::cout << "Predicted output: " << output << std::endl;

    return 0;
}