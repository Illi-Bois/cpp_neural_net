#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include <vector>
 
class Perceptron{
private:
    std::vector<float> weights;
    float bias = 0;
    float learningrate;
    int activationfunction(float z);
public:
    Perceptron(int numfeatures, float rate = 0.1);
    void train(const std::vector<std::vector<float>>& input, const std::vector<int>& label, int epoch);
    int predict(const std::vector<float>& input);
};

#endif