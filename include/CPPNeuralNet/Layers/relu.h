#ifndef CPP_NN_RELU
#define CPP_NN_RELU

#include "include/CPPNeuralNet/layer.h"
#include "include/CPPNeuralNet/Utils/utils.h"

namespace cpp_nn{

class RelU : public Layer {
  public:
    util::Vector<double> forward(const util::Vector<double>& input) override;
    util::Vector<double> backward(const util::Vector<double>& gradient) override;
};

} //cpp_nn

#endif //CPP_NN_RELU