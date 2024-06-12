#ifndef CPP_NN_LINEAR
#define CPP_NN_LINEAR

#include "include/CPPNeuralNet/layer.h"
#include "include/CPPNeuralNet/Utils/utils.h"


namespace cpp_nn{
class Linear : public Layer {
  private:
    util::Matrix<double> weights_;
    util::Vector<double> biases_;
    double learning_rate_;
  public:
    Linear(int in_features, int out_features, double lr = 0.01);

    util::Vector<double> forward(const util::Vector<double>& input) override;
    util::Vector<double> backward(const util::Vector<double>& gradient) override;

    const util::Matrix<double>& get_weights() const {
      return weights_;
    }
    const util::Vector<double>& get_biases() const {
      return biases_;
    }
    void set_lr(double lr) {
      learning_rate_ = lr;
    }
};


} // cpp_nn




#endif //CPP_NN_LINEAR