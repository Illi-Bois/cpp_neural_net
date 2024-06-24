#ifndef CPP_NN_LINEAR
#define CPP_NN_LINEAR

#include "layer.h"
#include "Utils/utils.h"

#include <random>


namespace cpp_nn {

class Linear : public Layer {
  
  private:
/** WAIT TILL TENSOR IS FINISHED  ---------------------------------------------------------------------------------------

    util::Matrix<double> weights_;
    util::Vector<double> biases_;
    double learning_rate_;
------------------------------------------------------------------------------------------------------------- */

  public:
/** WAIT TILL TENSOR IS FINISHED  ---------------------------------------------------------------------------------------
    Linear(int in_features, int out_features, double lr = 0.01);

    util::Vector<double> forward(const util::Vector<double>& input) override;
    util::Vector<double> backward(const util::Vector<double>& gradient) override;
------------------------------------------------------------------------------------------------------------- */

    /**
     * Linear Backswards:
     * as Layer is y=Wx + b
     * given grad_y(C) where C is cost or loss to be minimized, which is a vector
     * 
     * grad_W(C) = grad_y(C) * x^T
     * grac_b(C) = grad_y(C)
     * 
     * grad_x(C) = W^T * grad_y(C)
     * 
     * 
     * On d(Wx)/dW
     * https://math.stackexchange.com/questions/1621948/derivative-of-a-vector-with-respect-to-a-matrix
     */

/** WAIT TILL TENSOR IS FINISHED  ---------------------------------------------------------------------------------------
    const util::Matrix<double>& get_weights() const {
      return weights_;
    }
    const util::Vector<double>& get_biases() const {
      return biases_;
    }
    void set_lr(double lr) {
      learning_rate_ = lr;
    }
------------------------------------------------------------------------------------------------------------- */

};


} // cpp_nn




#endif //CPP_NN_LINEAR