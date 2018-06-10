#include <iostream>
#include <climits>
#include <Eigen/Eigen>

template <typename ArrayXXd>
class LazyMultArray {
  public:
    LazyMultArray(ArrayXXd &array) : array_(array){
    }
    void lazy_mult(double num) {
      if (mult_ * num < 1e-180) {
        refresh(num);
      } else {
        mult_ *= num;
      }
    }
    void set(int i, int j, double value) {
      array_(i, j) = value / mult_;
    }
    double get(int i, int j) {
      return array_(i, j) * mult_;
    }

    void refresh(double num = 1.0) {
      for (int num_rows = array_.rows(), r = 0; r < num_rows; ++r) {
        for (int num_cols = array_.cols(), c = 0; c < num_cols; ++c) {
          array_(r, c) = (array_(r, c) * mult_) * num;
        }
      }
      mult_ = 1.0;
    }
  public:
    ArrayXXd &array_;
    double mult_ = 1.0;
};
