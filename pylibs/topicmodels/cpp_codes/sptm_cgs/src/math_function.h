#pragma once

#include <cmath>
#include <iostream>



namespace math_function {
  using Real = double;
  inline Real pow(Real a, Real b) {
    return std::pow(a, b);
  }

  inline Real pow(Real a, int b) {
    return std::pow(a, b);
  }

  inline Real abs(Real x) {
    return x > 0 ? x : -x;
  }

  inline Real ln(Real x) {
    return std::log(x);
  }

  inline Real lngamma(Real x) {
    return std::lgamma(x);
  }

  inline Real exp(Real x) {
    return std::exp(x);
  }


};

