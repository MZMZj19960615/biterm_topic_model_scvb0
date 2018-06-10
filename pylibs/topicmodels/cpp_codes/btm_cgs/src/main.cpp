#include "btm_cgs.h"
#include <pcg_random.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <random>
#include <cinttypes>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <tuple>
#include <iomanip>
#include <chrono>
#include <Eigen/Eigen>

namespace {
  namespace py = pybind11;
}

class BTM_CGS {
  private:
    pcg32 urng;
    std::vector<std::tuple<int, int>> biterms;
    topicmodel::btm_cgs_core model;
    const double alpha;
    const double beta;
  public:

  BTM_CGS(
      int n_topics,
      int n_word_types,
      const std::vector<std::tuple<int, int>> &_biterms,
      double _alpha,
      double _beta,
      std::uint64_t seed
  ) : urng(seed), biterms(_biterms), model(biterms, n_word_types, n_topics, _alpha, _beta, urng), alpha(_alpha), beta(_beta) {
  }

  void update() {
    model.update(urng);
  }

  Eigen::ArrayXd get_thetak() {
    Eigen::ArrayXd thetak = model.Nk().cast<double>() + alpha;
    thetak /= thetak.sum();
    return thetak;
  }

  Eigen::ArrayXXd get_phikv() {
    Eigen::ArrayXXd phikv = model.Nkv().cast<double>() + beta;
    for (int k = 0; k < phikv.rows(); ++k) {
      double sum = 0.0;
      for (int v = 0; v < phikv.cols(); ++v) {
        sum += phikv(k, v);
      }
      for (int v = 0; v < phikv.cols(); ++v) {
        phikv(k, v) /= sum;
      }
    }
    return phikv;
  }
};

PYBIND11_PLUGIN(mod) {

    py::module m("mod", "mod");

    py::class_<BTM_CGS>(m, "BTM_CGS")
       .def(py::init<int, int, const std::vector<std::tuple<int, int>> &, double, double, std::uint64_t>(),
           py::arg("n_topics"),
           py::arg("n_word_types"),
           py::arg("biterms"),
           py::arg("alpha") = 0.1,
           py::arg("beta") = 0.01,
           py::arg("seed") = 0
        )
       .def("update", &BTM_CGS::update, "update")
       .def("thetak", &BTM_CGS::get_thetak)
       .def("phikv", &BTM_CGS::get_phikv);
    return m.ptr();
}
