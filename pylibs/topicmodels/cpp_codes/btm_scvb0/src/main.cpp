#include "btm_scvb0/btm_scvb0_core.h"

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

class BTM_SCVB0 {
  private:
    pcg32 urng;
    std::vector<std::tuple<int, int>> biterms;
    topicmodel::btm_scvb0_core model;
    const double alpha;
    const double beta;
  public:

  BTM_SCVB0(
      int n_topics,
      int n_word_types,
      const std::vector<std::tuple<int, int>> &_biterms,
      double _alpha,
      double _beta,
      double tau,
      double kappa,
      std::uint64_t seed
  ) : urng(seed), biterms(_biterms), model(n_word_types, static_cast<int>(biterms.size()), n_topics, _alpha, _beta, tau, kappa, urng), alpha(_alpha), beta(_beta) {
  }

  void update() {
    std::shuffle(std::begin(biterms), std::end(biterms), urng);
    for (const auto &biterm : biterms) {
      model.update_with_biterm(std::get<0>(biterm), std::get<1>(biterm));
    }
  }

  Eigen::ArrayXd get_thetak() {
    Eigen::ArrayXd thetak = model.Nk() + alpha;
    thetak /= thetak.sum();
    return thetak;
  }

  Eigen::ArrayXXd get_phikv() {
    Eigen::ArrayXXd phikv = model.Nkv() + beta;
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

    py::class_<BTM_SCVB0>(m, "BTM_SCVB0")
       .def(py::init<int, int, const std::vector<std::tuple<int, int>> &, double, double, double, double, std::uint64_t>(),
           py::arg("n_topics"),
           py::arg("n_word_types"),
           py::arg("biterms"),
           py::arg("alpha") = 0.1,
           py::arg("beta") = 0.01,
           py::arg("tau") = 1000,
           py::arg("kappa") = 0.8,
           py::arg("seed") = 0
        )
       .def("update", &BTM_SCVB0::update, "update")
       .def("thetak", &BTM_SCVB0::get_thetak)
       .def("phikv", &BTM_SCVB0::get_phikv);

    return m.ptr();
}
