#include "core.h"

#include <pcg_random.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <random>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <tuple>
#include <iomanip>
#include <chrono>
#include <Eigen/Eigen>
#include "macros.h"

namespace {
  namespace py = pybind11;
}


using CLS = topicmodels::SPTM<pcg32>;


PYBIND11_PLUGIN(mod) {
    py::module m("mod", "mod");
    py::class_<CLS>(m, "SPTM_CGS")
       .def(py::init<int, int, int, const std::vector<std::vector<int>> &, std::uint32_t>(),
           py::arg("n_topics"),
           py::arg("n_word_types"),
           py::arg("n_pseudo_docs"),
           py::arg("docs"),
           py::arg("seed")=0
        )
       .def("update", &CLS::update, "update")
       .def("log_marginalized_joint", &CLS::log_marginalized_joint, "log_marginalized_joint")
       .def("phikv", &CLS::get_phikv)
       .def("thetak", &CLS::get_thetak)
       .def("thetadk", &CLS::get_thetadk)
       .def("ld", &CLS::get_ld);

    return m.ptr();
}
