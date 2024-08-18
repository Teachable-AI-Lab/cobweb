#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "cobweb_continuous_node.h"
#include "cobweb_continuous_tree.h"

namespace py = pybind11;

PYBIND11_MODULE(cobweb_continuous, m)
{
    m.doc() = "cobweb continuous"; // optional module docstring

    py::class_<CobwebContinuousNode>(m, "CobwebContinuousNode")
        .def(py::init<int>())
        // .def("increment_counts", &CobwebContinuousNode::increment_counts)
        .def("__str__", &CobwebContinuousNode::__str__)
        .def_readonly("count", &CobwebContinuousNode::count)
        .def("output_json", &CobwebContinuousNode::output_json)
        .def_readonly("children", &CobwebContinuousNode::children,
                py::return_value_policy::reference)
        .def_readonly("parent", &CobwebContinuousNode::parent,
                py::return_value_policy::reference)
        .def_readonly("tree", &CobwebContinuousNode::tree,
                py::return_value_policy::reference)
        .def_readonly("sum_sq", &CobwebContinuousNode::sum_sq,
                py::return_value_policy::reference)
        .def_readonly("mean", &CobwebContinuousNode::mean,
                py::return_value_policy::reference);

    py::class_<CobwebContinuousTree>(m, "CobwebContinuousTree")
        .def(py::init<int>())
        .def("ifit", &CobwebContinuousTree::ifit, py::return_value_policy::reference)
        // .def("fit", &CobwebTree::fit,
        //      py::arg("instances") = std::vector<AV_COUNT_TYPE>(),
        //      py::arg("mode"),
        //      py::arg("iterations") = 1,
        //      py::arg("randomizeFirst") = true)
        // .def("categorize", &CobwebTree::categorize,
        //      py::arg("instance") = std::vector<AV_COUNT_TYPE>(),
        //      // py::arg("get_best_concept") = false,
        //      py::return_value_policy::reference)
        // .def("predict_probs", &CobwebTree::predict_probs_mixture)
        // .def("predict_probs_parallel", &CobwebTree::predict_probs_mixture_parallel)
        .def("clear", &CobwebContinuousTree::clear)
        .def("__str__", &CobwebContinuousTree::__str__)
        // .def("dump_json", &CobwebTree::dump_json)
        // .def("load_json", &CobwebTree::load_json)
        // .def("load_json_stream", &CobwebTree::load_json_stream)
        .def_readonly("root", &CobwebContinuousTree::root, py::return_value_policy::reference);

}
