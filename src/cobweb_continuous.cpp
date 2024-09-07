#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include "cobweb_continuous_node.h"
#include "cobweb_continuous_tree.h"

namespace nb = nanobind;

NB_MODULE(cobweb_continuous, m)
{
    m.doc() = "cobweb continuous"; // optional module docstring

    nb::class_<CobwebContinuousNode>(m, "CobwebContinuousNode")
        .def(nb::init<int>())
        // .def("increment_counts", &CobwebContinuousNode::increment_counts)
        .def("log_prob", &CobwebContinuousNode::log_prob)
        .def("log_prob_class_given_instance", &CobwebContinuousNode::log_prob_class_given_instance)
        .def("__str__", &CobwebContinuousNode::__str__)
        .def("output_json", &CobwebContinuousNode::output_json)
        .def_ro("count", &CobwebContinuousNode::count)
        .def_ro("children", &CobwebContinuousNode::children,
                nb::rv_policy::reference)
        .def_ro("parent", &CobwebContinuousNode::parent,
                nb::rv_policy::reference)
        .def_ro("tree", &CobwebContinuousNode::tree,
                nb::rv_policy::reference)
        .def_ro("sum_sq", &CobwebContinuousNode::sum_sq,
                nb::rv_policy::reference)
        .def_ro("mean", &CobwebContinuousNode::mean,
                nb::rv_policy::reference);

    nb::class_<CobwebContinuousTree>(m, "CobwebContinuousTree")
        .def(nb::init<int, int, int>(),
            nb::arg("size"),
            nb::arg("covar_type") = 1,
            nb::arg("covar_from") = 1)
        .def("ifit", &CobwebContinuousTree::ifit, nb::rv_policy::reference)
        .def("predict", &CobwebContinuousTree::predict,
            nb::arg("instance"),
            nb::arg("max_nodes") = 1000,
            nb::arg("greedy") = false)
        .def("log_prob", &CobwebContinuousTree::log_prob,
            nb::arg("instance"),
            nb::arg("max_nodes") = 1000,
            nb::arg("greedy") = false)
        // .def("fit", &CobwebTree::fit,
        //      nb::arg("instances") = std::vector<AV_COUNT_TYPE>(),
        //      nb::arg("mode"),
        //      nb::arg("iterations") = 1,
        //      nb::arg("randomizeFirst") = true)
        // .def("categorize", &CobwebTree::categorize,
        //      nb::arg("instance") = std::vector<AV_COUNT_TYPE>(),
        //      // nb::arg("get_best_concept") = false,
        //      nb::rv_policy::reference)
        // .def("predict_probs", &CobwebTree::predict_probs_mixture)
        // .def("predict_probs_parallel", &CobwebTree::predict_probs_mixture_parallel)
        .def("clear", &CobwebContinuousTree::clear)
        .def("__str__", &CobwebContinuousTree::__str__)
        // .def("dump_json", &CobwebTree::dump_json)
        // .def("load_json", &CobwebTree::load_json)
        // .def("load_json_stream", &CobwebTree::load_json_stream)
        .def_ro("root", &CobwebContinuousTree::root, nb::rv_policy::reference);

}
