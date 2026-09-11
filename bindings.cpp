#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "module.hpp"



NB_MODULE(hnsw, module) {

    nanobind::enum_<Metric>(module, "Metric")
        .value("Cosine", Metric::Cosine)
        .value("Euclidean", Metric::Euclidean);

    nanobind::class_<Node>(module, "Node")
        .def(
            nanobind::init<std::string, std::vector<float>>(),
            nanobind::arg("content"),
            nanobind::arg("embedding")
        )
        .def_rw("content", &Node::content)
        .def_rw("embedding", &Node::embedding);


    nanobind::class_<Index>(module, "Index")
        .def(
            nanobind::init<Metric>(), 
            nanobind::arg("metric") = Metric::Cosine
        )
        .def("insert", &Index::insert)
        .def("search", &Index::search);

}
