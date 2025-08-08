// #ifdef PYTHON
// pybind.cpp - Vulkan Ray Query Example (Headless, No GUI)
// Output: Occlusion matrix (30x20) for fixed Txs and Rxs testing against a box
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ray_query_App.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(VulkanRayQuery, m)
{
    m.doc() = "Vulkan-based Ray Query plugin for Python using Ray-Triangle intersection.";

    py::class_<RayQueryApp>(m, "RayQueryApp")
        .def(py::init<std::vector<float>, std::vector<uint32_t>, std::string, std::string, std::string>())
        .def("QueryForLOS", &RayQueryApp::QueryForLOS, "txs"_a, "rxs"_a)
        .def("QueryForNLOS", &RayQueryApp::QueryForNLOS, "rxs"_a, "rayDirs"_a)
        .def("__del__", &RayQueryApp::cleanUp);

    py::class_<Vertex>(m, "Vertex")
        .def(py::init<>()) // default constructor
        .def_readwrite("pos", &Vertex::pos)
        .def_readwrite("pad", &Vertex::pad);
}

// #endif