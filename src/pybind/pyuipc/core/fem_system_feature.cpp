#include <pyuipc/core/fem_system_feature.h>
#include <uipc/core/fem_system_feature.h>
#include <pybind11/stl.h>

namespace pyuipc::core
{
using namespace uipc::core;
PyFEMSystemFeature::PyFEMSystemFeature(py::module& m)
{
    auto class_FEMSystemFeature =
        py::class_<FEMSystemFeature, IFeature, S<FEMSystemFeature>>(
            m, "FEMSystemFeature", R"(Feature for computing FEM energy, gradients, and Hessians.)");

    class_FEMSystemFeature.def(
        "fem_energy",
        [](FEMSystemFeature& self, U64 uid, geometry::Geometry& energy)
        { self.fem_energy(uid, energy); },
        py::arg("uid"),
        py::arg("energy"),
        R"(Compute FEM energy for a constitution.
Args:
    uid: constitution UID.
    energy: Geometry to store energy values (modified in-place).)");

    class_FEMSystemFeature.def(
        "fem_gradient",
        [](FEMSystemFeature& self, U64 uid, geometry::Geometry& vert_grad)
        { self.fem_gradient(uid, vert_grad); },
        py::arg("uid"),
        py::arg("vert_grad"),
        R"(Compute fem gradient for a constitution.
Args:
    uid: constitution UID.
    vert_grad: Geometry to store vertex gradients (modified in-place).)");

    class_FEMSystemFeature.def(
        "fem_hessian",
        [](FEMSystemFeature& self, U64 uid, geometry::Geometry& vert_hess)
        { self.fem_hessian(uid, vert_hess); },
        py::arg("uid"),
        py::arg("vert_hess"),
        R"(Compute fem Hessian for a constitution.
Args:
    uid: constitution UID.
    vert_hess: Geometry to store vertex Hessians (modified in-place).)");

    class_FEMSystemFeature.def(
        "fem_energy",
        [](FEMSystemFeature& self, const constitution::IConstitution& c, geometry::Geometry& prims)
        { self.fem_energy(c, prims); },
        py::arg("constitution"),
        py::arg("prims"),
        R"(Compute fem energy for a constitution.
Args:
    constitution: Constitution to compute energy for.
    prims: Geometry containing primitives (modified in-place with energy values).)");

    class_FEMSystemFeature.def(
        "fem_gradient",
        [](FEMSystemFeature& self, const constitution::IConstitution& c, geometry::Geometry& vert_grad)
        { self.fem_gradient(c, vert_grad); },
        py::arg("constitution"),
        py::arg("vert_grad"),
        R"(Compute fem gradient for a constitution.
Args:
    constitution: Constitution to compute gradient for.
    vert_grad: Geometry to store vertex gradients (modified in-place).)");

    class_FEMSystemFeature.def(
        "fem_hessian",
        [](FEMSystemFeature& self, const constitution::IConstitution& c, geometry::Geometry& vert_hess)
        { self.fem_hessian(c, vert_hess); },
        py::arg("constitution"),
        py::arg("vert_hess"),
        R"(Compute fem Hessian for a constitution.
Args:
    constitution: Constitution to compute Hessian for.
    vert_hess: Geometry to store vertex Hessians (modified in-place).)");

    class_FEMSystemFeature.attr("FeatureName") = FEMSystemFeature::FeatureName;
}
}  // namespace pyuipc::core
