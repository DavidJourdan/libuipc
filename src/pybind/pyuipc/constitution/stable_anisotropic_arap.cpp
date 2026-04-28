#include <pyuipc/constitution/stable_anisotropic_arap.h>
#include <uipc/constitution/stable_anisotropic_arap.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
using namespace uipc::geometry;
PyStableAnisotropicARAP::PyStableAnisotropicARAP(py::module& m)
{
    auto class_StableAnisotropicARAP = py::class_<StableAnisotropicARAP, FiniteElementConstitution>(
        m, "StableAnisotropicARAP", R"(StableAnisotropicARAP constitution for stable anisotropic ARAP hyperelastic material.)");

    class_StableAnisotropicARAP.def(py::init<const Json&>(),
                               py::arg("config") = StableAnisotropicARAP::default_config(),
                               R"(Create a StableAnisotropicARAP constitution.
Args:
    config: Configuration dictionary (optional, uses default if not provided).)");

    class_StableAnisotropicARAP.def_static("default_config",
                                      &StableAnisotropicARAP::default_config,
                                      R"(Get the default StableAnisotropicARAP configuration.
Returns:
    dict: Default configuration dictionary.)");

    class_StableAnisotropicARAP.def("apply_to",
                               [](StableAnisotropicARAP& self,
                                  SimplicialComplex& sc,
                                  py::array_t<Float> direction,
                                  Float anisotropy_modulus)
                               { self.apply_to(sc, to_matrix<Vector3>(direction), anisotropy_modulus); },
                               py::arg("sc"),
                               py::arg("direction"),
                               py::arg("anisotropy_modulus") = 120.0_kPa,
                               R"(Apply StableAnisotropicARAP constitution to a simplicial complex.
Args:
    sc: SimplicialComplex to apply to.
    direction: orientation of the anisotropic material
    anisotropy_modulus: directional modulus (default: 120.0 kPa).)");
}
}  // namespace pyuipc::constitution
