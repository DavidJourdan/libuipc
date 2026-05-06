#include <pyuipc/constitution/muscle.h>
#include <uipc/constitution/muscle.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
using namespace uipc::geometry;
PyMuscle::PyMuscle(py::module& m)
{
    auto class_Muscle = py::class_<Muscle, FiniteElementExtraConstitution>(
        m, "Muscle", R"(Muscle constitution for a muscle material (Hill model).)");

    class_Muscle.def(py::init<const Json&>(),
                               py::arg("config") = Muscle::default_config(),
                               R"(Create a Muscle constitution.
Args:
    config: Configuration dictionary (optional, uses default if not provided).)");

    class_Muscle.def_static("default_config",
                                      &Muscle::default_config,
                                      R"(Get the default muscle configuration.
Returns:
    dict: Default configuration dictionary.)");

    class_Muscle.def("apply_to",
                               [](Muscle& self,
                                  SimplicialComplex& sc,
                                  py::array_t<Float> direction,
                                  Float anisotropy_modulus)
                               { self.apply_to(sc, to_matrix<Vector3>(direction), anisotropy_modulus); },
                               py::arg("sc"),
                               py::arg("direction"),
                               py::arg("passive_modulus") = 120.0_kPa,
                               py::arg("active_modulus") = 120.0_kPa,
                               R"(Apply muscle constitution to a simplicial complex.
Args:
    sc: SimplicialComplex to apply to.
    direction: orientation of the anisotropic material
    anisotropy_modulus: directional modulus (default: 120.0 kPa).)");
}
}  // namespace pyuipc::constitution
