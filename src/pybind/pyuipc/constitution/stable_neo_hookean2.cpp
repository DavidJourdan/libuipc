#include <pyuipc/constitution/stable_neo_hookean2.h>
#include <uipc/constitution/stable_neo_hookean2.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
using namespace uipc::geometry;
PyStableNeoHookean2::PyStableNeoHookean2(py::module& m)
{
    auto class_StableNeoHookean2 = py::class_<StableNeoHookean2, FiniteElementConstitution>(
        m, "StableNeoHookean2", R"(StableNeoHookean2 constitution for stable neo-Hookean hyperelastic material.)");

    class_StableNeoHookean2.def(py::init<const Json&>(),
                               py::arg("config") = StableNeoHookean2::default_config(),
                               R"(Create a StableNeoHookean2 constitution.
Args:
    config: Configuration dictionary (optional, uses default if not provided).)");

    class_StableNeoHookean2.def_static("default_config",
                                      &StableNeoHookean2::default_config,
                                      R"(Get the default StableNeoHookean2 configuration.
Returns:
    dict: Default configuration dictionary.)");

    class_StableNeoHookean2.def("apply_to",
                               [](StableNeoHookean2& self,
                                  SimplicialComplex& sc,
                                  const ElasticModuli& moduli,
                                  Float mass_density)
                               { self.apply_to(sc, moduli, mass_density); },
                               py::arg("sc"),
                               py::arg("moduli") =
                                   ElasticModuli::youngs_poisson(120.0_kPa, 0.49),
                               py::arg("mass_density") = 1.0e3,
                               R"(Apply StableNeoHookean2 constitution to a simplicial complex.
Args:
    sc: SimplicialComplex to apply to.
    moduli: ElasticModuli (default: Young's modulus 120.0 kPa, Poisson's ratio 0.49).
    mass_density: Mass density (default: 1000.0).)");
}
}  // namespace pyuipc::constitution
