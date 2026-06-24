#include <pyuipc/core/muscle_controller_feature.h>
#include <uipc/core/muscle_controller_feature.h>
#include <pybind11/stl.h>

namespace pyuipc::core
{
using namespace uipc::core;
PyMuscleControllerFeature::PyMuscleControllerFeature(py::module& m)
{
    auto class_MuscleControllerFeature =
        py::class_<MuscleControllerFeature, IFeature, S<MuscleControllerFeature>>(
            m, "MuscleControllerFeature", R"(Feature for controlling muscle activations.)");

    class_MuscleControllerFeature.def(
        "copy_active_coeffs_from",
        [](MuscleControllerFeature& self, geometry::SimplicialComplex& geo)
        { self.copy_active_coeffs_from(geo); },
        py::arg("geometry"),
        R"(Copy per-tet active moduli from target geometry.
Args:
    geometry: SimplicialComplex where active moduli are stored in.)");

    class_MuscleControllerFeature.attr("FeatureName") = MuscleControllerFeature::FeatureName;
}
}  // namespace pyuipc::core
