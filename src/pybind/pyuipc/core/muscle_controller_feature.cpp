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

    class_MuscleControllerFeature.def(
        "read_muscle_groups_from",
        [](MuscleControllerFeature& self, geometry::SimplicialComplex& geo)
        { self.read_muscle_groups_from(geo); },
        py::arg("geometry"),
        R"(Read per-tet muscle ids from target geometry.
Args:
    geometry: SimplicialComplex where muscle ids are stored in.)");

    class_MuscleControllerFeature.def(
        "read_activations", [](MuscleControllerFeature& self, const std::unordered_map<int, std::vector<float>>& activations)
        { self.read_activations(activations); },
        py::arg("activations"),
        R"(Read the dictionary of activation lists and convert it internally to an Eigen matrix
Args:
    activations: Dict of activation values for each muscle)");

    class_MuscleControllerFeature.def(
        "update_next_frame", [](MuscleControllerFeature& self, geometry::SimplicialComplex& geo, int frame_id)
        { self.update_next_frame(geo, frame_id); },
        py::arg("geometry"),
        py::arg("frame_id"),
        R"(Update the active coeffs following the stored activation pattern and the provided frame_id 
Args:
    geometry: SimplicialComplex where active moduli are stored in.
    frame_id: Index into the activation map)");

    class_MuscleControllerFeature.attr("FeatureName") = MuscleControllerFeature::FeatureName;
}
}  // namespace pyuipc::core
