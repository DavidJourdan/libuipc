#include <uipc/core/muscle_controller_feature.h>
#include <iostream>

namespace uipc::core
{
MuscleControllerFeature::MuscleControllerFeature(
    S<MuscleControllerFeatureOverrider> overrider)
    : m_impl(std::move(overrider))
{
}

void MuscleControllerFeature::copy_active_coeffs_from(geometry::SimplicialComplex& geo)
{
    m_impl->do_copy_active_coeffs_from(geo);
}

void MuscleControllerFeature::read_muscle_groups_from(geometry::SimplicialComplex& geo)
{
    auto labels_attr = geo.tetrahedra().find<Float>("muscle_id");
    UIPC_ASSERT(labels_attr, "[MuscleControllerFeature] Geometry has no 'muscle_id' attribute.");
    // for(int i = 0; i < MAX_MUSCLE_ID; ++i)
    //     muscle_ids_to_idx[i] = -1;

    Float max_muscle_id = 0;
    for(Float muscle_id : labels_attr->view())
    {
        if(muscle_id > max_muscle_id)
            max_muscle_id = muscle_id;
    }
    muscle_ids_to_idx = std::vector<int>(int(max_muscle_id) + 1, -1);

    int i = 0;
    for(IndexT muscle_id : labels_attr->view())
    {
        UIPC_ASSERT(muscle_id < MAX_MUSCLE_ID, "[MuscleControllerFeature] Muscle ID is out of range.")
        if(muscle_id > 0)
        {
            if(muscle_ids_to_idx[muscle_id] == -1)
            {
                muscle_ids_to_idx[muscle_id] = muscle_groups.size();
                muscle_groups.push_back(std::vector<int>{});
            }
            muscle_groups[muscle_ids_to_idx[muscle_id]].push_back(i);
        }
        i += 1;
    }
}

void MuscleControllerFeature::read_activations(const std::unordered_map<int, std::vector<float>>& activation_dict)
{
    // Check if dictionary is empty
    UIPC_ASSERT(activation_dict.size() > 0, "[MuscleControllerFeature] activation dict is empty");

    // Get the size of the first vector (assuming all vectors have same size)
    int nb_frames = activation_dict.begin()->second.size();
    int nb_muscles = activation_dict.size();

    UIPC_ASSERT(nb_muscles == muscle_groups.size(), "[MuscleControllerFeature] incorrect number of muscles in activation dict");

    // Create Eigen matrix with dimensions (nb_muscles, nb_frames)
    activations = Eigen::MatrixXd(nb_muscles, nb_frames);

    // Fill the matrix
    for (const auto& [key, values] : activation_dict) {
        // Check that all vectors have same size
        UIPC_ASSERT(values.size() == nb_frames, "[MuscleControllerFeature] inconsistent number of frames provided in activation dict");

        // Copy values to the matrix row
        for (int col = 0; col < nb_frames; ++col) {
            activations(muscle_ids_to_idx[key], col) = values[col];
        }
    }
}

void MuscleControllerFeature::update_next_frame(geometry::SimplicialComplex& geo, int frame_id)
{
    UIPC_ASSERT(frame_id < activations.cols() && frame_id >= 0,
            "[MuscleControllerFeature] Size mismatch: "
            "frame_id = {} does not match activation size = {}.",
            frame_id,
            activations.cols());

    UIPC_ASSERT(active_modulus > 0, "[MuscleControllerFeature] active_modulus has not been set.");

    auto attr = geo.tetrahedra().find<Float>("active_modulus");
    auto attr_view = view(*attr);

    for(int i = 0; i < activations.rows(); ++i)
    {
        for(int idx : muscle_groups[i])
        {
            attr_view[idx] = activations(i, frame_id) * active_modulus;
        }
    }

    copy_active_coeffs_from(geo);
}


std::string_view MuscleControllerFeature::get_name() const
{
    return FeatureName;
}
}  // namespace uipc::core
