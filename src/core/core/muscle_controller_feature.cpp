#include <uipc/core/muscle_controller_feature.h>
#include <uipc/common/log.h>
#include <fmt/ranges.h>
#include <algorithm>
#include <vector>

namespace uipc::core
{
MuscleControllerFeature::MuscleControllerFeature(S<MuscleControllerFeatureOverrider> overrider)
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

    Float max_muscle_id = 0;
    for(Float muscle_id : labels_attr->view())
    {
        if(muscle_id > max_muscle_id)
            max_muscle_id = muscle_id;
    }
    muscle_ids_to_idx = std::vector<int>(static_cast<int>(max_muscle_id) + 1, -1);

    int i = 0;
    for(Float muscle_id : labels_attr->view())
    {
        if(muscle_id > 0)
        {
            if(muscle_ids_to_idx[static_cast<int>(muscle_id)] == -1)
            {
                muscle_ids_to_idx[static_cast<int>(muscle_id)] = muscle_groups.size();
                muscle_groups.push_back(std::vector<int>{});
            }
            muscle_groups[muscle_ids_to_idx[static_cast<int>(muscle_id)]].push_back(i);
        }
        i += 1;
    }
}

void MuscleControllerFeature::read_activations(const std::unordered_map<int, std::vector<float>>& activation_dict)
{
    UIPC_ASSERT(!activation_dict.empty(), "[MuscleControllerFeature] activation dict is empty");
    UIPC_ASSERT(!muscle_groups.empty(),
                "[MuscleControllerFeature] no muscle groups read yet, call read_muscle_groups_from() first");

    // the frame count is dictated by the csv, all columns must agree on it
    int nb_frames = static_cast<int>(activation_dict.begin()->second.size());

    // muscles without an activation column simply keep their zero activation
    activations = Eigen::MatrixXd::Zero(static_cast<int>(muscle_groups.size()), nb_frames);

    std::vector<int> dropped;
    for(const auto& [muscle_id, values] : activation_dict)
    {
        UIPC_ASSERT(static_cast<int>(values.size()) == nb_frames,
                    "[MuscleControllerFeature] inconsistent number of frames provided in activation dict");

        bool known = muscle_id >= 0
                     && muscle_id < static_cast<int>(muscle_ids_to_idx.size())
                     && muscle_ids_to_idx[muscle_id] >= 0;
        if(!known)
        {
            dropped.push_back(muscle_id);  // csv column of a muscle absent from the mesh
            continue;
        }

        for(int col = 0; col < nb_frames; ++col)
            activations(muscle_ids_to_idx[muscle_id], col) = values[col];
    }

    if(!dropped.empty())
    {
        std::sort(dropped.begin(), dropped.end());
        logger::warn("[MuscleControllerFeature] dropped activation data of {} muscles absent from the mesh: {}",
                     dropped.size(),
                     fmt::join(dropped, ", "));
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

    auto attr      = geo.tetrahedra().find<Float>("active_modulus");
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
