#pragma once
#include <uipc/core/feature.h>
#include <uipc/geometry/simplicial_complex.h>
#include <unordered_map>

namespace uipc::core
{
class UIPC_CORE_API MuscleControllerFeatureOverrider
{
  public:
    virtual ~MuscleControllerFeatureOverrider() = default;
    virtual void do_copy_active_coeffs_from(geometry::SimplicialComplex& geo) = 0;
};


class UIPC_CORE_API MuscleControllerFeature : public Feature
{
  public:
    constexpr static std::string_view FeatureName = "core/muscle_controller";

    MuscleControllerFeature(S<MuscleControllerFeatureOverrider>);

    void copy_active_coeffs_from(geometry::SimplicialComplex& geo);

    // void setup_muscle_coeffs(geometry::SimplicialComplex& geo, double passive_coeff);
    void read_muscle_groups_from(geometry::SimplicialComplex& geo);
    void read_activations(const std::unordered_map<int, std::vector<float>>& activations);
    void update_next_frame(geometry::SimplicialComplex& geo, int frame_id);

  private:
    virtual std::string_view get_name() const override;
    S<MuscleControllerFeatureOverrider> m_impl;

    static constexpr int MAX_MUSCLE_ID = 1100;
    std::array<int, MAX_MUSCLE_ID> muscle_ids_to_idx;
    std::vector<std::vector<int>> muscle_groups;
    Eigen::MatrixXd activations;
};
}  // namespace uipc::core
