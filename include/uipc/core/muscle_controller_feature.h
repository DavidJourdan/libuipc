#pragma once
#include <uipc/core/feature.h>
#include <uipc/geometry/simplicial_complex.h>

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

  private:
    virtual std::string_view                      get_name() const override;
    S<MuscleControllerFeatureOverrider> m_impl;

};
}  // namespace uipc::core
