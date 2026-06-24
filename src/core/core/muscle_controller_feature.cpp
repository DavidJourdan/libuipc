#include <uipc/core/muscle_controller_feature.h>

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

std::string_view MuscleControllerFeature::get_name() const
{
    return FeatureName;
}
}  // namespace uipc::core
