#include <finite_element/constitutions/muscle_controller.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(MuscleController);

// =====================================================================
// Overrider
// =====================================================================
MuscleControllerFeatureOverrider::MuscleControllerFeatureOverrider(MuscleController* accessor)
{
    UIPC_ASSERT(accessor, "MuscleController must not be null");
    m_accessor = *accessor;
}

void MuscleControllerFeatureOverrider::do_copy_active_coeffs_from(geometry::SimplicialComplex& geo)
{
    m_accessor->copy_active_coeffs_from(geo);
}

// =====================================================================
// SimSystem
// =====================================================================
void MuscleController::do_build()
{
    m_constitution = require<Muscle>(QueryOptions{.exact = false});

    auto overrider = std::make_shared<MuscleControllerFeatureOverrider>(this);
    auto feature   = std::make_shared<core::MuscleControllerFeature>(overrider);
    features().insert(feature);
}

void MuscleController::copy_active_coeffs_from(geometry::SimplicialComplex& geo)
{
    auto attr = geo.tetrahedra().find<Float>("active_modulus");
    UIPC_ASSERT(attr,
                "[MuscleAccessor] Geometry has no 'active_modulus' attribute.");

    auto attr_view = view(*attr);
    UIPC_ASSERT(attr_view.size() == m_constitution->active_coeffs.size(),
                "[MuscleAccessor] Size mismatch: "
                "geometry has {} entries, device buffer has {}.",
                attr_view.size(),
                m_constitution->active_coeffs.size());

    // std::vector<Float> active_moduli;
    // active_moduli.assign(attr_view.begin(), attr_view.end());
    m_constitution->active_coeffs.view().copy_from(attr_view.data());
}



}  // namespace uipc::backend::cuda
