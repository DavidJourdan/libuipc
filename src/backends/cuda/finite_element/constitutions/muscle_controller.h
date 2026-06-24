#pragma once
#include <sim_system.h>
#include <uipc/core/muscle_controller_feature.h>
#include <finite_element/constitutions/muscle.h>

namespace uipc::backend::cuda
{
class MuscleController;

class MuscleControllerFeatureOverrider final : public core::MuscleControllerFeatureOverrider
{
  public:
    MuscleControllerFeatureOverrider(MuscleController* accessor);

  private:
    void do_copy_active_coeffs_from(geometry::SimplicialComplex& geo) override;
    SimSystemSlot<MuscleController> m_accessor;
};

class MuscleController final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  private:
    friend class MuscleControllerFeatureOverrider;
    SimSystemSlot<Muscle> m_constitution;
    void do_build() override;
    void copy_active_coeffs_from(geometry::SimplicialComplex& geo);
};
}  // namespace uipc::backend::cuda
