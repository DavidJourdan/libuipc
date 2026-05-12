#pragma once
#include <sim_system.h>
#include <uipc/core/fem_system_feature.h>

namespace uipc::backend::cuda
{
class FEMExporterManager;

class FEMSystemFeatureOverrider final : public core::FEMSystemFeatureOverrider
{
  public:
    FEMSystemFeatureOverrider(FEMExporterManager* fem_system);

  private:
    void get_fem_gradient(U64 uid, geometry::Geometry& vert_grad) override;
    void get_fem_hessian(U64 uid, geometry::Geometry& vert_hess) override;
    void get_fem_energy(U64 uid, geometry::Geometry& prims) override;

    SimSystemSlot<FEMExporterManager> m_manager;
};
}  // namespace uipc::backend::cuda