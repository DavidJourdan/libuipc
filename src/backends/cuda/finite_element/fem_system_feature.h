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
    vector<std::string> get_fem_primitive_types() const override;

    void get_fem_gradient(std::string_view prim_type, geometry::Geometry& vert_grad) override;
    void get_fem_hessian(std::string_view prim_type, geometry::Geometry& vert_hess) override;
    void get_fem_energy(std::string_view prim_type, geometry::Geometry& prims) override;

    SimSystemSlot<FEMExporterManager> m_manager;
};
}  // namespace uipc::backend::cuda