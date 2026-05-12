#pragma once
#include <sim_system.h>
#include <uipc/geometry/geometry.h>
#include <string>
#include <unordered_map>

namespace uipc::backend::cuda
{
class FEMExporter;

class FEMExporterManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  private:
    friend class FEMSystemFeatureOverrider;

    void do_build() override;
    void init();

    void get_fem_energy(U64 uid, geometry::Geometry& prim_energy);
    void get_fem_gradient(U64 uid, geometry::Geometry& prim_grad);
    void get_fem_hessian(U64 uid, geometry::Geometry& prim_hess);

    unordered_map<U64, FEMExporter*> m_exporter_map;
    SimSystemSlotCollection<FEMExporter>     m_exporters;
    vector<U64>                      m_uids;

    friend class FEMExporter;
    void add_exporter(FEMExporter* exporter);

    FEMExporter* find_exporter(U64 uid) const;
};
}  // namespace uipc::backend::cuda