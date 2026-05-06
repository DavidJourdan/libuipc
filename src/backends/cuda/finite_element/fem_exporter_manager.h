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

    void get_fem_energy(std::string_view uid, geometry::Geometry& prim_energy);
    void get_fem_gradient(std::string_view uid, geometry::Geometry& prim_grad);
    void get_fem_hessian(std::string_view uid, geometry::Geometry& prim_hess);

    // vector<std::string> get_fem_primitive_types() const;

    unordered_map<std::string, FEMExporter*> m_exporter_map;
    SimSystemSlotCollection<FEMExporter>     m_exporters;
    vector<std::string>                      m_uids;

    friend class FEMExporter;
    void add_exporter(FEMExporter* exporter);

    FEMExporter* find_exporter(std::string_view uid) const;
    // void _create_uid_on_geo(std::string_view uid, geometry::Geometry& geo);
};
}  // namespace uipc::backend::cuda