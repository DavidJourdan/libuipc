#pragma once
#include <sim_system.h>

namespace uipc::backend::cuda
{
// is forward decl needed?
// class FEMExporterManager;
class FEMExporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    // std::string_view prim_type() const noexcept; // probably not needed

    class BuildInfo
    {
      public:
    };

  /** Constitution UID this exporter handles, e.g. "StableNeoHookean". */
  std::string_view uid() const noexcept;

  protected:
    virtual void             do_build(BuildInfo& info)      = 0;
    virtual std::string_view get_uid() const noexcept       = 0;
    // virtual std::string_view get_prim_type() const noexcept = 0;

    virtual void get_fem_energy(std::string_view    uid,
                                geometry::Geometry& energy_geo) = 0;

    virtual void get_fem_gradient(std::string_view    uid,
                                  geometry::Geometry& vert_grad) = 0;

    virtual void get_fem_hessian(std::string_view    uid,
                                 geometry::Geometry& vert_hess) = 0;

  private:
    virtual void do_build() override final;

    friend class FEMExporterManager;
    void fem_energy(std::string_view uid, geometry::Geometry& energy_geo);
    void fem_gradient(std::string_view uid, geometry::Geometry& vert_grad);
    void fem_hessian(std::string_view uid, geometry::Geometry& vert_hess);
};
}  // namespace uipc::backend::cuda
