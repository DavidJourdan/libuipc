#pragma once
#include <sim_system.h>
#include <uipc/common/type_define.h>  // U64

namespace uipc::backend::cuda
{
class FEMExporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    /** Constitution UID this exporter handles, e.g. "StableNeoHookean". */
    U64 uid() const noexcept;

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual U64  get_uid() const noexcept  = 0;

    virtual void get_fem_energy(geometry::Geometry& energy_geo) = 0;

    virtual void get_fem_gradient(geometry::Geometry& vert_grad) = 0;

    virtual void get_fem_hessian(geometry::Geometry& vert_hess) = 0;

  private:
    virtual void do_build() override final;

    friend class FEMExporterManager;
    void fem_energy(geometry::Geometry& energy_geo);
    void fem_gradient(geometry::Geometry& vert_grad);
    void fem_hessian(geometry::Geometry& vert_hess);
};
}  // namespace uipc::backend::cuda
