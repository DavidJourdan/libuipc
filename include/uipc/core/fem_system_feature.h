#pragma once
#include <uipc/core/feature.h>
#include <uipc/backend/buffer_view.h>
#include <uipc/geometry/geometry.h>
#include <uipc/constitution/constitution.h>

namespace uipc::core
{
class UIPC_CORE_API FEMSystemFeatureOverrider
{
  public:
    virtual void get_fem_energy(U64 uid, geometry::Geometry& energy_geo) = 0;

    virtual void get_fem_gradient(U64 uid, geometry::Geometry& vert_grad) = 0;

    virtual void get_fem_hessian(U64 uid, geometry::Geometry& vert_hess) = 0;
};

class UIPC_CORE_API FEMSystemFeature final : public Feature
{
  public:
    constexpr static std::string_view FeatureName = "core/fem_system";

    FEMSystemFeature(S<FEMSystemFeatureOverrider> overrider);

    void fem_energy(U64 uid, geometry::Geometry& energy);

    void fem_gradient(U64 uid, geometry::Geometry& vert_grad);

    void fem_hessian(U64 uid, geometry::Geometry& vert_hess);

    void fem_energy(const constitution::IConstitution& c, geometry::Geometry& energy);

    void fem_gradient(const constitution::IConstitution& c, geometry::Geometry& vert_grad);

    void fem_hessian(const constitution::IConstitution& c, geometry::Geometry& vert_hess);

  private:
    virtual std::string_view     get_name() const override;
    S<FEMSystemFeatureOverrider> m_impl;
};
}  // namespace uipc::core