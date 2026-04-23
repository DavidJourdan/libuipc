#include <uipc/constitution/stable_anisotropic_arap.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/constitution/conversion.h>
#include <uipc/common/log.h>

namespace uipc::constitution
{
REGISTER_CONSTITUTION_UIDS()
{
    using namespace uipc::builtin;
    list<UIDInfo> uids;
    uids.push_back(UIDInfo{.uid  = 33,
                           .name = "StableAnisotropicARAP",
                           .type = string{builtin::FiniteElement}});
    return uids;
}

StableAnisotropicARAP::StableAnisotropicARAP(const Json& config) noexcept
    : m_config(config)
{
}

void StableAnisotropicARAP::apply_to(geometry::SimplicialComplex& sc, Vector3 direction, Float mu) const
{
    UIPC_ASSERT(sc.dim() == 3, "StableAnisotropicARAP only supports 3D simplicial complex");

    auto mu_attr = sc.tetrahedra().find<Float>("mu");
    if(!mu_attr)
        mu_attr = sc.tetrahedra().create<Float>("mu", mu);
    std::ranges::fill(geometry::view(*mu_attr), mu);

    auto direction_attr = sc.tetrahedra().find<Vector3>("direction");
    if(!direction_attr)
        direction_attr = sc.tetrahedra().create<Vector3>("direction", direction);
    std::ranges::fill(geometry::view(*direction_attr), direction);
}

Json StableAnisotropicARAP::default_config() noexcept
{
    return Json::object();
}

U64 StableAnisotropicARAP::get_uid() const noexcept
{
    return 33;
}
}  // namespace uipc::constitution
