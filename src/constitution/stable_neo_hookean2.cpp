#include <uipc/constitution/stable_neo_hookean2.h>
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
    uids.push_back(UIDInfo{.uid  = 88888,
                           .name = "StableNeoHookean2",
                           .type = string{builtin::FiniteElement}});
    return uids;
}

StableNeoHookean2::StableNeoHookean2(const Json& config) noexcept
    : m_config(config)
{
}

void StableNeoHookean2::apply_to(geometry::SimplicialComplex& sc,
                                const ElasticModuli&         moduli,
                                Float                        mass_density) const
{
    Base::apply_to(sc, mass_density);

    auto mu     = moduli.mu();
    auto lambda = moduli.lambda();

    auto snh_mu = 4 * mu / 3;
    auto snh_lambda = lambda + 5 * mu / 6;

    UIPC_ASSERT_THROW(sc.dim() == 3, "StableNeoHookean2 only supports 3D simplicial complex");

    auto mu_attr = sc.tetrahedra().find<Float>("mu");
    if(!mu_attr)
        mu_attr = sc.tetrahedra().create<Float>("mu", snh_mu);
    std::ranges::fill(geometry::view(*mu_attr), snh_mu);

    auto lambda_attr = sc.tetrahedra().find<Float>("lambda");
    if(!lambda_attr)
        lambda_attr = sc.tetrahedra().create<Float>("lambda", snh_lambda);
    std::ranges::fill(geometry::view(*lambda_attr), snh_lambda);
}

Json StableNeoHookean2::default_config() noexcept
{
    return Json::object();
}

U64 StableNeoHookean2::get_uid() const noexcept
{
    return 10;
}
}  // namespace uipc::constitution
