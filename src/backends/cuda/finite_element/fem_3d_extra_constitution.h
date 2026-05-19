#pragma once
#include <finite_element/finite_element_extra_constitution.h>

namespace uipc::backend::cuda
{
/**
 * @brief Extra constitution base class for 3D (tetrahedral) finite elements.
 *
 * Analogous to FEM3DConstitution, but for constitutions referenced via
 * builtin::extra_constitution_uids rather than the geometry's primary UID.
 *
 * Subclasses must implement:
 *   get_uid()
 *   do_build(BuildInfo&)
 *   do_init(FilteredInfo&)                    ← inherited from parent
 *   do_report_extent(ReportExtentInfo&)       ← inherited from parent
 *   do_compute_energy(ComputeEnergyInfo&)
 *   do_compute_gradient_hessian(ComputeGradientHessianInfo&)
 *
 * During do_report_extent the subclass may call the protected geo_infos()
 * method (inherited from FiniteElementExtraConstitution) to sum up primitive
 * counts across all matched geometries.
 */
class FEM3DExtraConstitution : public FiniteElementExtraConstitution
{
  public:
    using FiniteElementExtraConstitution::FiniteElementExtraConstitution;

    muda::CBufferView<Vector4i>        element_indices()   const noexcept;
    muda::CBufferView<Float>           element_energies()  const noexcept;
    muda::CDoubletVectorView<Float, 3> element_gradients() const noexcept;
    muda::CTripletMatrixView<Float, 3> element_hessians()  const noexcept;

    class BuildInfo
    {
      public:
    };

    class BaseInfo
    {
      public:
        BaseInfo(FEM3DExtraConstitution* impl, Float dt) noexcept
            : m_impl(impl)
            , m_dt(dt)
        {
        }

        Float dt() const noexcept { return m_dt; }

        /**
         * @brief GeoInfo for every geometry carrying this extra-constitution
         *        UID.  Use primitive_offset / primitive_count to slice the
         *        full-buffer accessors below.
         */
        span<const FiniteElementMethod::GeoInfo> geo_infos() const noexcept;

        // Full global position buffer.
        muda::CBufferView<Vector3> xs() const noexcept;

        // Full global rest-position buffer.
        muda::CBufferView<Vector3> x_bars() const noexcept;

        // Full global tet-index buffer (Vector4i).
        muda::CBufferView<Vector4i> indices() const noexcept;

        // Full global Dm-inverse buffer (Matrix3x3).
        muda::CBufferView<Matrix3x3> Dm_invs() const noexcept;

        // Full global rest-volume buffer.
        muda::CBufferView<Float> rest_volumes() const noexcept;

      protected:
        FEM3DExtraConstitution* m_impl = nullptr;
        Float                   m_dt   = 0.0;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(FEM3DExtraConstitution*              impl,
                          FiniteElementExtraConstitution::ComputeEnergyInfo* base_info,
                          Float                                              dt)
            : BaseInfo(impl, dt)
            , m_base_info(base_info)
        {
        }

        auto energies() const noexcept { return m_base_info->energies(); }

      private:
        FiniteElementExtraConstitution::ComputeEnergyInfo* m_base_info = nullptr;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        ComputeGradientHessianInfo(
            FEM3DExtraConstitution*                                       impl,
            FiniteElementExtraConstitution::ComputeGradientHessianInfo*   base_info,
            Float                                                         dt)
            : BaseInfo(impl, dt)
            , m_base_info(base_info)
        {
        }

        auto gradient_only() const noexcept { return m_base_info->gradient_only(); }
        auto gradients()     const noexcept { return m_base_info->gradients(); }
        auto hessians()      const noexcept { return m_base_info->hessians(); }

      private:
        FiniteElementExtraConstitution::ComputeGradientHessianInfo* m_base_info = nullptr;
    };

  protected:
    virtual void do_build(BuildInfo& info)                                     = 0;
    virtual void do_compute_energy(ComputeEnergyInfo& info)                    = 0;
    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) = 0;

  private:
    virtual void do_build(
        FiniteElementExtraConstitution::BuildInfo& info) override final;

    virtual void do_compute_energy(
        FiniteElementExtraConstitution::ComputeEnergyInfo& info) override final;

    virtual void do_compute_gradient_hessian(
        FiniteElementExtraConstitution::ComputeGradientHessianInfo& info) override final;

    muda::CBufferView<Vector4i>        m_element_indices;
    muda::CBufferView<Float>           m_element_energies;
    muda::CDoubletVectorView<Float, 3> m_element_gradients;
    muda::CTripletMatrixView<Float, 3> m_element_hessians;
};
}  // namespace uipc::backend::cuda