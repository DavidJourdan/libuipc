#include <finite_element/constitutions/stable_neo_hookean_3d_function.h>
#include <finite_element/fem_3d_constitution.h>
#include <finite_element/fem_exporter.h>
#include <finite_element/fem_utils.h>
#include <kernel_cout.h>
#include <cuda_tool/cuda_tool.h>
#include <Eigen/Dense>
#include <utils/matrix_assembler.h>

namespace uipc::backend::cuda
{
namespace
{
    // Stiff-GIPC SNK1 (energy, gradient and the analytically SPD-projected
    // Hessian) replaces the SymEigen-generated SNH + generic make_spd EVD
    namespace SNH = snk1;

    constexpr SizeT StencilSize     = 4;
    constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    __global__ void StableNeoHookean3D_do_compute_energy_kernel(
        cuda_tool::CBufferView<Float>     mus,
        cuda_tool::CBufferView<Float>     lambdas,
        cuda_tool::BufferView<Float>      energies,
        cuda_tool::CBufferView<Vector4i>  indices,
        cuda_tool::CBufferView<Vector3>   xs,
        cuda_tool::CBufferView<Matrix3x3> Dm_invs,
        cuda_tool::CBufferView<Float>     volumes,
        Float                             dt,
        int                               n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        const Vector4i&  tet    = indices(I);
        const Matrix3x3& Dm_inv = Dm_invs(I);
        Float            mu     = mus(I);
        Float            lambda = lambdas(I);

        const Vector3& x0 = xs(tet(0));
        const Vector3& x1 = xs(tet(1));
        const Vector3& x2 = xs(tet(2));
        const Vector3& x3 = xs(tet(3));

        auto F = fem::F(x0, x1, x2, x3, Dm_inv);

        Float E;

        SNH::E(E, mu, lambda, F);
        E *= dt * dt * volumes(I);
        energies(I) = E;
    }

    __global__ void StableNeoHookean3D_do_compute_gradient_hessian_kernel(
        cuda_tool::CBufferView<Float>          mus,
        cuda_tool::CBufferView<Float>          lambdas,
        cuda_tool::CBufferView<Vector4i>       indices,
        cuda_tool::CBufferView<Vector3>        xs,
        cuda_tool::CBufferView<Matrix3x3>      Dm_invs,
        cuda_tool::DoubletVectorView<Float, 3> G3s,
        cuda_tool::TripletMatrixView<Float, 3> H3x3s,
        cuda_tool::CBufferView<Float>          volumes,
        Float                                  dt,
        bool                                   gradient_only,
        int                                    n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        const Vector4i&  tet    = indices(I);
        const Matrix3x3& Dm_inv = Dm_invs(I);
        Float            mu     = mus(I);
        Float            lambda = lambdas(I);

        const Vector3& x0 = xs(tet(0));
        const Vector3& x1 = xs(tet(1));
        const Vector3& x2 = xs(tet(2));
        const Vector3& x3 = xs(tet(3));

        auto F = fem::F(x0, x1, x2, x3, Dm_inv);

        auto Vdt2 = volumes(I) * dt * dt;

        Matrix3x3 dEdF;
        SNH::dEdF(dEdF, mu, lambda, F);
        auto VecdEdF = flatten(dEdF);
        VecdEdF *= Vdt2;

        auto shape_gradients = fem::tetrahedron_shape_gradients(Dm_inv);

        DoubletVectorAssembler DVA{G3s};
#pragma unroll
        for(int i = 0; i < StencilSize; ++i)
        {
            Vector3 G = fem::project_F_gradient(shape_gradients.col(i), VecdEdF);
            DVA(I * StencilSize + i).write(tet(i), G);
        }

        if(gradient_only)
            return;

        // analytically SPD-projected 9x9 energy Hessian (Stiff SNK1);
        // scaling by the positive Vdt2 commutes with the projection
        Matrix9x9 ddEddF;
        SNH::ddEddF_spd(ddEddF, mu, lambda, F);
        ddEddF *= Vdt2;

        IndexT hessian_offset = I * HalfHessianSize;
#pragma unroll
        for(int i = 0; i < StencilSize; ++i)
        {
#pragma unroll
            for(int j = i; j < StencilSize; ++j)
            {
                int left  = i;
                int right = j;
                if(tet(left) > tet(right))
                {
                    left  = j;
                    right = i;
                }

                Matrix3x3 H = fem::project_F_hessian_block(
                    shape_gradients.col(left), ddEddF, shape_gradients.col(right));
                H3x3s(hessian_offset++).write(tet(left), tet(right), H);
            }
        }
    }
}  // namespace

class StableNeoHookean3D final : public FEM3DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 10;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DConstitution::FEM3DConstitution;

    vector<Float> h_mus;
    vector<Float> h_lambdas;

    cuda_tool::DeviceBuffer<Float> mus;
    cuda_tool::DeviceBuffer<Float> lambdas;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(mus.size());
        info.gradient_count(mus.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(mus.size() * HalfHessianSize);
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_mus.resize(N);
        h_lambdas.resize(N);

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) -> auto
            {
                auto mu     = sc.tetrahedra().find<Float>("mu");
                auto lambda = sc.tetrahedra().find<Float>("lambda");

                return zip(mu->view(), lambda->view());
            },
            [&](const ForEachInfo& I, auto mu_and_lambda)
            {
                auto&& [mu, lambda] = mu_and_lambda;

                auto vI = I.global_index();

                h_mus[vI]     = mu;
                h_lambdas[vI] = lambda;
            });

        mus.resize(N);
        mus.view().copy_from(h_mus.data());

        lambdas.resize(N);
        lambdas.view().copy_from(h_lambdas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        auto k = StableNeoHookean3D_do_compute_energy_kernel;
        int  n = (int)info.indices().size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                mus.cview(),
                lambdas.cview(),
                info.energies(),
                info.indices(),
                info.xs(),
                info.Dm_invs(),
                info.rest_volumes(),
                info.dt(),
                n);
        }
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        auto k = StableNeoHookean3D_do_compute_gradient_hessian_kernel;
        int  n = (int)info.indices().size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                mus.cview(),
                lambdas.cview(),
                info.indices(),
                info.xs(),
                info.Dm_invs(),
                info.gradients(),
                info.hessians(),
                info.rest_volumes(),
                info.dt(),
                info.gradient_only(),
                n);
        }
    }
};

REGISTER_SIM_SYSTEM(StableNeoHookean3D);

class StableNeoHookean3DFEMExporter final : public FEMExporter
{
  public:
    using FEMExporter::FEMExporter;

    SimSystemSlot<FEM3DConstitution> fem_constitution;

    // ------------------------------------------------------------------
    // Identity — must match uipc::constitution::StableNeoHookean::constitution_uid()
    // Grep: src/constitution/stable_neo_hookean.cpp for constitution_uid()
    // ------------------------------------------------------------------
    U64 get_uid() const noexcept override
    {
        return StableNeoHookean3D::ConstitutionUID;
    }

    void do_build(BuildInfo&) override
    {
        fem_constitution = require<StableNeoHookean3D>(QueryOptions{.exact = false});
    }

    // ------------------------------------------------------------------
    // Energy  —  one Float per tet, plus tet topology
    //
    // Geometry layout after call:
    //   energy_geo.instances()[t] = { "topo": Vector4i, "energy": Float }
    //   t = 0 … N_tets-1
    // ------------------------------------------------------------------
    void get_fem_energy(geometry::Geometry& energy_geo) override
    {
        auto indices = fem_constitution->element_indices();  // CBufferView<Vector4i>
        auto energies = fem_constitution->element_energies();  // CBufferView<Float>

        energy_geo.instances().resize(indices.size());

        // Topology
        auto topo = energy_geo.instances().find<Vector4i>("topo");
        if(!topo)
            topo = energy_geo.instances().create<Vector4i>("topo", Vector4i::Zero());
        auto topo_view = view(*topo);
        indices.copy_to(topo_view.data());

        // Per-element energy
        auto energy = energy_geo.instances().find<Float>("energy");
        if(!energy)
            energy = energy_geo.instances().create<Float>("energy", 0.0f);
        auto energy_view = view(*energy);
        energies.copy_to(energy_view.data());
    }

    // ------------------------------------------------------------------
    // Gradient — doublet set: (i: IndexT, grad: Vector3)
    //   N_doublets = 4 * N_tets  (one per tet-corner vertex)
    //   Elastic force on vertex i = -sum of all grad entries with index i
    // ------------------------------------------------------------------
    void get_fem_gradient(geometry::Geometry& grad_geo) override
    {
        auto grads = fem_constitution->element_gradients();  // CDoubletVectorView<Float,3>

        grad_geo.instances().resize(grads.doublet_count());

        auto i = grad_geo.instances().find<IndexT>("i");
        if(!i)
            i = grad_geo.instances().create<IndexT>("i", -1);
        auto i_view = view(*i);
        grads.indices().copy_to(i_view.data());

        auto grad = grad_geo.instances().find<Vector3>("grad");
        if(!grad)
            grad = grad_geo.instances().create<Vector3>("grad", Vector3::Zero());
        auto grad_view = view(*grad);
        grads.values().copy_to(grad_view.data());
    }

    // ------------------------------------------------------------------
    // Hessian — triplet set: (i: IndexT, j: IndexT, hess: Matrix3x3)
    //   N_triplets = 16 * N_tets  (4×4 blocks per tet, off-diagonal included)
    // ------------------------------------------------------------------
    void get_fem_hessian(geometry::Geometry& hess_geo) override
    {
        auto hess = fem_constitution->element_hessians();  // CTripletMatrixView<Float,3>

        hess_geo.instances().resize(hess.triplet_count());

        auto i = hess_geo.instances().find<IndexT>("i");
        if(!i)
            i = hess_geo.instances().create<IndexT>("i", -1);
        auto i_view = view(*i);
        hess.row_indices().copy_to(i_view.data());

        auto j = hess_geo.instances().find<IndexT>("j");
        if(!j)
            j = hess_geo.instances().create<IndexT>("j", -1);
        auto j_view = view(*j);
        hess.col_indices().copy_to(j_view.data());

        auto h = hess_geo.instances().find<Matrix3x3>("hess");
        if(!h)
            h = hess_geo.instances().create<Matrix3x3>("hess", Matrix3x3::Zero());
        auto h_view = view(*h);
        hess.values().copy_to(h_view.data());
    }
};

REGISTER_SIM_SYSTEM(StableNeoHookean3DFEMExporter);
}  // namespace uipc::backend::cuda
