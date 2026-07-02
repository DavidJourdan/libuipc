#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/soft_position_constraint.h>
#include <uipc/constitution/muscle.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <uipc/core/fem_system_feature.h>
#include <uipc/common/type_define.h>
#include <fstream>
#include <iostream>


using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

Matrix3x3 Ds(const Vector3& x0, const Vector3& x1, const Vector3& x2, const Vector3& x3)
{
    Matrix3x3 Ds;
    Ds.col(0) = x1 - x0;
    Ds.col(1) = x2 - x0;
    Ds.col(2) = x3 - x0;
    return Ds;
}

// void printSpan(std::span<int> sp) {
//     for (auto&& elem : sp)
//         std::cout << elem << ' ';
//     std::cout << '\n';
// }

int main()
{
    Engine engine{"cuda"};
    logger::set_level(spdlog::level::warn);

    World world{engine};
    auto  config      = Scene::default_config();
    config["gravity"] = Vector3{0, -9.8, 0};
    config["dt"]      = 0.01_s;

    Scene scene{config};

    // create constitution and contact model
    StableNeoHookean snh_constitution;
    Muscle           muscle_constitution;
    SoftPositionConstraint spc;

    // create a regular tetrahedron
    vector<Vector3>  Vs = {Vector3{0, 0, 1},
                           Vector3{0, -1, 0},
                           Vector3{-std::sqrt(3) / 2, 0, -0.5},
                           Vector3{std::sqrt(3) / 2, 0, -0.5}};

    Matrix3x3 DmInv = Ds(Vs[0], Vs[1], Vs[2], Vs[3]).inverse();

    vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};

    // setup a base mesh to reduce the later work
    SimplicialComplex base_mesh = tetmesh(Vs, Ts);
    // apply the constitution and contact model to the base mesh
    auto moduli = ElasticModuli::youngs_poisson(5.0_kPa, 0.49);
    snh_constitution.apply_to(base_mesh, moduli);
    spc.apply_to(base_mesh, 100.0);  // constraint strength ratio
    muscle_constitution.apply_to(base_mesh, Vector3{0, 1, 0}, 200.0_kPa, 100.0_kPa);

    // label the surface, enable the contact
    label_surface(base_mesh);
    // label the triangle orientation to export the correct surface mesh
    label_triangle_orient(base_mesh);

    // find the is_fixed attribute
    auto is_fixed = base_mesh.vertices().find<IndexT>(builtin::is_fixed);
    // set the first instance to be fixed
    auto is_fixed_view = view(*is_fixed);
    is_fixed_view[0]   = 1;
    is_fixed_view[2]   = 1;
    is_fixed_view[3]   = 1;

    auto object = scene.objects().create("tets");
    object->geometries().create(base_mesh);

    auto& animator = scene.animator();
    animator.insert(
        *object,
        [](Animation::UpdateInfo& info)
        {
            auto geo_slots = info.geo_slots();
            auto geo = geo_slots[0]->geometry().as<SimplicialComplex>();
            auto rest_geo_slots = info.rest_geo_slots();
            auto rest_geo = rest_geo_slots[0]->geometry().as<SimplicialComplex>();

            auto is_constrained = geo->vertices().find<IndexT>(builtin::is_constrained);
            auto is_constrained_view = view(*is_constrained);
            auto aim_position = geo->vertices().find<Vector3>(builtin::aim_position);
            auto aim_position_view = view(*aim_position);
            auto rest_position_view = rest_geo->positions().view();

            is_constrained_view[1]   = 1;

            auto t = info.dt() * info.frame();
            auto theta = 2 * std::numbers::pi * t;
            auto y = -std::sin(theta);

            aim_position_view[1] = rest_position_view[1] + Vector3::UnitY() * y;
        });


    world.init(scene);
    SceneIO sio{scene};

    auto this_output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);

    sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, 0));


    // ------------------------------------------------------------------
    // Open CSV file and write header
    // ------------------------------------------------------------------
    std::ofstream csv{this_output_path + "fem_energy.csv"};
    UIPC_ASSERT(csv.is_open(), "Failed to open fem_energy.csv for writing.");

    csv << "frame,"
        << "lambda,"
        << "energy,"
        << "force,"
        << "position,"
        << "hessian"
        << "\n";

    auto fem_system = world.features().find<FEMSystemFeature>();
    auto fem_state_accessor = world.features().find<FiniteElementStateAccessorFeature>();

    auto fem_state_geo = fem_state_accessor->create_geometry();
    auto fem_positions = fem_state_geo.vertices().create<Vector3>(builtin::position, Vector3::Zero());
    auto fem_velocities = fem_state_geo.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());

    for(int i = 1; i < 100; i++)
    {
        world.advance();
        world.sync();
        world.retrieve();
        sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, i));

        fem_state_accessor->copy_to(fem_state_geo);

        Geometry snh_geo, muscle_geo, hessian_geo;
        fem_system->fem_energy(snh_constitution.uid(), snh_geo);
        fem_system->fem_energy(muscle_constitution.uid(), muscle_geo);
        fem_system->fem_gradient(muscle_constitution.uid(), muscle_geo);
        fem_system->fem_hessian(muscle_constitution.uid(), hessian_geo);

        auto snh_energy_attr = snh_geo.instances().find<Float>("energy");
        auto muscle_energy_attr = muscle_geo.instances().find<Float>("energy");
        auto muscle_grad_attr = muscle_geo.instances().find<Vector3>("grad");
        auto muscle_hessian_attr = hessian_geo.instances().find<Matrix3x3>("hess");

        auto energies = view(*muscle_energy_attr);

        auto V = view(*fem_state_geo.vertices().find<Vector3>(builtin::position));

        Matrix3x3 F = Ds(V[0], V[1], V[2], V[3]) * DmInv;
        Vector3 a {0,1,0};
        Float lambda = (F * a).norm();

        Vector3 force = -view(*muscle_grad_attr)[1];
        Matrix3x3 hess = view(*muscle_hessian_attr)[4];

        // auto rows  = view(*hessian_geo.instances().find<IndexT>("i"));
        // auto cols  = view(*hessian_geo.instances().find<IndexT>("j"));

        // printSpan(rows);
        // printSpan(cols);

        fmt::print("[{:2d}] Lambda: {:.3e} | Energy: {:.3e} J | Force: {:.3e} {:.3e} {:.3e} | Hess: {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} | Pos: {}\n",
                   i,
                   lambda,
                   view(*muscle_energy_attr)[0],
                   force(0), force(1), force(2),
                   hess(0, 0), hess(0, 1), hess(0, 2),
                   hess(1, 0), hess(1, 1), hess(1, 2),
                   hess(2, 0), hess(2, 1), hess(2, 2),
                   V[1][1]);

        csv << fmt::format("{}, {}, {}, {}, {}, {}\n", i, lambda, view(*muscle_energy_attr)[0], force(1), V[1][1], hess(1,1));

        // topos    = np.array(geo_e.instances().find("topo").view())    # (N_tet, 4)
        // energies = np.array(geo_e.instances().find("energy").view())  # (N_tet,)
        // print(f"[{frame}] Total elastic PE = {energies.sum():.6e} J")
    }

    csv.close();

    for(int i = 0; i < 100; i++)
    {
        world.advance();
        world.sync();
        world.retrieve();
        sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, i));
    }

}
