#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/muscle.h>

int main()
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    // Engine engine{"none"};
    Engine engine{"cuda"};

    World world{engine};
    auto  config      = Scene::default_config();
    config["gravity"] = Vector3{0, -9.8, 0};
    config["dt"]      = 0.01_s;

    Scene scene{config};

    // create constitution and contact model
    StableNeoHookean snh;
    Muscle sna;

    // create a regular tetrahedron
    vector<Vector3>  Vs = {Vector3{0, 0, 1},
                            Vector3{0, -1, 0},
                            Vector3{-std::sqrt(3) / 2, 0, -0.5},
                            Vector3{std::sqrt(3) / 2, 0, -0.5}};
    vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};

    // setup a base mesh to reduce the later work
    SimplicialComplex base_mesh = tetmesh(Vs, Ts);
    // apply the constitution and contact model to the base mesh
    auto moduli = ElasticModuli::youngs_poisson(5.0_kPa, 0.49);
    snh.apply_to(base_mesh, moduli, 1e4);
    sna.apply_to(base_mesh, Vector3{0, 1, 0}, 100.0_kPa, 100.0_kPa);

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

    world.init(scene);

    SceneIO sio{scene};

    auto this_output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);

    sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, 0));

    for(int i = 1; i < 50; i++)
    {
        world.advance();
        world.sync();
        world.retrieve();
        sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, i));
    }

    // this does nothing unfortunately :(
    auto passive_modulus = base_mesh.tetrahedra().find<Float>("passive_modulus");
    auto passive_modulus_view = view(*passive_modulus);
    passive_modulus_view[0]   = 1;

    for(int i = 50; i < 100; i++)
    {
        world.advance();
        world.sync();
        world.retrieve();
        sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, i));
    }
}
