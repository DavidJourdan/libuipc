#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/muscle.h>
#include <uipc/core/muscle_controller_feature.h>
#include <uipc/builtin/geometry_type.h>
#include <fstream>
#include <iostream>
#include "read_activation_csv.h"
#include "save_msh.h"

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

int main()
{
    // Path to the JSON file
    std::string jsonFilePath = CONFIG_FOLDER "config.json";  // Replace with your JSON file path

    // Read the JSON file
    std::ifstream inputFile(jsonFilePath);
    if(!inputFile.is_open())
    {
        std::cerr << "Error: Could not open JSON file." << std::endl;
        return 1;
    }

    // Parse the JSON file
    Json data;
    try
    {
        inputFile >> data;
    }
    catch(const Json::parse_error& e)
    {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return 1;
    }

    Engine engine{"cuda"};
    logger::set_level(spdlog::level::warn);

    World world{engine};
    auto  config = Scene::default_config();

    double muscle_passive, muscle_active, youngs_modulus, poisson_ratio;
    std::unordered_map<int, std::vector<float>> activations;
    std::string                                 larva_mesh_path;

    try
    {
        larva_mesh_path = data.at("larva_mesh").get<std::string>();

        config["dt"] = data.at("dt").get<double>();

        bool gravity = data.at("gravity").get<bool>();
        if(gravity)
            config["gravity"] = Vector3{0, -9.8, 0};
        else
            config["gravity"] = Vector3{0, 0, 0};

        config["contact"]["d_hat"] = data.at("d_hat").get<double>();

        std::string muscle_contraction_pattern_path =
            data.at("muscle_contraction_pattern").get<std::string>();
        auto [times, raw_activations] =
            read_activation_csv(CONFIG_FOLDER + muscle_contraction_pattern_path);
        activations = resample_activations(times, raw_activations, config["dt"]);

        muscle_active  = data.at("muscle_active").get<double>();
        youngs_modulus = data.at("youngs_modulus").get<double>();
        poisson_ratio  = data.at("poisson_ratio").get<double>();
        muscle_passive = data.at("muscle_passive").get<double>();
    }
    catch(const Json::exception& e)
    {
        std::cerr << "Error accessing JSON fields: " << e.what() << std::endl;
        return 1;
    }

    Scene scene{config};

    SimplicialComplexIO io;
    SimplicialComplex   larva_mesh = io.read(CONFIG_FOLDER + larva_mesh_path);

    StableNeoHookean snh_constitution;
    Muscle           muscle_constitution;
    snh_constitution.apply_to(larva_mesh,
                              ElasticModuli::youngs_poisson(youngs_modulus, poisson_ratio));
    muscle_constitution.apply_to(larva_mesh, muscle_passive);

    // label the surface, enable the contact
    label_surface(larva_mesh);
    // label the triangle orientation to export the correct surface mesh
    label_triangle_orient(larva_mesh);
    
    auto object = scene.objects().create("larva");
    auto larva_mesh_slot = object->geometries().create(larva_mesh);
    UIPC_ASSERT(larva_mesh_slot.geometry->geometry().type() == builtin::SimplicialComplex,
        "larva_mesh is not a simplicial complex");

    world.init(scene);
    auto muscle_controller = world.features().find<MuscleControllerFeature>();

    muscle_controller->set_active_modulus(muscle_active);
    muscle_controller->read_muscle_groups_from(larva_mesh);
    muscle_controller->read_activations(activations);

    auto    this_output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);
    save_msh(fmt::format("{}scene_mesh{}.msh", this_output_path, 0), larva_mesh_slot.geometry->geometry());

    SimplicialComplex mesh = io.read(fmt::format("{}scene_mesh{}.msh", this_output_path, 0));
    auto active_modulus = mesh.tetrahedra().find<double>("active_modulus");
    auto active_modulus_view = active_modulus->view();
    std::cout << active_modulus_view[0] << "\n";

    for(int i = 0; i < muscle_controller->nb_frames(); i++)
    {
        muscle_controller->update_next_frame(larva_mesh_slot.geometry->geometry(), i);
        world.advance();
        world.sync();
        world.retrieve();
        save_msh(fmt::format("{}scene_mesh{}.msh", this_output_path, i + 1), larva_mesh_slot.geometry->geometry());
    }
}
