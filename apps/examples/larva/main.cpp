#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/muscle.h>
#include <uipc/core/muscle_controller_feature.h>
#include <uipc/builtin/geometry_type.h>
#include <fstream>
#include <iostream>
#include "read_activation_csv.h"
#include "mesh_export.h"
#include "segment_tracker.h"
#include "energy_logger.h"

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

int main(int argc, char* argv[])
{
    auto output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);
    auto filetype    = save_type::surface;

    // read first argument as config file path
    std::string config_file_path = CONFIG_FOLDER "hunch.json";  // default config file path
    if(argc > 1)
    {
        config_file_path = CONFIG_FOLDER + std::string(argv[1]);
    }

    // Read the JSON file
    std::ifstream inputFile(config_file_path);
    if(!inputFile.is_open())
    {
        std::cerr << "Error: Could not open JSON file." << std::endl;
        return 1;
    }

    // Parse the JSON file
    Json data;
    inputFile >> data;

    Engine engine{"cuda"};
    logger::set_level(spdlog::level::warn);

    World world{engine};

    // Setup config from JSON file
    auto config = Scene::default_config();

    config["dt"] = data.at("dt").get<double>();

    if(data.contains("gravity") && data["gravity"].get<bool>())
        config["gravity"] = Vector3{0, -9.8, 0};
    else
        config["gravity"] = Vector3{0, 0, 0};

    config["contact"]["d_hat"] = data.at("d_hat").get<double>();
    if(data.contains("kappa_eval_scale"))
        config["contact"]["adaptive"]["kappa_eval_scale"] =
            data.at("kappa_eval_scale").get<double>();

    Scene scene{config};

    // Create larva rest mesh
    SimplicialComplexIO io;
    SimplicialComplex   larva_rest_mesh =
        io.read(CONFIG_FOLDER + data.at("rest_geometry").get<std::string>());

    label_surface(larva_rest_mesh);
    label_triangle_orient(larva_rest_mesh);

    // copy consititution attributes to rest mesh
    StableNeoHookean snh_constitution;
    Muscle           muscle_constitution;

    double youngs_modulus = data.at("youngs_modulus").get<double>();
    double poisson_ratio  = data.at("poisson_ratio").get<double>();
    auto lamé_param = ElasticModuli::youngs_poisson(youngs_modulus, poisson_ratio);

    snh_constitution.apply_to(larva_rest_mesh, lamé_param);
    muscle_constitution.apply_to(larva_rest_mesh, data.at("muscle_passive").get<double>());

    if(data.contains("ground") && data["ground"].get<bool>())
    {
        // create the ground implicit geometry
        auto floor = ground(data.at("ground_height").get<double>());

        // set up contact
        scene.contact_tabular().default_model(0.5, 1.0_GPa);
        auto default_contact = scene.contact_tabular().default_element();

        default_contact.apply_to(larva_rest_mesh);
        default_contact.apply_to(floor);

        // add floor to the scene
        scene.objects().create("ground")->geometries().create(floor);
    }

    // Create the larva object and its geometry slot
    ObjectGeometrySlots<SimplicialComplex> larva_mesh_slot;
    if(data.contains("curr_geometry"))
    {
        // Create larva current mesh
        SimplicialComplex larva_curr_mesh_temp =
            io.read(CONFIG_FOLDER + data.at("curr_geometry").get<std::string>());
        SimplicialComplex larva_curr_mesh = larva_rest_mesh;
        larva_curr_mesh.vertices().copy_from(larva_curr_mesh_temp.vertices());

        auto object = scene.objects().create("larva");
        larva_mesh_slot = object->geometries().create(larva_curr_mesh, larva_rest_mesh);
    }
    else
    {  // Initialize current mesh as rest mesh duplicate
        auto object     = scene.objects().create("larva");
        larva_mesh_slot = object->geometries().create(larva_rest_mesh);
    }

    world.init(scene);

    auto muscle_controller = world.features().find<MuscleControllerFeature>();

    // ---- per-frame energy bookkeeping (FEM data export, issue #344) -----
    // Enabled by the `export_energy` key in the config; a no-op otherwise.
    EnergyLogger energy_logger;
    if(data.contains("export_energy") && data["export_energy"].get<bool>())
    {
        energy_logger = EnergyLogger{world,
                                     larva_rest_mesh,
                                     snh_constitution,
                                     muscle_constitution,
                                     data,
                                     fmt::format("{}fem_energy.csv", output_path)};
        energy_logger.write_row(0);  // rest pose (or the hunch start pose)
    }

    // are there any muscle activations?
    bool muscle_control = data.contains("muscle_active") && (data["muscle_active"] > 0)
                          && data.contains("muscle_contraction_pattern");

    if(muscle_control)
    {
        // first set active modulus and read muscle groups
        muscle_controller->set_active_modulus(data["muscle_active"]);
        muscle_controller->read_muscle_groups_from(larva_rest_mesh);

        // then read activations (order matters); muscles without a csv
        // column simply get a zero activation
        auto activation_path = data["muscle_contraction_pattern"].get<std::string>();
        auto [times, raw_activations] = read_activation_csv(CONFIG_FOLDER + activation_path);
        auto activations = resample_activations(times, raw_activations, config["dt"]);

        muscle_controller->read_activations(activations);
    }

    // ---- segment interfaces (optional) -------------------------------
    // When the config has a `segment_interfaces` key, every frame appends
    // the mean X of each interface and the segment lengths (and their
    // change vs frame 0) to segment_lengths.csv; a no-op otherwise.
    SegmentTracker segment_tracker;
    if(data.contains("segment_interfaces"))
    {
        segment_tracker =
            SegmentTracker{CONFIG_FOLDER + data.at("segment_interfaces").get<std::string>(),
                           fmt::format("{}segment_lengths.csv", output_path)};
    }

    // 2D / 3D mesh export
    MeshExporter mesh_exporter(larva_mesh_slot, filetype);
    mesh_exporter.save_vertices_binary(0);
    mesh_exporter.save_scene(0);  // static topology for the viewer (frame 0 obj)
    mesh_exporter.muscle_groups_to_json(fmt::format("{}muscle_groups.json", output_path));
    segment_tracker.write_row(larva_mesh_slot.geometry->geometry(), 0);

    int nb_frames = 0;
    if(muscle_control)
        nb_frames = muscle_controller->nb_frames();
    else if(data.contains("nb_frames"))
        nb_frames = data["nb_frames"];

    for(int i = 0; i < nb_frames; i++)
    {
        auto start = std::chrono::system_clock::now();
        if(muscle_control)
            muscle_controller->update_next_frame(larva_mesh_slot.geometry->geometry(), i);

        world.advance();
        world.sync();
        world.retrieve();

        energy_logger.write_row(i + 1);

        // mesh_exporter.save_scene(i + 1);
        mesh_exporter.save_vertices_binary(i + 1);
        segment_tracker.write_row(larva_mesh_slot.geometry->geometry(), i + 1);

        auto end = std::chrono::system_clock::now();
        std::cout << fmt::format(
            "Iteration {}: {} ms elapsed.\n",
            i + 1,
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }

    energy_logger.finish();

    // Only runs that start from the rest pose (no `curr_geometry`, i.e. drop)
    // export a hand-off `larva_curr.msh`; runs that *consumed* one (hunch)
    // would otherwise leave their deformed end-of-run pose in the output
    // folder, one careless `cp` away from clobbering the committed file.
    if(!data.contains("curr_geometry"))
    {
        save_msh(fmt::format("{}larva_curr.msh", output_path),
                 larva_mesh_slot.geometry->geometry());
    }
}
