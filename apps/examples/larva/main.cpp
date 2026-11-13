#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/muscle.h>
#include <uipc/core/muscle_controller_feature.h>
#include <uipc/builtin/geometry_type.h>
#include <uipc/common/enumerate.h>
#include <fstream>
#include <iostream>
#include <tuple>
#include <array>
#include <algorithm>
#include <unordered_set>
#include "read_activation_csv.h"
#include "mesh_export.h"

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

    // are there any muscle activations?
    bool muscle_control = data.contains("muscle_active") && (data["muscle_active"] > 0)
                          && data.contains("muscle_contraction_pattern");

    if(muscle_control)
    {
        // first set active modulus and read muscle groups
        muscle_controller->set_active_modulus(data["muscle_active"]);
        muscle_controller->read_muscle_groups_from(larva_rest_mesh);

        // then read activations (order matters)
        auto activation_path = data["muscle_contraction_pattern"].get<std::string>();
        auto [times, raw_activations] = read_activation_csv(CONFIG_FOLDER + activation_path);
        auto activations = resample_activations(times, raw_activations, config["dt"]);

        // collect the muscle ids present in the mesh
        auto muscle_id_attr = larva_rest_mesh.tetrahedra().find<Float>("muscle_id");
        UIPC_ASSERT(muscle_id_attr, "Rest mesh has no 'muscle_id' attribute.");
        std::unordered_set<int> mesh_muscle_ids;
        for(Float muscle_id : muscle_id_attr->view())
        {
            if(muscle_id > 0)
                mesh_muscle_ids.insert(static_cast<int>(muscle_id));
        }

        // muscles missing from the csv get zero activation,
        // csv columns of muscles missing from the mesh are dropped
        auto [zero_activated, dropped_columns] =
            match_activations_to_muscles(activations, mesh_muscle_ids);
        if(!zero_activated.empty())
        {
            std::cout << fmt::format(
                "Note: {} muscles have no activation data, they get a zero activation: {}\n",
                zero_activated.size(),
                fmt::join(zero_activated, ", "));
        }
        if(!dropped_columns.empty())
        {
            std::cout << fmt::format(
                "Note: dropped activation data of {} muscles absent from the mesh: {}\n",
                dropped_columns.size(),
                fmt::join(dropped_columns, ", "));
        }

        muscle_controller->read_activations(activations);
    }

    // ---- segment interfaces (optional) -------------------------------
    // Groups of vertex indices sitting exactly on the segment boundary
    // planes of the rest mesh (exported by larva-bodies' generate_tet_mesh
    // alongside the mesh).  Their mean X position is written to
    // segment_lengths.csv every frame, giving the segment lengths and their
    // change relative to frame 0 during the simulation.
    std::vector<std::tuple<std::string, std::vector<IndexT>>> interfaces;
    std::vector<std::tuple<std::string, IndexT, IndexT>>      segments;  // anterior, posterior
    std::ofstream segment_csv;
    std::vector<Float> segment_lengths_0;

    if(data.contains("segment_interfaces"))
    {
        std::ifstream segment_file(CONFIG_FOLDER
                                    + data.at("segment_interfaces").get<std::string>());
        UIPC_ASSERT(segment_file.is_open(), "Could not open segment interfaces file.");

        Json segment_data;
        segment_file >> segment_data;

        for(auto& [name, entry] : segment_data.at("interfaces").items())
        {
            std::vector<IndexT> vertex_ids;
            for(auto& id : entry.at("vertex_ids"))
                vertex_ids.push_back(id.get<IndexT>());
            interfaces.emplace_back(name, std::move(vertex_ids));
        }

        for(auto& [name, boundaries] : segment_data.at("segments").items())
        {
            std::array<IndexT, 2> boundary_ids{};
            for(int k = 0; k < 2; ++k)
            {
                auto boundary_name = boundaries[k].get<std::string>();
                auto it = std::find_if(interfaces.begin(),
                                       interfaces.end(),
                                       [&](const auto& interface) {
                                           return std::get<0>(interface) == boundary_name;
                                       });
                UIPC_ASSERT(it != interfaces.end(),
                            "Segment `{}` references unknown interface `{}`.",
                            name,
                            boundary_name);
                boundary_ids[k] = static_cast<IndexT>(it - interfaces.begin());
            }
            segments.emplace_back(name, boundary_ids[0], boundary_ids[1]);
        }

        segment_csv.open(fmt::format("{}segment_lengths.csv", output_path));
        segment_csv << "frame";
        for(const auto& [name, vertex_ids] : interfaces)
            segment_csv << fmt::format(",x_{}", name);
        for(const auto& [name, anterior, posterior] : segments)
            segment_csv << fmt::format(",length_{},dlength_{}", name, name);
        segment_csv << "\n";
    }

    // mean X of every interface at the current frame, plus the segment
    // lengths it derives (no-op when segment tracking is disabled)
    auto write_segment_row = [&](int frame)
    {
        if(!segment_csv.is_open())
            return;

        auto positions = larva_mesh_slot.geometry->geometry().positions().view();

        std::vector<Float> xs(interfaces.size());
        for(auto&& [i, interface] : enumerate(interfaces))
        {
            const auto& [name, vertex_ids] = interface;
            Float mean_x = 0;
            for(auto id : vertex_ids)
                mean_x += positions[id].x();
            xs[i] = mean_x / static_cast<Float>(vertex_ids.size());
        }

        std::vector<Float> lengths(segments.size());
        for(auto&& [s, segment] : enumerate(segments))
            lengths[s] = xs[std::get<1>(segment)] - xs[std::get<2>(segment)];
        if(segment_lengths_0.empty())
            segment_lengths_0 = lengths;  // frame 0 is the reference

        segment_csv << frame;
        for(Float x : xs)
            segment_csv << fmt::format(",{}", x);
        for(auto&& [s, segment] : enumerate(segments))
            segment_csv << fmt::format(",{},{}", lengths[s], lengths[s] - segment_lengths_0[s]);
        segment_csv << "\n";
    };

    // 2D / 3D mesh export
    MeshExporter mesh_exporter(larva_mesh_slot, filetype);
    mesh_exporter.save_vertices_binary(0);
    mesh_exporter.save_scene(0);  // static topology for the viewer (frame 0 obj)
    mesh_exporter.muscle_groups_to_json(fmt::format("{}muscle_groups.json", output_path));
    write_segment_row(0);

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


        // mesh_exporter.save_scene(i + 1);
        mesh_exporter.save_vertices_binary(i + 1);
        write_segment_row(i + 1);

        auto end = std::chrono::system_clock::now();
        std::cout << fmt::format(
            "Iteration {}: {} ms elapsed.\n",
            i + 1,
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }

    save_msh(fmt::format("{}larva_curr.msh", output_path),
             larva_mesh_slot.geometry->geometry());
}
