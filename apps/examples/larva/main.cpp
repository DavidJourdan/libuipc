#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/muscle.h>
#include <uipc/core/muscle_controller_feature.h>
#include <uipc/core/fem_system_feature.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <uipc/core/contact_system_feature.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/builtin/geometry_type.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include "read_activation_csv.h"
#include "mesh_export.h"
#include "segment_tracker.h"

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

    // ---- FEM data export (issue #344): energy bookkeeping ----------------
    // Every frame, the elastic energy of both constitutions (SNH + Muscle),
    // the kinetic energy, the gravitational potential energy and the
    // contact (IPC barrier) energy are appended to fem_energy.csv.
    // All exports carry the solver's dt^2 incremental-potential scaling
    // and are divided by dt^2 here.
    auto fem_system = world.features().find<FEMSystemFeature>();
    UIPC_ASSERT_THROW(fem_system, "FEMSystemFeature not found: this app requires the CUDA backend.");
    auto fem_state = world.features().find<FiniteElementStateAccessorFeature>();
    UIPC_ASSERT_THROW(fem_state, "FiniteElementStateAccessorFeature not found.");
    auto contact_system = world.features().find<ContactSystemFeature>();
    UIPC_ASSERT_THROW(contact_system, "ContactSystemFeature not found.");
    auto contact_types = contact_system->contact_primitive_types();

    // both constitutions must be exportable (the Muscle one exercises the
    // extra-constitution path of the exporter)
    {
        auto uids = fem_system->fem_constitution_uids();
        UIPC_ASSERT_THROW(std::find(uids.begin(), uids.end(), snh_constitution.uid()) != uids.end(),
                          "SNH constitution not exportable");
        UIPC_ASSERT_THROW(std::find(uids.begin(), uids.end(), muscle_constitution.uid()) != uids.end(),
                          "Muscle constitution not exportable");
    }

    const Float dt = data.at("dt").get<double>();

    auto state_geo = fem_state->create_geometry();
    state_geo.vertices().create<Vector3>(builtin::position, Vector3::Zero());
    state_geo.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());

    // per-vertex masses, matching the backend mass assembly:
    // mass = mass_density * lumped vertex volume; the density is the
    // StableNeoHookean/Muscle apply_to default (1e3)
    vector<Float> vertex_mass;
    {
        auto volume = larva_rest_mesh.vertices().find<Float>(builtin::volume);
        UIPC_ASSERT_THROW(volume, "`builtin::volume` missing: it is created by apply_to");
        vertex_mass.resize(volume->size());
        std::ranges::transform(geometry::view(*volume),
                               vertex_mass.begin(),
                               [](Float v) { return 1e3 * v; });
    }

    // gravitational potential datum: the ground plane when there is one
    const Float potential_datum =
        (data.contains("ground") && data["ground"].get<bool>()) ?
            data.at("ground_height").get<double>() :
            0.0;

    std::ofstream energy_csv{fmt::format("{}fem_energy.csv", output_path)};
    UIPC_ASSERT_THROW(energy_csv.is_open(), "Failed to open fem_energy.csv for writing.");
    energy_csv << "frame,"
               << "time,"
               << "snh_energy,"
               << "muscle_energy,"
               << "elastic_energy,"
               << "kinetic_energy,"
               << "potential_energy,"
               << "contact_energy,"
               << "total_energy"
               << "\n";

    // total elastic energy of one constitution, dt^2-scaled like the
    // solver's incremental potential
    auto fem_energy_sum = [&](const IConstitution& c) -> Float
    {
        Geometry geo;
        fem_system->fem_energy(c, geo);
        auto e = geo.instances().find<Float>("energy");
        if(!e)
            return 0.0;
        auto ev = geometry::view(*e);
        return std::accumulate(ev.begin(), ev.end(), 0.0);
    };

    // frame-0 elastic energies. A drop run starts at the rest pose, so this
    // is exactly the constant rest offset of the constitutions (SNK1's
    // Psi(I) = mu^2/(2 lambda_hat) plus the Muscle passive rest energy);
    // a hunch run starts from the settled pose, so the offset is the
    // settled strain. Either way `elastic_energy` reports the elastic
    // energy relative to the frame-0 pose.
    fem_state->copy_to(state_geo);
    const Float snh_offset    = fem_energy_sum(snh_constitution) / (dt * dt);
    const Float muscle_offset = fem_energy_sum(muscle_constitution) / (dt * dt);

    // per-frame energy row
    auto write_energy_row = [&](int frame, Float snh_e, Float muscle_e)
    {
        auto positions = geometry::view(*state_geo.vertices().find<Vector3>(builtin::position));
        auto velocities = geometry::view(*state_geo.vertices().find<Vector3>(builtin::velocity));

        const Float elastic_e = snh_e + muscle_e - snh_offset - muscle_offset;

        Float kinetic_e   = 0.0;  // 1/2 m v^2
        Float potential_e = 0.0;  // m g h, datum at the ground plane
        for(SizeT v = 0; v < vertex_mass.size(); ++v)
        {
            kinetic_e += 0.5 * vertex_mass[v] * velocities[v].squaredNorm();
            potential_e += vertex_mass[v] * 9.8 * (positions[v].y() - potential_datum);
        }

        // contact (IPC barrier) energy of the contact pairs; the exported
        // values are dt^2-scaled too
        Float contact_e = 0.0;
        for(auto&& prim_type : contact_types)
        {
            Geometry contact_geo;
            contact_system->contact_energy(prim_type, contact_geo);
            if(auto e = contact_geo.instances().find<Float>("energy"))
            {
                auto ev = geometry::view(*e);
                contact_e += std::accumulate(ev.begin(), ev.end(), 0.0);
            }
        }
        contact_e /= dt * dt;

        const Float total_e = elastic_e + kinetic_e + potential_e + contact_e;

        energy_csv << fmt::format("{}, {}, {}, {}, {}, {}, {}, {}, {}\n",
                                  frame,
                                  frame * dt,
                                  snh_e,
                                  muscle_e,
                                  elastic_e,
                                  kinetic_e,
                                  potential_e,
                                  contact_e,
                                  total_e);
        return total_e;
    };

    Float    export_seconds = 0.0;
    uint64_t export_calls   = 0;

    // frame 0: rest pose (or the hunch start pose), zero velocity
    write_energy_row(0, snh_offset, muscle_offset);

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

        // FEM export: elastic energies of both constitutions at the new state
        fem_state->copy_to(state_geo);
        auto t0 = std::chrono::steady_clock::now();
        const Float snh_e    = fem_energy_sum(snh_constitution) / (dt * dt);
        const Float muscle_e = fem_energy_sum(muscle_constitution) / (dt * dt);
        auto t1 = std::chrono::steady_clock::now();
        export_seconds += std::chrono::duration<Float>(t1 - t0).count();
        export_calls += 2;

        write_energy_row(i + 1, snh_e, muscle_e);

        // mesh_exporter.save_scene(i + 1);
        mesh_exporter.save_vertices_binary(i + 1);
        segment_tracker.write_row(larva_mesh_slot.geometry->geometry(), i + 1);

        auto end = std::chrono::system_clock::now();
        std::cout << fmt::format(
            "Iteration {}: {} ms elapsed.\n",
            i + 1,
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }

    energy_csv.close();
    fmt::print("Wrote {}fem_energy.csv\n", output_path);
    fmt::print("Average FEM energy export time: {:.3f} ms/call\n",
               export_seconds / export_calls * 1000.0);

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
