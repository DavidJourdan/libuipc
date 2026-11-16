#pragma once

#include <uipc/uipc.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/muscle.h>
#include <uipc/core/fem_system_feature.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <uipc/core/contact_system_feature.h>
#include <uipc/builtin/attribute_name.h>
#include <chrono>
#include <fstream>
#include <numeric>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

// Per-frame energy bookkeeping via FEMSystemFeature/ContactSystemFeature
// (issue #344). Every write_row() appends the elastic energies of both
// constitutions (SNH + Muscle), the kinetic energy, the gravitational
// potential energy and the contact (IPC barrier) energy to a csv file.
//
// The exports carry the solver's dt^2 incremental-potential scaling and are
// divided by dt^2 here. `elastic_energy` is relative to the frame-0 pose:
// a drop run starts at the rest pose, so the frame-0 elastic energy is
// exactly the constitutions' constant rest offset (SNK1's nonzero Psi(I)
// plus the Muscle passive rest energy); a hunch run starts from the
// settled pose, so the offset is the settled strain.
//
// Enabled by constructing with the scene data; a default-constructed
// logger (no "export_energy" key in the json config) is a no-op.
class EnergyLogger
{
  public:
    EnergyLogger() = default;

    // world:       must already be init()-ed (the features are registry lookups);
    // rest_mesh:   source of the per-vertex lumped volumes for the masses;
    // snh/muscle:  the constitutions whose energies are exported;
    // config:      the app's json config (dt, ground, ground_height);
    // csv_file:    where the per-frame rows are appended.
    EnergyLogger(World&                world,
                 SimplicialComplex&    rest_mesh,
                 const StableNeoHookean& snh,
                 const Muscle&         muscle,
                 const Json&           config,
                 const std::string&    csv_file)
        : _snh_uid(snh.uid())
        , _muscle_uid(muscle.uid())
        , _dt(config.at("dt").get<Float>())
    {
        _fem_system = world.features().find<FEMSystemFeature>();
        UIPC_ASSERT_THROW(_fem_system,
                          "FEMSystemFeature not found: this app requires the CUDA backend.");
        _fem_state = world.features().find<FiniteElementStateAccessorFeature>();
        UIPC_ASSERT_THROW(_fem_state, "FiniteElementStateAccessorFeature not found.");
        _contact_system = world.features().find<ContactSystemFeature>();
        UIPC_ASSERT_THROW(_contact_system, "ContactSystemFeature not found.");
        _contact_types = _contact_system->contact_primitive_types();

        // both constitutions must be exportable (the Muscle one exercises
        // the extra-constitution path of the exporter)
        {
            auto uids = _fem_system->fem_constitution_uids();
            UIPC_ASSERT_THROW(std::find(uids.begin(), uids.end(), _snh_uid) != uids.end(),
                              "SNH constitution not exportable");
            UIPC_ASSERT_THROW(std::find(uids.begin(), uids.end(), _muscle_uid) != uids.end(),
                              "Muscle constitution not exportable");
        }

        _state_geo = std::make_unique<SimplicialComplex>(_fem_state->create_geometry());
        _state_geo->vertices().create<Vector3>(builtin::position, Vector3::Zero());
        _state_geo->vertices().create<Vector3>(builtin::velocity, Vector3::Zero());

        // per-vertex masses, matching the backend mass assembly:
        // mass = mass_density * lumped vertex volume; the density is the
        // StableNeoHookean/Muscle apply_to default (1e3)
        {
            auto volume = rest_mesh.vertices().find<Float>(builtin::volume);
            UIPC_ASSERT_THROW(volume, "`builtin::volume` missing: it is created by apply_to");
            _vertex_mass.resize(volume->size());
            std::ranges::transform(view(*volume),
                                   _vertex_mass.begin(),
                                   [](Float v) { return 1e3 * v; });
        }

        // gravitational potential datum: the ground plane when there is one
        _potential_datum = (config.contains("ground") && config["ground"].get<bool>()) ?
                               config.at("ground_height").get<Float>() :
                               0.0;

        _csv.open(csv_file);
        UIPC_ASSERT_THROW(_csv.is_open(), "Could not open `{}` for writing.", csv_file);
        _csv << "frame,"
             << "time,"
             << "snh_energy,"
             << "muscle_energy,"
             << "elastic_energy,"
             << "kinetic_energy,"
             << "potential_energy,"
             << "contact_energy,"
             << "total_energy"
             << "\n";

        // frame-0 elastic energies, the offset subtracted from every
        // later `elastic_energy` value
        _fem_state->copy_to(*_state_geo);
        _snh_offset    = _fem_energy_sum(_snh_uid) / (_dt * _dt);
        _muscle_offset = _fem_energy_sum(_muscle_uid) / (_dt * _dt);
    }

    // Appends the energies at the backend's current state. No-op for a
    // default-constructed (disabled) logger.
    void write_row(int frame)
    {
        if(!_csv.is_open())
            return;

        _fem_state->copy_to(*_state_geo);

        // elastic energies of both constitutions at the new state
        auto t0 = std::chrono::steady_clock::now();
        const Float snh_e    = _fem_energy_sum(_snh_uid) / (_dt * _dt);
        const Float muscle_e = _fem_energy_sum(_muscle_uid) / (_dt * _dt);
        auto t1              = std::chrono::steady_clock::now();
        _export_seconds += std::chrono::duration<Float>(t1 - t0).count();
        _export_calls += 2;

        const Float elastic_e = snh_e + muscle_e - _snh_offset - _muscle_offset;

        auto positions  = view(*_state_geo->vertices().find<Vector3>(builtin::position));
        auto velocities = view(*_state_geo->vertices().find<Vector3>(builtin::velocity));

        Float kinetic_e   = 0.0;  // 1/2 m v^2
        Float potential_e = 0.0;  // m g h, datum at the ground plane
        for(SizeT v = 0; v < _vertex_mass.size(); ++v)
        {
            kinetic_e += 0.5 * _vertex_mass[v] * velocities[v].squaredNorm();
            potential_e += _vertex_mass[v] * 9.8 * (positions[v].y() - _potential_datum);
        }

        // contact (IPC barrier) energy of the contact pairs; the exported
        // values are dt^2-scaled too
        Float contact_e = 0.0;
        for(auto&& prim_type : _contact_types)
        {
            Geometry contact_geo;
            _contact_system->contact_energy(prim_type, contact_geo);
            if(auto e = contact_geo.instances().find<Float>("energy"))
            {
                auto ev = view(*e);
                contact_e += std::accumulate(ev.begin(), ev.end(), 0.0);
            }
        }
        contact_e /= _dt * _dt;

        const Float total_e = elastic_e + kinetic_e + potential_e + contact_e;

        _csv << fmt::format("{}, {}, {}, {}, {}, {}, {}, {}, {}\n",
                            frame,
                            frame * _dt,
                            snh_e,
                            muscle_e,
                            elastic_e,
                            kinetic_e,
                            potential_e,
                            contact_e,
                            total_e);
    }

    // Closes the csv and prints the measured export cost. No-op for a
    // default-constructed (disabled) logger.
    void finish()
    {
        if(!_csv.is_open())
            return;

        _csv.close();
        fmt::print("Wrote fem_energy.csv ({} energy exports, {:.3f} ms/call)\n",
                   _export_calls,
                   _export_seconds / _export_calls * 1000.0);
    }

  private:
    // total elastic energy of one constitution, dt^2-scaled like the
    // solver's incremental potential
    Float _fem_energy_sum(U64 uid)
    {
        Geometry geo;
        _fem_system->fem_energy(uid, geo);
        auto e = geo.instances().find<Float>("energy");
        if(!e)
            return 0.0;
        auto ev = view(*e);
        return std::accumulate(ev.begin(), ev.end(), 0.0);
    }

    S<FEMSystemFeature>                   _fem_system;
    S<FiniteElementStateAccessorFeature>  _fem_state;
    S<ContactSystemFeature>              _contact_system;
    vector<std::string>                  _contact_types;
    std::unique_ptr<SimplicialComplex>   _state_geo;
    std::vector<Float>                  _vertex_mass;
    std::ofstream                     _csv;
    U64                               _snh_uid    = 0;
    U64                               _muscle_uid = 0;
    Float                             _dt         = 0.0;
    Float                             _potential_datum = 0.0;
    Float                             _snh_offset    = 0.0;
    Float                             _muscle_offset = 0.0;
    Float                             _export_seconds = 0.0;
    uint64_t                          _export_calls   = 0;
};
