#pragma once

#include <uipc/uipc.h>
#include <uipc/common/enumerate.h>
#include <array>
#include <algorithm>
#include <fstream>
#include <tuple>

using namespace uipc;
using namespace uipc::geometry;

// Tracks the segment interfaces of the larva mesh: groups of vertex indices
// sitting exactly on the segment boundary planes of the rest mesh (written
// alongside the mesh by larva-bodies' generate_tet_mesh). Every write_row()
// appends the mean X of each interface and the segment lengths (and their
// change relative to frame 0) to a csv file.
class SegmentTracker
{
  public:
    SegmentTracker() = default;

    // interfaces_file: json with an "interfaces" map (name -> vertex_ids)
    //                  and a "segments" map (name -> [anterior, posterior]
    //                  interface names);
    // csv_file:        where the per-frame rows are appended.
    SegmentTracker(const std::string& interfaces_file, const std::string& csv_file)
    {
        std::ifstream segment_file{interfaces_file};
        UIPC_ASSERT(segment_file.is_open(),
                    "Could not open segment interfaces file `{}`.",
                    interfaces_file);

        Json segment_data;
        segment_file >> segment_data;

        for(auto& [name, entry] : segment_data.at("interfaces").items())
        {
            std::vector<IndexT> vertex_ids;
            for(auto& id : entry.at("vertex_ids"))
                vertex_ids.push_back(id.get<IndexT>());
            _interfaces.emplace_back(name, std::move(vertex_ids));
        }

        for(auto& [name, boundaries] : segment_data.at("segments").items())
        {
            std::array<IndexT, 2> boundary_ids{};
            for(int k = 0; k < 2; ++k)
            {
                auto boundary_name = boundaries[k].get<std::string>();
                auto it            = std::find_if(_interfaces.begin(),
                                       _interfaces.end(),
                                       [&](const auto& interface)
                                       {
                                           return std::get<0>(interface) == boundary_name;
                                       });
                UIPC_ASSERT(it != _interfaces.end(),
                            "Segment `{}` references unknown interface `{}`.",
                            name,
                            boundary_name);
                boundary_ids[k] = static_cast<IndexT>(it - _interfaces.begin());
            }
            _segments.emplace_back(name, boundary_ids[0], boundary_ids[1]);
        }

        _csv.open(csv_file);
        UIPC_ASSERT(_csv.is_open(), "Could not open `{}` for writing.", csv_file);

        _csv << "frame";
        for(const auto& [name, vertex_ids] : _interfaces)
            _csv << fmt::format(",x_{}", name);
        for(const auto& [name, anterior, posterior] : _segments)
            _csv << fmt::format(",length_{},dlength_{}", name, name);
        _csv << "\n";
    }

    // Appends the mean X of every interface at the current frame and the
    // segment lengths derived from them. No-op for a default-constructed
    // (disabled) tracker.
    void write_row(const SimplicialComplex& mesh, int frame)
    {
        if(!_csv.is_open())
            return;

        auto positions = mesh.positions().view();

        std::vector<Float> xs(_interfaces.size());
        for(auto&& [i, interface] : enumerate(_interfaces))
        {
            const auto& [name, vertex_ids] = interface;
            Float mean_x                   = 0;
            for(auto id : vertex_ids)
                mean_x += positions[id].x();
            xs[i] = mean_x / static_cast<Float>(vertex_ids.size());
        }

        std::vector<Float> lengths(_segments.size());
        for(auto&& [s, segment] : enumerate(_segments))
            lengths[s] = xs[std::get<1>(segment)] - xs[std::get<2>(segment)];
        if(_lengths_0.empty())
            _lengths_0 = lengths;  // frame 0 is the reference

        _csv << frame;
        for(Float x : xs)
            _csv << fmt::format(",{}", x);
        for(auto&& [s, segment] : enumerate(_segments))
            _csv << fmt::format(",{},{}", lengths[s], lengths[s] - _lengths_0[s]);
        _csv << "\n";
    }

  private:
    std::vector<std::tuple<std::string, std::vector<IndexT>>> _interfaces;
    std::vector<std::tuple<std::string, IndexT, IndexT>> _segments;  // anterior, posterior
    std::ofstream      _csv;
    std::vector<Float> _lengths_0;
};
