#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <cmath>
#include <algorithm>

namespace details
{
// Helper to strip trailing carriage returns ('\r') for cross-platform support (Windows/Linux/macOS)
inline void trim_cr(std::string& s)
{
    if (!s.empty() && s.back() == '\r')
    {
        s.pop_back();
    }
}

// 1D Linear Interpolation helper for monotonically increasing time series
inline float interpolate_1d(const std::vector<float>& times, const std::vector<float>& values, float t)
{
    if (t <= times.front()) return values.front();
    if (t >= times.back()) return values.back();

    // Binary search for upper bound
    auto it = std::upper_bound(times.begin(), times.end(), t);
    size_t idx1 = std::distance(times.begin(), it);
    size_t idx0 = idx1 - 1;

    float t0 = times[idx0];
    float t1 = times[idx1];
    float v0 = values[idx0];
    float v1 = values[idx1];

    float alpha = (t - t0) / (t1 - t0);
    return (1.0f - alpha) * v0 + alpha * v1;
}
} // namespace details

/**
 * @brief Reads timestamps and muscle activation coefficients from a CSV file.
 * 
 * Expected CSV format:
 * time (s), muscle_id1, muscle_id2, ...
 * 0.00,     0.1,        0.5,        ...
 * 0.01,     0.2,        0.6,        ...
 * 
 * @param csv_path Path to the CSV file.
 * @return std::tuple<std::vector<float>, std::unordered_map<int, std::vector<float>>>
 *         - Element 0: Full sequence of time stamps from column 0.
 *         - Element 1: Mapping of muscle_id -> activation time series.
 */
inline std::tuple<std::vector<float>, std::unordered_map<int, std::vector<float>>> read_activation_csv(const std::string& csv_path)
{
    std::ifstream file(csv_path);
    if (!file.is_open())
    {
        throw std::runtime_error("[read_activation_csv] Failed to open CSV file: " + csv_path);
    }

    std::string line;

    // 1. Parse Header Row: time (s), muscle_id1, muscle_id2, ...
    if (!std::getline(file, line))
    {
        throw std::runtime_error("[read_activation_csv] CSV file is empty: " + csv_path);
    }

    details::trim_cr(line);
    std::stringstream header_stream(line);
    std::string cell;

    // Skip the first column header (e.g., "time (s)")
    if (!std::getline(header_stream, cell, ','))
    {
        throw std::runtime_error("[read_activation_csv] Invalid CSV header format in: " + csv_path);
    }

    std::vector<int> column_to_muscle_id;
    std::unordered_map<int, std::vector<float>> activation_dict;

    // Read muscle IDs from column 1 onwards
    while (std::getline(header_stream, cell, ','))
    {
        details::trim_cr(cell);
        if (cell.empty())
            continue;

        int muscle_id = std::stoi(cell);
        column_to_muscle_id.push_back(muscle_id);

        // Pre-initialize vector for this muscle ID
        activation_dict[muscle_id] = std::vector<float>();
    }

    if (column_to_muscle_id.empty())
    {
        throw std::runtime_error("[read_activation_csv] No muscle IDs found in CSV header: " + csv_path);
    }

    // 2. Parse Data Rows
    std::vector<float> times;

    while (std::getline(file, line))
    {
        details::trim_cr(line);
        if (line.empty())
            continue;

        std::stringstream line_stream(line);

        // Parse time value in column 0
        if (!std::getline(line_stream, cell, ','))
            continue;

        details::trim_cr(cell);
        if (!cell.empty())
        {
            times.push_back(std::stof(cell));
        }

        // Parse muscle activation values in remaining columns
        size_t col_idx = 0;
        while (std::getline(line_stream, cell, ',') && col_idx < column_to_muscle_id.size())
        {
            details::trim_cr(cell);
            if (!cell.empty())
            {
                float val = std::stof(cell);
                int muscle_id = column_to_muscle_id[col_idx];
                activation_dict[muscle_id].push_back(val);
            }
            col_idx++;
        }
    }

    if (times.size() < 2)
    {
        throw std::runtime_error("[read_activation_csv] File contains fewer than 2 data frames: " + csv_path);
    }

    return {times, activation_dict};
}

/**
 * @brief Computes average dt of the input time vector and resamples activation_dict if 
 *        the average dt differs from target_dt.
 * 
 * @param times Vector of timestamps corresponding to the original activations.
 * @param activation_dict Original map of muscle_id -> vector of activation values.
 * @param target_dt Desired simulation frame time step.
 * @param eps Tolerance threshold for float comparison.
 * @return std::unordered_map<int, std::vector<float>> Resampled activation dictionary.
 */
inline std::unordered_map<int, std::vector<float>> resample_activations(
    const std::vector<float>& times,
    const std::unordered_map<int, std::vector<float>>& activation_dict,
    float target_dt,
    float eps = 1e-6f)
{
    if (times.size() < 2)
    {
        throw std::invalid_argument("[resample_activations] At least 2 time points are required.");
    }
    if (target_dt <= 0.0f)
    {
        throw std::invalid_argument("[resample_activations] target_dt must be positive.");
    }

    // 1. Compute average dt from original times
    float t_start = times.front();
    float t_end = times.back();
    float total_duration = t_end - t_start;
    float avg_dt = total_duration / static_cast<float>(times.size() - 1);

    // 2. Check if resampling is needed
    if (std::abs(avg_dt - target_dt) <= eps)
    {
        return activation_dict; // No resampling needed
    }

    // 3. Construct new time grid
    std::vector<float> resampled_times;

    float t = t_start;
    while(t <= t_end)
    {
        resampled_times.push_back(t);
        t += target_dt;
    }

    // 4. Resample activations for each muscle ID using linear interpolation
    std::unordered_map<int, std::vector<float>> resampled_dict;

    for (const auto& [muscle_id, values] : activation_dict)
    {
        std::vector<float> resampled_values;
        resampled_values.reserve(resampled_times.size());

        for (float t_target : resampled_times)
        {
            float interpolated_val = details::interpolate_1d(times, values, t_target);
            resampled_values.push_back(interpolated_val);
        }

        resampled_dict[muscle_id] = std::move(resampled_values);
    }

    return resampled_dict;
}