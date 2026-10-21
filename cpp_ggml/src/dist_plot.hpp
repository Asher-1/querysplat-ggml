// Pure C++ re-plots of the official gaussian_{alpha,scale}_distribution.png
// (scripts/output_utils.py, matplotlib). Same figure geometry (14x5.5 in at
// 150 dpi -> 2100x825), colors, axes (log-reversed alpha axis / linear scale
// axis), tick values + labels, threshold markers and statistics text box; the
// rasterization is this file's, not matplotlib's (no pixel-identity claim).
#pragma once

#include <string>
#include <vector>

namespace qs {

struct AlphaPlot {
    std::vector<double> edges;       // log bin edges (ascending, last = 1.0)
    std::vector<double> percent;     // per-bin % of all gaussians
    int log_min_exp = -6;
    std::vector<double> thresholds;  // official threshold tuple order
    std::vector<double> threshold_percent;
    long long total = 0;
    double zero_percent = 0;
};
bool plot_alpha_distribution(const std::string& png_path, const AlphaPlot& p);

struct ScalePlot {
    double max_scale = 0;
    std::vector<double> edges;           // num_bins + 1 linear edges
    std::vector<double> percent;         // per-bin %
    std::vector<double> cumulative_percent;
    long long total = 0;                 // scale values (3 per gaussian)
    double zero_percent = 0;
    double above_percent = 0;
};
bool plot_scale_distribution(const std::string& png_path, const ScalePlot& p);

}  // namespace qs
