// Pure C++ re-plots of the official distribution PNGs. See dist_plot.hpp.
//
// Geometry follows matplotlib's defaults for the official calls: 14x5.5 in
// figure at 150 dpi, tight_layout pad 1.08 x 10 pt, tick length 3.5 pt /
// pad 3.5 pt, labelpad 4 pt, title pad 6 pt, DejaVu Sans glyphs, AutoLocator
// y ticks (MaxNLocator steps 1/2/2.5/5/10), grid/dash patterns scaled by line
// width. Everything is drawn into a float RGB canvas with analytic box
// coverage (rects/lines) and 2x2 supersampled glyphs.

#include "dist_plot.hpp"

#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace qs {

namespace {

#include "plot_font.inc"

constexpr int kFigW = 2100, kFigH = 825;  // 14 x 5.5 in @ 150 dpi
constexpr double kPt = 150.0 / 72.0;      // px per point

struct Color { float r, g, b; };
constexpr Color kBlack{0, 0, 0};
constexpr Color kWhite{1, 1, 1};
Color hex(unsigned v) {
    return {((v >> 16) & 0xFF) / 255.f, ((v >> 8) & 0xFF) / 255.f, (v & 0xFF) / 255.f};
}

// ---- glyph atlas -------------------------------------------------------------

struct Atlas {
    int w = 0, h = 0;
    std::vector<unsigned char> cov;
};

const Atlas& atlas() {
    static const Atlas a = [] {
        Atlas at;
        std::string png;  // base64 decode
        int acc = 0, bits = 0;
        for (const char* p = kFontAtlasB64; *p; p++) {
            const char c = *p;
            int v = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26
                  : c >= '0' && c <= '9' ? c - '0' + 52 : c == '+' ? 62 : c == '/' ? 63 : -1;
            if (v < 0) continue;
            acc = (acc << 6) | v;
            bits += 6;
            if (bits >= 8) { bits -= 8; png += (char)((acc >> bits) & 0xFF); }
        }
        int n = 0;
        unsigned char* px = stbi_load_from_memory(
            reinterpret_cast<const unsigned char*>(png.data()), (int)png.size(), &at.w, &at.h, &n, 1);
        if (px) {
            at.cov.assign(px, px + (size_t)at.w * at.h);
            stbi_image_free(px);
        }
        return at;
    }();
    return a;
}

float atlas_sample(float x, float y) {  // bilinear coverage in [0,1]
    const Atlas& a = atlas();
    if (a.cov.empty()) return 0.f;
    x -= 0.5f; y -= 0.5f;
    const int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    const float fx = x - x0, fy = y - y0;
    auto at = [&](int xx, int yy) -> float {
        if (xx < 0 || yy < 0 || xx >= a.w || yy >= a.h) return 0.f;
        return a.cov[(size_t)yy * a.w + xx] / 255.f;
    };
    return (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) +
           fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
}

// matplotlib lays a line out on the "lp" ink box: ~0.76 em above the
// baseline, ~0.24 em below (DejaVu Sans)
constexpr double kAsc = 0.76, kDesc = 0.24;

double text_width(const std::string& s, double pt) {
    const double scale = pt * kPt / kFontPx;
    double w = 0;
    for (unsigned char c : s)
        if (c >= 32 && c < 127) w += kGlyphW[c - 32] * scale;
    return w;
}
double text_height(double pt) { return (kAsc + kDesc) * pt * kPt; }

enum class Ha { Left, Center, Right };
enum class Va { Top, Center, Bottom, Baseline };

// ---- canvas ------------------------------------------------------------------

struct Canvas {
    std::vector<float> px = std::vector<float>((size_t)kFigW * kFigH * 3, 1.f);

    void blend(int x, int y, Color c, float a) {
        if (x < 0 || y < 0 || x >= kFigW || y >= kFigH || a <= 0) return;
        float* p = &px[((size_t)y * kFigW + x) * 3];
        a = std::min(a, 1.f);
        p[0] += (c.r - p[0]) * a;
        p[1] += (c.g - p[1]) * a;
        p[2] += (c.b - p[2]) * a;
    }

    // axis-aligned rectangle with exact per-pixel area coverage
    void rect(double x0, double y0, double x1, double y1, Color c, float a) {
        if (x1 < x0) std::swap(x0, x1);
        if (y1 < y0) std::swap(y0, y1);
        for (int y = (int)std::floor(y0); y < (int)std::ceil(y1); y++) {
            const double oy = std::min(y1, y + 1.0) - std::max(y0, (double)y);
            if (oy <= 0) continue;
            for (int x = (int)std::floor(x0); x < (int)std::ceil(x1); x++) {
                const double ox = std::min(x1, x + 1.0) - std::max(x0, (double)x);
                if (ox > 0) blend(x, y, c, (float)(ox * oy) * a);
            }
        }
    }

    // straight axis-aligned line, optionally dashed (on/off lengths in px)
    void line(double x0, double y0, double x1, double y1, double lw, Color c, float a,
              double on = 0, double off = 0) {
        const bool horiz = std::abs(y1 - y0) < 1e-9;
        const double len = horiz ? std::abs(x1 - x0) : std::abs(y1 - y0);
        const double sx = horiz ? std::min(x0, x1) : x0, sy = horiz ? y0 : std::min(y0, y1);
        auto seg = [&](double t0, double t1) {
            if (horiz) rect(sx + t0, sy - lw / 2, sx + t1, sy + lw / 2, c, a);
            else rect(sx - lw / 2, sy + t0, sx + lw / 2, sy + t1, c, a);
        };
        if (on <= 0) { seg(0, len); return; }
        for (double t = 0; t < len; t += on + off) seg(t, std::min(t + on, len));
    }

    // single-line text, matplotlib rotation_mode="default": rotate, then
    // align the rotated bounding box to (ax, ay) per ha/va
    void text(const std::string& s, double pt, double ax, double ay, Ha ha, Va va,
              double rot_deg, Color c, float alpha = 1.f) {
        const double em = pt * kPt, scale = em / kFontPx;
        std::vector<double> start;
        double w = 0;
        for (unsigned char ch : s) {
            start.push_back(w);
            if (ch >= 32 && ch < 127) w += kGlyphW[ch - 32] * scale;
        }
        const double top = -kAsc * em, bot = kDesc * em;
        const double th = rot_deg * M_PI / 180.0, cs = std::cos(th), sn = std::sin(th);
        double mnx = 1e30, mxx = -1e30, mny = 1e30, mxy = -1e30;
        for (double u : {0.0, w})
            for (double v : {top, bot}) {
                const double x = u * cs + v * sn, y = -u * sn + v * cs;
                mnx = std::min(mnx, x); mxx = std::max(mxx, x);
                mny = std::min(mny, y); mxy = std::max(mxy, y);
            }
        const double ox = ha == Ha::Left ? ax - mnx : ha == Ha::Right ? ax - mxx : ax - (mnx + mxx) / 2;
        double oy = va == Va::Top ? ay - mny : va == Va::Bottom ? ay - mxy : ay - (mny + mxy) / 2;
        if (va == Va::Baseline) oy = ay;
        const double gpad = 2 * scale;  // atlas gutter: catch glyph overhang
        // glyph rows span the full atlas cell, which may extend past the
        // lp box; widen the scan by the cell excess
        const double margin = (double)kFontCellH * scale;
        for (int py = (int)std::floor(oy + mny - margin); py <= (int)std::ceil(oy + mxy + margin); py++)
            for (int pxl = (int)std::floor(ox + mnx - margin); pxl <= (int)std::ceil(ox + mxx + margin); pxl++) {
                float cov = 0;
                for (int sy = 0; sy < 2; sy++)
                    for (int sx = 0; sx < 2; sx++) {
                        const double x = pxl + 0.25 + 0.5 * sx - ox, y = py + 0.25 + 0.5 * sy - oy;
                        const double u = x * cs - y * sn, v = x * sn + y * cs;
                        const float ay_ = (float)(v / scale + kFontAscent);
                        if (ay_ < 0 || ay_ > kFontCellH || u < -gpad || u > w + gpad) continue;
                        float best = 0;
                        for (size_t k = 0; k < s.size(); k++) {
                            const unsigned char ch = (unsigned char)s[k];
                            if (ch < 33 || ch >= 127) continue;
                            const double lu = u - start[k];
                            if (lu < -gpad || lu > kGlyphW[ch - 32] * scale + gpad) continue;
                            best = std::max(best, atlas_sample((float)(kGlyphX[ch - 32] + lu / scale), ay_));
                        }
                        cov += best;
                    }
                blend(pxl, py, c, cov / 4.f * alpha);
            }
    }

    // matplotlib's Agg savefig writes opaque RGBA; same PNG color type here
    bool save(const std::string& path) const {
        const size_t n = (size_t)kFigW * kFigH;
        std::vector<unsigned char> out(n * 4, 255);
        for (size_t k = 0; k < n; k++)
            for (int c = 0; c < 3; c++)
                out[k * 4 + c] =
                    (unsigned char)std::lround(std::clamp(px[k * 3 + c], 0.f, 1.f) * 255.f);
        return stbi_write_png(path.c_str(), kFigW, kFigH, 4, out.data(), kFigW * 4) != 0;
    }
};

// rounded box (FancyBboxPatch boxstyle="round") fill + stroke, 4x4 samples
void rounded_box(Canvas& cv, double x0, double y0, double x1, double y1, double r,
                 Color face, Color edge, double lw, float alpha) {
    auto inside = [&](double x, double y, double inset) {
        const double a0 = x0 + inset, b0 = y0 + inset, a1 = x1 - inset, b1 = y1 - inset;
        const double rr = std::max(r - inset, 0.0);
        if (x < a0 || x > a1 || y < b0 || y > b1) return false;
        const double cx = std::clamp(x, a0 + rr, a1 - rr), cy = std::clamp(y, b0 + rr, b1 - rr);
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= rr * rr;
    };
    for (int py = (int)std::floor(y0 - lw); py <= (int)std::ceil(y1 + lw); py++)
        for (int px = (int)std::floor(x0 - lw); px <= (int)std::ceil(x1 + lw); px++) {
            int nf = 0, ne = 0;
            for (int s = 0; s < 16; s++) {
                const double x = px + (s % 4 + 0.5) / 4, y = py + (s / 4 + 0.5) / 4;
                const bool outer = inside(x, y, -lw / 2), inner = inside(x, y, lw / 2);
                nf += inside(x, y, 0);
                ne += outer && !inner;
            }
            cv.blend(px, py, face, nf / 16.f * alpha);
            cv.blend(px, py, edge, ne / 16.f * alpha);
        }
}

// ---- axes ----------------------------------------------------------------------

constexpr double kTickLen = 3.5 * kPt, kTickPad = 3.5 * kPt, kLabelPad = 4 * kPt;
constexpr double kTitlePad = 6 * kPt, kLayoutPad = 1.08 * 10 * kPt;
constexpr double kTickFont = 10, kLabelFont = 10, kTitleFont = 12;
const Color kGrid = hex(0xb0b0b0);

struct YTicks {
    std::vector<double> v;
    std::vector<std::string> label;
};

// AutoLocator == MaxNLocator(nbins="auto" -> min(axis_len_pt / 20, 9),
// steps=[1, 2, 2.5, 5, 10]) on [0, top]; ScalarFormatter shows the fewest
// decimals that represent every tick exactly.
YTicks y_ticks(double top, double axis_px) {
    const int nbins = std::max(1, std::min((int)std::floor(axis_px / kPt / (kTickFont * 2)), 9));
    const double raw = top / nbins;
    const double sc = std::pow(10.0, std::floor(std::log10(raw)));
    double step = 10 * sc;
    for (double m : {1.0, 2.0, 2.5, 5.0, 10.0})
        if (m * sc >= raw * (1 - 1e-10)) { step = m * sc; break; }
    YTicks t;
    for (int k = 0; k * step <= top * (1 + 1e-10); k++) t.v.push_back(k * step);
    int dec = 0;
    for (; dec < 8; dec++) {
        bool ok = true;
        for (double v : t.v) {
            const double s = v * std::pow(10.0, dec);
            if (std::abs(s - std::round(s)) > 1e-6 * std::max(1.0, std::abs(s))) ok = false;
        }
        if (ok) break;
    }
    for (double v : t.v) {
        char b[32];
        std::snprintf(b, sizeof(b), "%.*f", dec, v);
        t.label.emplace_back(b);
    }
    return t;
}

struct Frame { double L, R, T, B; };

struct AxesSpec {
    std::vector<double> xt;          // major x tick data values
    std::vector<std::string> xtl;    // their labels (rotated 45, ha=right)
    double ytop = 1;                 // ylim(0, ytop)
    std::string xlabel, ylabel, title;
    double top_extra_frac = -1;      // data-y fraction where rotated labels start
    double top_extra_len = 0;        // their extent above that point (px)
};

// tight_layout: margins sized from the decorations' extents
Frame layout(const AxesSpec& a) {
    // y tick labels depend on the axis length; two passes converge
    Frame f{0, kFigW - kLayoutPad, 0, 0};
    double rot_h = 0;
    for (const auto& s : a.xtl)
        rot_h = std::max(rot_h, (text_width(s, kTickFont) + text_height(kTickFont)) * std::sqrt(0.5));
    f.B = kFigH - (kLayoutPad + text_height(kLabelFont) + kLabelPad + rot_h + kTickPad + kTickLen);
    f.T = kLayoutPad + text_height(kTitleFont) + kTitlePad;
    for (int pass = 0; pass < 2; pass++) {
        if (a.top_extra_frac >= 0) {
            // rotated labels reaching above the axes must stay inside the pad
            const double need = (kLayoutPad + a.top_extra_len - a.top_extra_frac * f.B) /
                                (1 - a.top_extra_frac);
            f.T = std::max(kLayoutPad + text_height(kTitleFont) + kTitlePad, need);
        }
        const YTicks yt = y_ticks(a.ytop, f.B - f.T);
        double yw = 0;
        for (const auto& s : yt.label) yw = std::max(yw, text_width(s, kTickFont));
        f.L = kLayoutPad + text_height(kLabelFont) + kLabelPad + yw + kTickPad + kTickLen;
    }
    return f;
}

// grid, spines, ticks, tick labels, axis labels, title (bars drawn by caller
// before, overlays after)
void draw_axes_decor(Canvas& cv, const Frame& f, const AxesSpec& a,
                     const std::vector<double>& xt_px, const std::vector<double>& xminor_px,
                     double xgrid_alpha, bool draw_grid_first_only) {
    (void)draw_grid_first_only;
    const YTicks yt = y_ticks(a.ytop, f.B - f.T);
    const double glw = 0.6 * kPt;
    // y grid: dashed '--' = [3.7, 1.6] x lw
    for (double v : yt.v) {
        const double y = f.B - v / a.ytop * (f.B - f.T);
        cv.line(f.L, y, f.R, y, glw, kGrid, 0.5f, 3.7 * glw, 1.6 * glw);
    }
    // x grid: dotted ':' = [1, 1.65] x lw
    for (double x : xt_px)
        cv.line(x, f.T, x, f.B, glw, kGrid, (float)xgrid_alpha, 1.0 * glw, 1.65 * glw);
    // spines (0.8 pt)
    const double slw = 0.8 * kPt;
    cv.line(f.L - slw / 2, f.T, f.R + slw / 2, f.T, slw, kBlack, 1);
    cv.line(f.L - slw / 2, f.B, f.R + slw / 2, f.B, slw, kBlack, 1);
    cv.line(f.L, f.T, f.L, f.B, slw, kBlack, 1);
    cv.line(f.R, f.T, f.R, f.B, slw, kBlack, 1);
    // ticks: major 3.5 pt x 0.8 pt, minor 2 pt x 0.6 pt, outward
    for (double x : xt_px) cv.line(x, f.B, x, f.B + kTickLen, 0.8 * kPt, kBlack, 1);
    for (double x : xminor_px) cv.line(x, f.B, x, f.B + 2 * kPt, 0.6 * kPt, kBlack, 1);
    for (size_t k = 0; k < yt.v.size(); k++) {
        const double y = f.B - yt.v[k] / a.ytop * (f.B - f.T);
        cv.line(f.L - kTickLen, y, f.L, y, 0.8 * kPt, kBlack, 1);
        cv.text(yt.label[k], kTickFont, f.L - kTickLen - kTickPad, y, Ha::Right, Va::Center, 0, kBlack);
    }
    double rot_h = 0;
    for (size_t k = 0; k < xt_px.size(); k++) {
        cv.text(a.xtl[k], kTickFont, xt_px[k], f.B + kTickLen + kTickPad, Ha::Right, Va::Top, 45, kBlack);
        rot_h = std::max(rot_h, (text_width(a.xtl[k], kTickFont) + text_height(kTickFont)) * std::sqrt(0.5));
    }
    cv.text(a.xlabel, kLabelFont, (f.L + f.R) / 2, f.B + kTickLen + kTickPad + rot_h + kLabelPad,
            Ha::Center, Va::Top, 0, kBlack);
    double yw = 0;
    for (const auto& s : yt.label) yw = std::max(yw, text_width(s, kTickFont));
    cv.text(a.ylabel, kLabelFont, f.L - kTickLen - kTickPad - yw - kLabelPad, (f.T + f.B) / 2,
            Ha::Right, Va::Center, 90, kBlack);
    cv.text(a.title, kTitleFont, (f.L + f.R) / 2, f.T - kTitlePad, Ha::Center, Va::Baseline, 0, kBlack);
}

// ax.text(0.02, 0.98, "\n".join(lines), va=top, ha=left, fontsize=8.5,
// bbox=round,pad=0.35 white/#cccccc alpha 0.92)
void stats_box(Canvas& cv, const Frame& f, const std::vector<std::string>& lines) {
    const double pt = 8.5, em = pt * kPt, step = 1.2 * (kAsc + kDesc) * em;
    double w = 0;
    for (const auto& s : lines) w = std::max(w, text_width(s, pt));
    const double h = text_height(pt) + step * (double)(lines.size() - 1);
    const double x0 = f.L + 0.02 * (f.R - f.L), y0 = f.B - 0.98 * (f.B - f.T);
    const double pad = 0.35 * em;
    rounded_box(cv, x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad, pad, kWhite, hex(0xcccccc),
                1.0 * kPt, 0.92f);
    for (size_t k = 0; k < lines.size(); k++)
        cv.text(lines[k], pt, x0, y0 + kAsc * em + step * (double)k, Ha::Left, Va::Baseline, 0, kBlack);
}

std::string fmt(const char* f, double v) {
    char b[96];
    std::snprintf(b, sizeof(b), f, v);
    return b;
}

}  // namespace

bool plot_alpha_distribution(const std::string& png_path, const AlphaPlot& p) {
    if (p.edges.size() < 2) return false;
    AxesSpec a;
    a.xlabel = "Gaussian alpha (log scale)";
    a.ylabel = "Percentage of Gaussians (%)";
    a.title = "Final 3DGS Alpha Distribution";
    a.xt.push_back(1.0);
    for (int e = -1; e >= p.log_min_exp; e--) {
        a.xt.push_back(5.0 * std::pow(10.0, e));
        a.xt.push_back(std::pow(10.0, e));
    }
    for (double v : a.xt) a.xtl.push_back(fmt("%g", v));
    double maxp = 0;
    for (double v : p.percent) maxp = std::max(maxp, v);
    const double y_max = std::max(maxp, 1.0);
    a.ytop = y_max * 1.35;
    std::vector<std::string> thr_label;
    for (size_t k = 0; k < p.thresholds.size(); k++) {
        if (p.thresholds[k] < p.edges[0]) { thr_label.emplace_back(); continue; }
        thr_label.push_back(">" + fmt("%g", p.thresholds[k]) + ": " + fmt("%.2f", p.threshold_percent[k]) + "%");
        a.top_extra_len = std::max(a.top_extra_len, text_width(thr_label.back(), 8));
    }
    a.top_extra_frac = 1.03 / 1.35;
    const Frame f = layout(a);
    // xlim(1.0, edges[0]): log axis, reversed
    const double lmin = std::log10(p.edges[0]);
    const auto X = [&](double v) { return f.L + (0.0 - std::log10(v)) / (0.0 - lmin) * (f.R - f.L); };
    const auto Y = [&](double v) { return f.B - v / a.ytop * (f.B - f.T); };

    Canvas cv;
    // bars (zorder 1): #377eb8, white 1 pt edges
    for (size_t b = 0; b + 1 < p.edges.size(); b++) {
        if (p.percent[b] <= 0) continue;
        cv.rect(X(p.edges[b + 1]), Y(p.percent[b]), X(p.edges[b]), f.B, hex(0x377eb8), 1);
    }
    const double elw = 1.0 * kPt;
    for (size_t b = 0; b + 1 < p.edges.size(); b++) {
        if (p.percent[b] <= 0) continue;
        const double xl = X(p.edges[b + 1]), xr = X(p.edges[b]), yt = Y(p.percent[b]);
        cv.line(xl, yt, xl, f.B, elw, kWhite, 1);
        cv.line(xr, yt, xr, f.B, elw, kWhite, 1);
        cv.line(xl, yt, xr, yt, elw, kWhite, 1);
    }
    std::vector<double> xt_px, minor_px;
    for (double v : a.xt) xt_px.push_back(X(v));
    for (int e = p.log_min_exp; e < 0; e++)
        for (int m = 2; m <= 9; m++)
            if (m != 5) minor_px.push_back(X(m * std::pow(10.0, e)));
    draw_axes_decor(cv, f, a, xt_px, minor_px, 0.4, false);
    // threshold markers: axvline #e41a1c '--' 0.9 pt alpha 0.65 + vertical label
    const double tlw = 0.9 * kPt;
    for (size_t k = 0; k < p.thresholds.size(); k++) {
        if (thr_label[k].empty()) continue;
        const double x = X(p.thresholds[k]);
        cv.line(x, f.T, x, f.B, tlw, hex(0xe41a1c), 0.65f, 3.7 * tlw, 1.6 * tlw);
        cv.text(thr_label[k], 8, x, Y(y_max * 1.03), Ha::Center, Va::Bottom, 90, hex(0x7f0000));
    }
    std::vector<std::string> lines = {"total GS: " + std::to_string(p.total),
                                      "zero alpha: " + fmt("%.2f", p.zero_percent) + "%"};
    for (size_t k = 0; k < p.thresholds.size(); k++)
        lines.push_back("alpha > " + fmt("%g", p.thresholds[k]) + ": " + fmt("%.2f", p.threshold_percent[k]) + "%");
    stats_box(cv, f, lines);
    return cv.save(png_path);
}

bool plot_scale_distribution(const std::string& png_path, const ScalePlot& p) {
    const int nb = (int)p.percent.size();
    if (nb <= 0 || !(p.max_scale > 0)) return false;
    AxesSpec a;
    a.xlabel = "Gaussian scale component";
    a.ylabel = "Percentage of Gaussians (%)";
    a.title = "Final 3DGS Scale Distribution";
    for (int k = 0; k <= 10; k++) a.xt.push_back(k == 10 ? p.max_scale : k * (p.max_scale / 10.0));
    for (double v : a.xt) a.xtl.push_back(fmt("%g", v));
    double maxp = 0;
    for (double v : p.percent) maxp = std::max(maxp, v);
    a.ytop = std::max(maxp, 1.0) * 1.35;
    const Frame f = layout(a);
    const auto X = [&](double v) { return f.L + v / p.max_scale * (f.R - f.L); };
    const auto Y = [&](double v) { return f.B - v / a.ytop * (f.B - f.T); };

    Canvas cv;
    // ax.bar(edges[:-1], pct, width=max/50, align="edge"): #4daf4a, white 0.4 pt
    const double bw = p.max_scale / (double)nb, elw = 0.4 * kPt;
    for (int b = 0; b < nb; b++)
        if (p.percent[(size_t)b] > 0)
            cv.rect(X(p.edges[(size_t)b]), Y(p.percent[(size_t)b]), X(p.edges[(size_t)b] + bw), f.B,
                    hex(0x4daf4a), 1);
    for (int b = 0; b < nb; b++) {
        if (p.percent[(size_t)b] <= 0) continue;
        const double xl = X(p.edges[(size_t)b]), xr = X(p.edges[(size_t)b] + bw), yt = Y(p.percent[(size_t)b]);
        cv.line(xl, yt, xl, f.B, elw, kWhite, 1);
        cv.line(xr, yt, xr, f.B, elw, kWhite, 1);
        cv.line(xl, yt, xr, yt, elw, kWhite, 1);
    }
    std::vector<double> xt_px;
    for (double v : a.xt) xt_px.push_back(X(v));
    draw_axes_decor(cv, f, a, xt_px, {}, 0.25, false);
    std::vector<std::string> lines = {
        "total scale values: " + std::to_string(p.total),
        "max scale config: " + fmt("%g", p.max_scale),
        "zero scale: " + fmt("%.2f", p.zero_percent) + "%",
        "> max scale: " + fmt("%.2f", p.above_percent) + "%"};
    // np.linspace(0, nb - 1, 10, dtype=int): truncated, last = nb - 1, deduped
    std::vector<int> idx;
    for (int k = 0; k < 10; k++) {
        const int v = k == 9 ? nb - 1 : (int)((double)k * ((double)(nb - 1) / 9.0));
        if (idx.empty() || idx.back() != v) idx.push_back(v);
    }
    for (int i : idx)
        lines.push_back("scale <= " + fmt("%g", p.edges[(size_t)i + 1]) + ": " +
                        fmt("%.2f", p.cumulative_percent[(size_t)i]) + "%");
    stats_box(cv, f, lines);
    return cv.save(png_path);
}

}  // namespace qs
