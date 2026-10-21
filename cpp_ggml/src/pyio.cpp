// Python-format I/O — implementation. See pyio.hpp.

#include "pyio.hpp"

#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace qs {

// ---- python json.dump(obj, f, indent=2) -------------------------------------

Json jnull() { return Json{}; }
Json jbool(bool v) { Json j; j.kind = Json::Bool; j.b = v; return j; }
Json jint(long long v) { Json j; j.kind = Json::Int; j.i = v; return j; }
Json jnum(double v) { Json j; j.kind = Json::Num; j.d = v; return j; }
Json jstr(const std::string& v) { Json j; j.kind = Json::Str; j.s = v; return j; }
Json jarr() { Json j; j.kind = Json::Arr; return j; }
Json jobj() { Json j; j.kind = Json::Obj; return j; }

// float.__repr__: shortest round-trip digits (to_chars == David Gay mode 0),
// laid out by CPython's format_float_short rules for 'r': exponent form iff
// decpt <= -4 or decpt > 16, "e" + sign + >= 2 exponent digits, ".0" added
// to integral fixed-form values.
std::string py_repr(double v) {
    if (std::isnan(v)) return "NaN";  // json.dump spellings
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
    std::string s(buf, r.ptr);
    const bool neg = s[0] == '-';
    if (neg) s.erase(0, 1);
    const size_t epos = s.find('e');
    std::string digits;
    for (size_t k = 0; k < epos; k++)
        if (s[k] != '.') digits += s[k];
    const int decpt = std::atoi(s.c_str() + epos + 1) + 1;
    const int nd = (int)digits.size();
    std::string out;
    if (decpt <= -4 || decpt > 16) {
        out = digits.substr(0, 1);
        if (nd > 1) out += "." + digits.substr(1);
        const int e = decpt - 1;
        char eb[16];
        std::snprintf(eb, sizeof(eb), "e%c%02d", e < 0 ? '-' : '+', std::abs(e));
        out += eb;
    } else if (decpt <= 0) {
        out = "0." + std::string((size_t)-decpt, '0') + digits;
    } else if (decpt < nd) {
        out = digits.substr(0, (size_t)decpt) + "." + digits.substr((size_t)decpt);
    } else {
        out = digits + std::string((size_t)(decpt - nd), '0') + ".0";
    }
    return neg ? "-" + out : out;
}

// f"{v:g}" == printf("%g") (precision 6, same exponent/trailing-zero rules)
std::string py_g(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%g", v);
    return b;
}

// UTF-8 -> code points (invalid bytes pass through as their byte value,
// close to python's surrogateescape for the ASCII names this sees in practice)
std::vector<uint32_t> utf8_codepoints(const std::string& s) {
    std::vector<uint32_t> cps;
    for (size_t k = 0; k < s.size();) {
        const unsigned char c = (unsigned char)s[k];
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (len == 0 || k + len > s.size()) { cps.push_back(c); k++; continue; }
        uint32_t cp = len == 1 ? c : c & (0x7F >> len);
        for (int m = 1; m < len; m++) cp = (cp << 6) | ((unsigned char)s[k + m] & 0x3F);
        cps.push_back(cp);
        k += (size_t)len;
    }
    return cps;
}

// json ensure_ascii string encoding
std::string json_quote(const std::string& s) {
    std::string out = "\"";
    char b[16];
    for (uint32_t cp : utf8_codepoints(s)) {
        switch (cp) {
            case '"': out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\n': out += "\\n"; continue;
            case '\r': out += "\\r"; continue;
            case '\t': out += "\\t"; continue;
            case '\b': out += "\\b"; continue;
            case '\f': out += "\\f"; continue;
            default: break;
        }
        if (cp >= 0x20 && cp <= 0x7E) {
            out += (char)cp;
        } else if (cp > 0xFFFF) {
            const uint32_t v = cp - 0x10000;
            std::snprintf(b, sizeof(b), "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
            out += b;
        } else {
            std::snprintf(b, sizeof(b), "\\u%04x", cp);
            out += b;
        }
    }
    return out + "\"";
}

void json_dump(const Json& j, int level, std::string& out) {
    const auto nl = [&](int l) { out += '\n'; out.append((size_t)l * 2, ' '); };
    switch (j.kind) {
        case Json::Null: out += "null"; return;
        case Json::Bool: out += j.b ? "true" : "false"; return;
        case Json::Int: out += std::to_string(j.i); return;
        case Json::Num: out += py_repr(j.d); return;
        case Json::Str: out += json_quote(j.s); return;
        case Json::Arr:
            if (j.arr.empty()) { out += "[]"; return; }
            out += '[';
            for (size_t k = 0; k < j.arr.size(); k++) {
                nl(level + 1);
                json_dump(j.arr[k], level + 1, out);
                if (k + 1 < j.arr.size()) out += ',';
            }
            nl(level);
            out += ']';
            return;
        case Json::Obj:
            if (j.obj.empty()) { out += "{}"; return; }
            out += '{';
            for (size_t k = 0; k < j.obj.size(); k++) {
                nl(level + 1);
                out += json_quote(j.obj[k].first) + ": ";
                json_dump(j.obj[k].second, level + 1, out);
                if (k + 1 < j.obj.size()) out += ',';
            }
            nl(level);
            out += '}';
            return;
    }
}

bool write_text(const std::string& path, const std::string& text) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    return std::fclose(f) == 0 && ok;
}

// json.dump(obj, f, indent=2); f.write("\n")
bool write_json(const std::string& path, const Json& j) {
    std::string out;
    json_dump(j, 0, out);
    return write_text(path, out + "\n");
}

// tensor.tolist() of float32 values: python floats holding the exact f32
Json f32_list(const float* v, int n) {
    Json a = jarr();
    for (int k = 0; k < n; k++) a.push(jnum((double)v[k]));
    return a;
}
Json mat_list(const float* m, int rows, int cols) {
    Json a = jarr();
    for (int r = 0; r < rows; r++) a.push(f32_list(m + r * cols, cols));
    return a;
}
Json int_list(const std::vector<long long>& v) {
    Json a = jarr();
    for (long long x : v) a.push(jint(x));
    return a;
}

// ---- .npy / .npz -------------------------------------------------------------

// numpy.lib.format.write_array (v1.0): sorted-key dict repr, 21-digit growth
// spare after the dict, '\n'-terminated header padded to a 64-byte boundary.
std::string npy_bytes(const std::string& descr, const std::vector<long long>& shape,
                      const void* data, size_t nbytes) {
    std::string shp = "(";
    for (size_t k = 0; k < shape.size(); k++) {
        if (k) shp += ", ";
        shp += std::to_string(shape[k]);
    }
    shp += shape.size() == 1 ? ",)" : ")";
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shp + ", }";
    if (!shape.empty()) header.append((size_t)(21 - (int)std::to_string(shape[0]).size()), ' ');
    const size_t hlen = header.size() + 1;
    const size_t padlen = 64 - ((8 + 2 + hlen) % 64);
    const size_t total = hlen + padlen;
    std::string out = "\x93NUMPY";
    out += (char)1;
    out += (char)0;
    out += (char)(total & 0xFF);
    out += (char)((total >> 8) & 0xFF);
    out += header;
    out.append(padlen, ' ');
    out += '\n';
    out.append(static_cast<const char*>(data), nbytes);
    return out;
}

std::string npy_f32(const float* data, const std::vector<long long>& shape) {
    long long n = 1;
    for (long long d : shape) n *= d;
    return npy_bytes("<f4", shape, data, (size_t)n * 4);
}

// np.asarray(list_of_str): '<U{max code points}' (>= 1), UTF-32LE zero-padded
std::string npy_str(const std::vector<std::string>& items) {
    std::vector<std::vector<uint32_t>> cps;
    size_t width = 1;
    for (const auto& s : items) {
        cps.push_back(utf8_codepoints(s));
        width = std::max(width, cps.back().size());
    }
    std::vector<uint32_t> buf(items.size() * width, 0);
    for (size_t k = 0; k < cps.size(); k++)
        std::copy(cps[k].begin(), cps[k].end(), buf.begin() + (long)(k * width));
    return npy_bytes("<U" + std::to_string(width), {(long long)items.size()},
                     buf.data(), buf.size() * 4);
}

uint32_t crc32(const std::string& data) {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }();
    uint32_t c = 0xFFFFFFFFu;
    for (unsigned char b : data) c = table[(c ^ b) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static void put16(std::string& o, uint32_t v) { o += (char)(v & 0xFF); o += (char)((v >> 8) & 0xFF); }
static void put32(std::string& o, uint32_t v) { put16(o, v & 0xFFFF); put16(o, v >> 16); }

// np.savez layout: a zip whose members are "<key>.npy", in keyword order.
// Stored (np.savez) — np.load reads stored and deflated (savez_compressed)
// members identically; the arrays, not the container bytes, are the contract.
bool write_npz(const std::string& path,
               const std::vector<std::pair<std::string, std::string>>& members) {
    std::string zip, central;
    for (const auto& [key, payload] : members) {
        const std::string name = key + ".npy";
        const uint32_t crc = crc32(payload);
        const uint32_t offset = (uint32_t)zip.size();
        put32(zip, 0x04034b50); put16(zip, 20); put16(zip, 0); put16(zip, 0);
        put16(zip, 0); put16(zip, 0x21);  // 1980-01-01 00:00 (zipfile default)
        put32(zip, crc); put32(zip, (uint32_t)payload.size()); put32(zip, (uint32_t)payload.size());
        put16(zip, (uint32_t)name.size()); put16(zip, 0);
        zip += name;
        zip += payload;
        put32(central, 0x02014b50); put16(central, 20); put16(central, 20);
        put16(central, 0); put16(central, 0); put16(central, 0); put16(central, 0x21);
        put32(central, crc); put32(central, (uint32_t)payload.size());
        put32(central, (uint32_t)payload.size());
        put16(central, (uint32_t)name.size()); put16(central, 0); put16(central, 0);
        put16(central, 0); put16(central, 0); put32(central, 0600u << 16);
        put32(central, offset);
        central += name;
    }
    const uint32_t cd_offset = (uint32_t)zip.size();
    zip += central;
    put32(zip, 0x06054b50); put16(zip, 0); put16(zip, 0);
    put16(zip, (uint32_t)members.size()); put16(zip, (uint32_t)members.size());
    put32(zip, (uint32_t)central.size()); put32(zip, cd_offset); put16(zip, 0);
    return write_text(path, zip);
}

// ---- json.loads (strict RFC 8259; the split JSONs are plain ASCII) ---------

const Json* Json::get(const std::string& key) const {
    if (kind != Obj) return nullptr;
    for (const auto& kv : obj)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

namespace {

struct JsonReader {
    const std::string& t;
    size_t p = 0;
    std::string err;

    void ws() {
        while (p < t.size() && (t[p] == ' ' || t[p] == '\t' || t[p] == '\n' || t[p] == '\r')) p++;
    }
    bool fail(const char* what) {
        if (err.empty()) err = std::string(what) + " at offset " + std::to_string(p);
        return false;
    }
    bool lit(const char* word) {
        const size_t n = std::char_traits<char>::length(word);
        if (t.compare(p, n, word) != 0) return fail("invalid literal");
        p += n;
        return true;
    }
    static void put_utf8(std::string& o, uint32_t cp) {
        if (cp < 0x80) { o += (char)cp; return; }
        if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); }
        else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); }
        else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); }
        o += (char)(0x80 | (cp & 0x3F));
    }
    bool hex4(uint32_t* v) {
        if (p + 4 > t.size()) return fail("truncated \\u escape");
        *v = 0;
        for (int k = 0; k < 4; k++) {
            const char c = t[p++];
            *v <<= 4;
            if (c >= '0' && c <= '9') *v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') *v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') *v |= (uint32_t)(c - 'A' + 10);
            else return fail("bad \\u escape");
        }
        return true;
    }
    bool str(std::string* o) {
        p++;  // opening quote
        while (p < t.size() && t[p] != '"') {
            const char c = t[p++];
            if ((unsigned char)c < 0x20) return fail("control character in string");
            if (c != '\\') { *o += c; continue; }
            if (p >= t.size()) break;
            const char e = t[p++];
            switch (e) {
                case '"': *o += '"'; break;
                case '\\': *o += '\\'; break;
                case '/': *o += '/'; break;
                case 'b': *o += '\b'; break;
                case 'f': *o += '\f'; break;
                case 'n': *o += '\n'; break;
                case 'r': *o += '\r'; break;
                case 't': *o += '\t'; break;
                case 'u': {
                    uint32_t cp = 0, lo = 0;
                    if (!hex4(&cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00 && t.compare(p, 2, "\\u") == 0) {
                        p += 2;
                        if (!hex4(&lo)) return false;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(*o, cp);
                    break;
                }
                default: return fail("bad escape");
            }
        }
        if (p >= t.size()) return fail("unterminated string");
        p++;
        return true;
    }
    bool num(Json* o) {
        const size_t b = p;
        bool integral = true;
        if (t[p] == '-') p++;
        while (p < t.size() && std::isdigit((unsigned char)t[p])) p++;
        if (p < t.size() && t[p] == '.') { integral = false; p++; while (p < t.size() && std::isdigit((unsigned char)t[p])) p++; }
        if (p < t.size() && (t[p] == 'e' || t[p] == 'E')) {
            integral = false;
            p++;
            if (p < t.size() && (t[p] == '+' || t[p] == '-')) p++;
            while (p < t.size() && std::isdigit((unsigned char)t[p])) p++;
        }
        const std::string s = t.substr(b, p - b);
        if (s.empty() || s == "-") return fail("bad number");
        if (integral) { *o = jint(std::strtoll(s.c_str(), nullptr, 10)); return true; }
        *o = jnum(std::strtod(s.c_str(), nullptr));
        return true;
    }
    bool value(Json* o, int depth) {
        if (depth > 256) return fail("nesting too deep");
        ws();
        if (p >= t.size()) return fail("unexpected end");
        const char c = t[p];
        if (c == '{') {
            *o = jobj();
            p++;
            ws();
            if (p < t.size() && t[p] == '}') { p++; return true; }
            for (;;) {
                ws();
                if (p >= t.size() || t[p] != '"') return fail("expected key");
                std::string key;
                if (!str(&key)) return false;
                ws();
                if (p >= t.size() || t[p] != ':') return fail("expected ':'");
                p++;
                Json v;
                if (!value(&v, depth + 1)) return false;
                o->add(key, std::move(v));
                ws();
                if (p < t.size() && t[p] == ',') { p++; continue; }
                if (p < t.size() && t[p] == '}') { p++; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            *o = jarr();
            p++;
            ws();
            if (p < t.size() && t[p] == ']') { p++; return true; }
            for (;;) {
                Json v;
                if (!value(&v, depth + 1)) return false;
                o->push(std::move(v));
                ws();
                if (p < t.size() && t[p] == ',') { p++; continue; }
                if (p < t.size() && t[p] == ']') { p++; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { *o = jstr(""); return str(&o->s); }
        if (c == 't') { *o = jbool(true); return lit("true"); }
        if (c == 'f') { *o = jbool(false); return lit("false"); }
        if (c == 'n') { *o = jnull(); return lit("null"); }
        if (c == '-' || std::isdigit((unsigned char)c)) return num(o);
        return fail("unexpected character");
    }
};

}  // namespace

bool parse_json(const std::string& text, Json* out, std::string* err) {
    JsonReader r{text, 0, {}};
    bool ok = r.value(out, 0);
    if (ok) {
        r.ws();
        if (r.p != text.size()) ok = r.fail("trailing data");
    }
    if (!ok && err) *err = r.err;
    return ok;
}

}  // namespace qs
