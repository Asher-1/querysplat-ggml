// Python-format I/O shared by the official-layout writers (output_parity.cpp)
// and the DL3DV evaluator (eval.cpp):
//   * JSON: `json.dump(obj, f, indent=2)` byte layout (floats in python repr,
//     ensure_ascii strings, insertion-ordered objects) + a strict RFC 8259
//     reader for the evaluation split JSONs;
//   * .npy / .npz: numpy.lib.format v1.0 headers in a stored zip (np.savez).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qs {

struct Json {
    enum Kind { Null, Bool, Int, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    long long i = 0;
    double d = 0;
    std::string s;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;  // insertion order

    Json& add(const std::string& key, Json v) {
        obj.emplace_back(key, std::move(v));
        return *this;
    }
    Json& push(Json v) {
        arr.push_back(std::move(v));
        return *this;
    }
    // reader helpers: member lookup (nullptr if absent / not an object) and
    // numeric value of an Int or Num
    const Json* get(const std::string& key) const;
    double number() const { return kind == Int ? (double)i : d; }
};

Json jnull();
Json jbool(bool v);
Json jint(long long v);
Json jnum(double v);
Json jstr(const std::string& v);
Json jarr();
Json jobj();

// float.__repr__ (shortest round-trip, CPython layout rules)
std::string py_repr(double v);
// f"{v:g}"
std::string py_g(double v);
std::vector<uint32_t> utf8_codepoints(const std::string& s);
std::string json_quote(const std::string& s);
void json_dump(const Json& j, int level, std::string& out);
bool write_text(const std::string& path, const std::string& text);
// json.dump(obj, f, indent=2); f.write("\n")
bool write_json(const std::string& path, const Json& j);
// json.loads: false + *err on malformed input
bool parse_json(const std::string& text, Json* out, std::string* err);

// tensor.tolist() helpers
Json f32_list(const float* v, int n);
Json mat_list(const float* m, int rows, int cols);
Json int_list(const std::vector<long long>& v);

std::string npy_bytes(const std::string& descr, const std::vector<long long>& shape,
                      const void* data, size_t nbytes);
std::string npy_f32(const float* data, const std::vector<long long>& shape);
std::string npy_str(const std::vector<std::string>& items);
uint32_t crc32(const std::string& data);
// np.savez(path, **members): members are (key, .npy payload) in keyword order
bool write_npz(const std::string& path,
               const std::vector<std::pair<std::string, std::string>>& members);

}  // namespace qs
