#pragma once

#include "hnswlib/hnswlib/bf16.h"
#include "hnswlib/hnswlib/computer.h"
#include "hnswlib/hnswlib/quant.h"

namespace hnswlib {

#if defined(__AVX512BF16__)
constexpr int32_t bf16_align = 32;
#else
constexpr int32_t bf16_align = 16;
#endif

template <int32_t metric, typename Template = Quantizer<metric, bf16_align, 16>>
struct BF16Quantizer : Template {
    using type = BF16Quantizer;
    using data_type = bf16;
    constexpr static bool enable_refine = false;

    BF16Quantizer() = default;

    explicit BF16Quantizer(int dim) : Template(dim) {
    }

    void
    train(const float*, int32_t) {
    }

    void
    add(const float* data, int32_t n) {
        this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
        for (int32_t i = 0; i < n; ++i) {
            encode(data + (int64_t)i * this->dim(), (data_type*)this->get_code(i));
        }
    }

    void
    encode(const float* from, data_type* to) const {
        for (int i = 0; i < this->dim(); ++i) {
            to[i] = data_type(from[i]);
        }
    }

    void
    decode(const data_type* from, float* to) const {
        for (int i = 0; i < this->dim(); ++i) {
            to[i] = float(from[i]);
        }
    }

    bool
    check_query(const float* q) const {
        return true;
    }

    void
    load(knowhere::MemoryIOReader& input) {
        this->storage.load(input);
    }

    void
    save(knowhere::MemoryIOWriter& output) {
        this->storage.save(output);
    }

    constexpr static auto dist_func = metric == 0 ? L2SqrBF16 : IPBF16;

    constexpr static auto dist_func_sym = metric == 0 ? L2SqrBF16Sym : IPBF16Sym;

    using ComputerType = ComputerImpl<dist_func, float, float, float, bf16>;
    using SymComputerType = SymComputerImpl<dist_func_sym, float, bf16>;

    auto
    get_computer(const float* query) const {
        return ComputerType(this->storage, query, MemCpyTag{});
    }

    auto
    get_sym_computer() const {
        return SymComputerType(this->storage);
    }

    auto
    get_accurate_computer(const float* query) const {
        return get_computer(query);
    }
};

}  // namespace hnswlib
