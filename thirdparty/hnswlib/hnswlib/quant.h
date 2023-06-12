#pragma once

#include <cstdint>

#include "hnswlib/hnswlib/tensor.h"
#include "index/vector_index/helpers/FaissIO.h"

namespace hnswlib {

template <int32_t METRIC, int32_t AlignWidth, int32_t NBits>
struct Quantizer {
    static_assert(AlignWidth * NBits % 8 == 0);

    constexpr static int32_t metric = METRIC;
    constexpr static int32_t align_width = AlignWidth;
    constexpr static int32_t nbits = NBits;

    Tensor storage;

    Quantizer() = default;

    explicit Quantizer(int32_t dim) : storage(dim, nbits, align_width) {
    }

    auto
    dim() const {
        return storage.dim();
    }

    auto
    dim_align() const {
        return storage.dim_align();
    }

    auto
    code_size() const {
        return storage.code_size();
    }

    auto
    get_code(int32_t u) const {
        return storage.get(u);
    }
};

}  // namespace hnswlib
