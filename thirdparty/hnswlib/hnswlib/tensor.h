#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <variant>

#include "hnswlib/hnswlib/hnswlib.h"
#include "hnswlib/hnswlib/simd.h"
#include "index/vector_index/helpers/FaissIO.h"

namespace hnswlib {

inline void*
align2M(size_t nbytes, uint8_t x = 0) {
    size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
    auto p = std::aligned_alloc(1 << 21, len);
    std::memset(p, x, len);
    return p;
}

inline void*
alloc64B(size_t nbytes, uint8_t x = 0) {
    size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
    auto p = std::aligned_alloc(1 << 6, len);
    std::memset(p, x, len);
    return p;
}

inline void*
align_alloc(size_t nbytes, uint8_t x = 0) {
    if (nbytes > 2 * 1024 * 1024) {
        return align2M(nbytes, x);
    } else {
        return alloc64B(nbytes, x);
    }
}

struct Tensor {
    int32_t align_width;
    int32_t nbits;

    int32_t nb = 0;
    int32_t d = 0;
    int32_t dalign = 0;
    int32_t csize = 0;
    char* codes = nullptr;

    Tensor() = default;

    Tensor(int32_t dim, int32_t nbits, int32_t align_width)
        : d(dim), dalign((dim + align_width - 1) / align_width * align_width), csize(dalign * nbits / 8) {
    }

    ~Tensor() {
        free(codes);
    }

    Tensor(int32_t n, int32_t dim, int32_t nbits, int32_t align_width) : Tensor(dim, nbits, align_width) {
        init(n);
    }

    Tensor(const Tensor& rhs) = delete;

    Tensor(Tensor&& rhs) {
        swap(*this, rhs);
    }

    Tensor&
    operator=(const Tensor& rhs) = delete;

    Tensor&
    operator=(Tensor&& rhs) {
        swap(*this, rhs);
        return *this;
    }

    friend void
    swap(Tensor& lhs, Tensor& rhs) {
        using std::swap;
        swap(lhs.nb, rhs.nb);
        swap(lhs.d, rhs.d);
        swap(lhs.dalign, rhs.dalign);
        swap(lhs.csize, rhs.csize);
        swap(lhs.codes, rhs.codes);
    }

    void
    init(int32_t n) {
        nb = n;
        this->codes = (char*)align_alloc((int64_t)n * csize);
    }

    char*
    get(int32_t u) const {
        return codes + (int64_t)u * csize;
    }

    void
    add(int32_t u, const char* x) {
        memcpy(get(u), x, d * 4);
    }

    void
    prefetch(int32_t u, int32_t num) const {
        mem_prefetch(get(u), num);
    }

    int32_t
    size() const {
        return nb;
    }
    int32_t
    dim() const {
        return d;
    }
    int32_t
    dim_align() const {
        return dalign;
    }
    int32_t
    code_size() const {
        return csize;
    }

    void
    load(knowhere::MemoryIOReader& input) {
        readBinaryPOD(input, nb);
        readBinaryPOD(input, d);
        readBinaryPOD(input, dalign);
        readBinaryPOD(input, csize);
        init(nb);
        input.read(codes, csize * nb);
    }

    void
    save(knowhere::MemoryIOWriter& output) {
        writeBinaryPOD(output, nb);
        writeBinaryPOD(output, d);
        writeBinaryPOD(output, dalign);
        writeBinaryPOD(output, csize);
        output.write(codes, csize * nb);
    }
};

}  // namespace hnswlib