#pragma once

#include <cmath>

#include "common/Log.h"
#include "hnswlib/hnswlib/bf16_quant.h"
#include "hnswlib/hnswlib/calibrator.h"
#include "hnswlib/hnswlib/quant.h"
#include "hnswlib/hnswlib/simd.h"

namespace hnswlib {

template <int32_t metric, typename Template = Quantizer<metric, 128, 4>>
struct SQ4Quantizer : Template {
    using type = SQ4Quantizer;
    using data_type = uint8_t;

    constexpr static float drop_ratio = 0.01f;
    constexpr static uint64_t ncount = 15;

    constexpr static bool enable_refine = true;

    using Calibrator = AffineCalibrator;
    Calibrator calibrator;

    using Refiner = BF16Quantizer<metric>;
    Refiner refiner;

    SQ4Quantizer() = default;

    explicit SQ4Quantizer(int dim) : Template(dim), calibrator(dim), refiner(dim) {
    }

    void
    train(const float* data, int32_t n) {
        calibrator.calibrate(data, (int64_t)n * this->dim(), drop_ratio);
        refiner.train(data, n);
    }

    void
    add(const float* data, int32_t n) {
        this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
        for (int32_t i = 0; i < n; ++i) {
            encode(data + (int64_t)i * this->dim(), (data_type*)this->get_code(i));
        }
        refiner.add(data, n);
    }

    void
    encode(const float* from, data_type* to) const {
        memset(to, 0, (this->dim() + 1) / 2);
        for (int j = 0; j < this->dim(); ++j) {
            float x = calibrator.transform(from[j]);
            uint8_t y = std::round(x * ncount);
            if (j & 1) {
                to[j / 2] |= y << 4;
            } else {
                to[j / 2] |= y;
            }
        }
    }

    void
    decode(const data_type* from, float* to) const {
        for (int j = 0; j < this->dim(); ++j) {
            uint8_t y;
            if (j & 1) {
                y = from[j / 2] >> 4 & 15;
            } else {
                y = from[j / 2] & 15;
            }
            float x = float(y) / ncount;
            to[j] = calibrator.transform_back(x);
        }
    }

    bool
    check_query(const float* q) const {
        // return calibrator.check_vec(q);
        return true;
    }

    void
    load(knowhere::MemoryIOReader& input) {
        this->storage.load(input);
        calibrator.load(input);
        refiner.load(input);
    }

    void
    save(knowhere::MemoryIOWriter& output) {
        this->storage.save(output);
        calibrator.save(output);
        refiner.save(output);
    }

    constexpr static auto dist_func = L2SqrSQ4Sym;

    using ComputerType = ComputerImpl<dist_func, int32_t, float, uint8_t, uint8_t>;
    using SymComputerType = SymComputerImpl<dist_func, int32_t, uint8_t>;

    auto
    get_computer(const float* query) const {
        return ComputerType(this->storage, query, [this](const float* from, data_type* to) { this->encode(from, to); });
    }

    auto
    get_sym_computer() const {
        // return SymComputerType(this->storage);
        return refiner.get_sym_computer();
    }

    auto
    get_accurate_computer(const float* query) const {
        return refiner.get_accurate_computer(query);
    }
};

}  // namespace hnswlib
