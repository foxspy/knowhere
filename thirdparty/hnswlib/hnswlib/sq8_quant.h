#pragma once

#include <cmath>

#include "common/Log.h"
#include "hnswlib/hnswlib/bf16_quant.h"
#include "hnswlib/hnswlib/calibrator.h"
#include "hnswlib/hnswlib/quant.h"
#include "hnswlib/hnswlib/simd.h"

namespace hnswlib {

template <int32_t metric, typename Template = Quantizer<metric, 64, 8>>
struct SQ8Quantizer : Template {
    using type = SQ8Quantizer;
    using data_type = int8_t;

    constexpr static float drop_ratio = 0.01f;
    constexpr static uint64_t ncount = 127;

    constexpr static bool enable_refine = true;

    using Calibrator = SymCalibrator;
    Calibrator calibrator;

    using Refiner = BF16Quantizer<metric>;
    Refiner refiner;

    SQ8Quantizer() = default;

    explicit SQ8Quantizer(int dim) : Template(dim), calibrator(dim), refiner(dim) {
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
        for (int j = 0; j < this->dim(); ++j) {
            float x = calibrator.transform(from[j]);
            int32_t y = std::round(x * ncount);
            to[j] = y;
        }
    }

    void
    decode(const data_type* from, float* to) const {
        for (int j = 0; j < this->dim(); ++j) {
            float x = float(from[j]) / ncount;
            to[j] = calibrator.transform_back(x);
        }
    }

    bool
    check_query(const float* q) const {
        return calibrator.check_vec(q);
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

    constexpr static auto dist_func = metric == 0 ? L2SqrSQ8 : IPSQ8;

    using ComputerType = ComputerImpl<dist_func, int32_t, float, int8_t, int8_t>;
    using SymComputerType = SymComputerImpl<dist_func, int32_t, int8_t>;

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
