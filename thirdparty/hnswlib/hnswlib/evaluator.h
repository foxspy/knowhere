#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "common/Log.h"
#include "hnswlib/hnswlib/sq4_quant.h"
#include "hnswlib/hnswlib/sq8_quant.h"

namespace hnswlib {

template <typename QuantType>
struct Evaluator {
    const QuantType& quant;

    explicit Evaluator(const QuantType& quant) : quant(quant) {
    }

    double
    evaluate(const float* data, size_t n) {
        std::vector<typename QuantType::data_type> code(quant.code_size());
        std::vector<float> fvec(quant.dim());
        std::vector<double> errs(n);
        for (size_t i = 0; i < n; ++i) {
            const float* cur_vec = data + i * quant.dim();
            quant.encode(cur_vec, code.data());
            quant.decode(code.data(), fvec.data());
            for (int32_t j = 0; j < quant.dim(); ++j) {
                fvec[j] -= cur_vec[j];
            }
            double d0 = 0.0, d1 = 0.0;
            for (int32_t j = 0; j < quant.dim(); ++j) {
                d0 += cur_vec[j] * cur_vec[j];
                d1 += fvec[j] * fvec[j];
            }
            errs[i] = d1 / (d0 + 1e-9);
        }
        std::sort(errs.rbegin(), errs.rend());
        double err_p95 = errs[n * 0.05];
        return err_p95;
    }
};

inline bool
evaluate_sq4(const float* data, size_t n, size_t d) {
    // constexpr double sq4_threshold = 0.05;
    // SQ4Quantizer<0> quant(d);
    // quant.train(data, n);
    // double err = Evaluator(quant).evaluate(data, n);
    // printf("sq4 err: %lf, %s\n", err, err < sq4_threshold ? "use sq4" : "try next");
    // return err < sq4_threshold;
    return true;
}

inline bool
evaluate_sq8(const float* data, size_t n, size_t d) {
    // constexpr double sq8_threshold = 0.05;
    // SQ8Quantizer<0> quant(d);
    // quant.train(data, n);
    // double err = Evaluator(quant).evaluate(data, n);
    // printf("sq8 err: %lf, %s\n", err, err < sq8_threshold ? "use sq8" : "try next");
    // return err < sq8_threshold;
    return true;
}

}  // namespace hnswlib
