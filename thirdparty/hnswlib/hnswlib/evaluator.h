#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "common/Log.h"
#include "hnswlib/hnswlib/sq4_quant.h"
#include "hnswlib/hnswlib/sq8_quant.h"

namespace hnswlib {

inline double
imbalance_factor(int k, const int64_t* hist) {
    double tot = 0, uf = 0;

    for (int i = 0; i < k; i++) {
        tot += hist[i];
        uf += hist[i] * (double)hist[i];
    }
    return uf / tot / tot;
}

inline bool
evaluate_sq4(const float* data, size_t n, size_t d) {
    constexpr double sq4_threshold = 0.5;
    SQ4Quantizer<0> quant(d);
    quant.train(data, n);
    constexpr int32_t ncount = 16;
    std::vector<int64_t> hist(ncount);
    std::vector<uint8_t> code(quant.code_size());
    for (int i = 0; i < n; ++i) {
        const float* cur_vec = data + i * d;
        quant.encode(cur_vec, code.data());
        for (int j = 0; j < d; ++j) {
            int32_t x;
            if (j & 1) {
                x = code[j / 2] >> 4 & 15;
            } else {
                x = code[j / 2] & 15;
            }
            hist[x]++;
        }
    }
    double err = imbalance_factor(ncount, hist.data());
    bool use_sq4 = err < sq4_threshold;
    LOG_KNOWHERE_INFO_ << "SQ4 imbalance factor: " << err << ", " << (use_sq4 ? "use sq4" : "try next");
    return use_sq4;
}

inline bool
evaluate_sq8(const float* data, size_t n, size_t d) {
    constexpr double sq8_threshold = 0.5;
    SQ8Quantizer<0> quant(d);
    quant.train(data, n);
    constexpr int32_t ncount = 255;
    std::vector<int64_t> hist(ncount);
    std::vector<int8_t> code(quant.code_size());
    for (int i = 0; i < n; ++i) {
        const float* cur_vec = data + i * d;
        quant.encode(cur_vec, code.data());
        for (int j = 0; j < d; ++j) {
            int32_t x = code[j] + 127;
            hist[x]++;
        }
    }
    double err = imbalance_factor(ncount, hist.data());
    bool use_sq8 = err < sq8_threshold;
    LOG_KNOWHERE_INFO_ << "SQ8 imbalance factor: " << err << ", " << (use_sq8 ? "use sq8" : "try next");
    return use_sq8;
}

}  // namespace hnswlib
