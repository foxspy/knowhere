#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <tuple>
#include <utility>

#include "index/vector_index/helpers/FaissIO.h"

namespace hnswlib {

inline float
limit_range(float x) {
    if (x < 0.0f) {
        x = 0.0f;
    }
    if (x > 1.0f) {
        x = 1.0f;
    }
    return x;
}

inline float
limit_range_sym(float x) {
    if (x < -1.0f) {
        x = -1.0f;
    }
    if (x > 1.0f) {
        x = 1.0f;
    }
    return x;
}

inline std::pair<float, float>
find_minmax(const float* data, int64_t nitems, float ratio = 0.0f) {
    size_t top = int64_t(nitems * ratio) + 1;
    std::priority_queue<float> mx_heap;
    std::priority_queue<float, std::vector<float>, std::greater<float>> mi_heap;
    for (int64_t i = 0; i < nitems; ++i) {
        if (mx_heap.size() < top) {
            mx_heap.push(data[i]);
        } else if (data[i] < mx_heap.top()) {
            mx_heap.pop();
            mx_heap.push(data[i]);
        }
        if (mi_heap.size() < top) {
            mi_heap.push(data[i]);
        } else if (data[i] > mi_heap.top()) {
            mi_heap.pop();
            mi_heap.push(data[i]);
        }
    }
    return {mx_heap.top(), mi_heap.top()};
}

inline float
find_absmax(const float* data, int64_t nitems, float ratio = 0.0f) {
    size_t top = int64_t(nitems * ratio) + 1;
    std::priority_queue<float, std::vector<float>, std::greater<float>> heap;
    for (int64_t i = 0; i < nitems; ++i) {
        float x = std::abs(data[i]);
        if (heap.size() < top) {
            heap.push(x);
        } else if (x > heap.top()) {
            heap.pop();
            heap.push(x);
        }
    }
    return heap.top();
}

inline void
find_minmax_perdim(std::vector<float>& mins, std::vector<float>& maxs, const float* data, int32_t n, int32_t d,
                   float ratio = 0.0f) {
    int64_t top = (int64_t)n * ratio + 1;
    std::vector<std::priority_queue<float>> mx_heaps(d);
    std::vector<std::priority_queue<float, std::vector<float>, std::greater<float>>> mi_heaps(d);
    for (int64_t i = 0; i < (int64_t)n * d; ++i) {
        auto& mx_heap = mx_heaps[i / n];
        auto& mi_heap = mi_heaps[i / n];
        if ((int64_t)mx_heap.size() < top) {
            mx_heap.push(data[i]);
        } else if (data[i] < mx_heap.top()) {
            mx_heap.pop();
            mx_heap.push(data[i]);
        }
        if ((int64_t)mi_heap.size() < top) {
            mi_heap.push(data[i]);
        } else if (data[i] > mi_heap.top()) {
            mi_heap.pop();
            mi_heap.push(data[i]);
        }
    }
    mins.resize(d);
    maxs.resize(d);
    for (int32_t i = 0; i < d; ++i) {
        mins[i] = mx_heaps[i].top();
        maxs[i] = mi_heaps[i].top();
    }
}

struct AffineCalibrator {
    int dim;
    float min = 0.0f;
    float dif = 0.0f;

    AffineCalibrator() = default;

    explicit AffineCalibrator(int d) : dim(d) {
    }

    void
    calibrate(const float* data, int64_t nitems, float drop_ratio = 0.0f) {
        float max;
        std::tie(this->min, max) = find_minmax(data, nitems, drop_ratio);
        this->dif = max - this->min;
        printf("AffineCalibrator calibration done, min = %f, max = %f, dif = %f\n", this->min, max, this->dif);
    }

    float
    transform(float x) const {
        x = (x - min) / dif;
        x = limit_range(x);
        return x;
    }

    float
    transform_back(float x) const {
        return x * dif + min;
    }

    bool
    check_vec(const float* q) const {
        for (int i = 0; i < dim; ++i) {
            if (q[i] > min + dif + dif * 0.1f || q[i] < min - dif * 0.1f) {
                return false;
            }
        }
        return true;
    }

    void
    load(knowhere::MemoryIOReader& input) {
        input.read(&dim, sizeof(dim));
        input.read(&min, sizeof(min));
        input.read(&dif, sizeof(dif));
    }

    void
    save(knowhere::MemoryIOWriter& output) {
        output.write(&dim, sizeof(dim));
        output.write(&min, sizeof(min));
        output.write(&dif, sizeof(dif));
    }
};

struct SymCalibrator {
    int dim;
    float max = 0.0f;

    SymCalibrator() = default;

    SymCalibrator(int d) : dim(d) {
    }

    void
    calibrate(const float* data, int64_t nitems, float drop_ratio = 0.0f) {
        max = find_absmax(data, nitems, drop_ratio);
        printf("SymCalibrator calibration done, max = %f\n", this->max);
    }

    float
    transform(float x) const {
        x = x / max;
        x = limit_range_sym(x);
        return x;
    }

    float
    transform_back(float x) const {
        return x * max;
    }

    bool
    check_vec(const float* q) const {
        for (int i = 0; i < dim; ++i) {
            if (q[i] > max * 1.1f) {
                return false;
            }
        }
        return true;
    }

    void
    load(knowhere::MemoryIOReader& input) {
        input.read(&dim, sizeof(dim));
        input.read(&max, sizeof(max));
    }

    void
    save(knowhere::MemoryIOWriter& output) {
        output.write(&dim, sizeof(dim));
        output.write(&max, sizeof(max));
    }
};

struct AffinePerDimCalibrator {
    int32_t d = 0;
    std::vector<float> mins;
    std::vector<float> difs;

    AffinePerDimCalibrator() = default;

    explicit AffinePerDimCalibrator(int32_t dim) : d(dim), mins(d), difs(d) {
    }

    void
    calibrate(const float* data, int32_t n, int32_t d, float drop_ratio = 0.0f) {
        std::vector<float> maxs;
        find_minmax_perdim(mins, maxs, data, n, d, drop_ratio);
        for (int32_t i = 0; i < d; ++i) {
            difs[i] = maxs[i] - mins[i];
        }
        printf("AffinePerDimCalibrator calibration done\n");
    }

    float
    transform(float x, int32_t dim) const {
        x = (x - mins[dim]) / difs[dim];
        x = limit_range(x);
        return x;
    }

    float
    transform_back(float x, int32_t dim) const {
        return x * difs[dim] + mins[dim];
    }
};

}  // namespace hnswlib
