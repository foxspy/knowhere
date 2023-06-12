#pragma once

#include <functional>

#include "hnswlib/hnswlib/tensor.h"

namespace hnswlib {

struct Computer {
    const Tensor& tensor;

    explicit Computer(const Tensor& tensor) : tensor(tensor) {
    }

    void
    prefetch(int32_t u, int32_t lines) const {
        tensor.prefetch(u, lines);
    }
};

struct MemCpyTag {};

template <auto dist_func, typename U, typename T, typename T1, typename T2, typename... Args>
struct ComputerImpl : Computer {
    using dist_type = U;
    using S = T;
    using X = T1;
    using Y = T2;
    static_assert(
        std::is_convertible_v<decltype(dist_func), std::function<dist_type(const X*, const Y*, int32_t, Args...)>>);
    X* q = nullptr;
    std::tuple<Args...> args;

    template <typename E>
    ComputerImpl(const Tensor& tensor, const S* query, const E& encoder, Args&&... args)
        : Computer(tensor),
          q((X*)align_alloc(this->tensor.dim_align() * sizeof(X))),
          args(std::forward<Args>(args)...) {
        if constexpr (std::is_same_v<std::decay_t<decltype(encoder)>, MemCpyTag>) {
            static_assert(std::is_same_v<S, X>);
            memcpy(q, query, this->tensor.dim() * sizeof(X));
        } else {
            encoder((const S*)query, q);
        }
    }

    ~ComputerImpl() {
        free(q);
    }

    __attribute__((always_inline)) dist_type
    operator()(int32_t u) const {
        return std::apply(
            [&](auto&&... args) {
                return dist_func(q, (const Y*)this->tensor.get(u), this->tensor.dim_align(), args...);
            },
            args);
    }
};

template <auto dist_func, typename U, typename T, typename... Args>
struct SymComputerImpl : Computer {
    using dist_type = U;
    using X = T;
    static_assert(
        std::is_convertible_v<decltype(dist_func), std::function<dist_type(const X*, const X*, int32_t, Args...)>>);

    std::tuple<Args...> args;

    explicit SymComputerImpl(const Tensor& tensor, Args&&... args)
        : Computer(tensor), args(std::forward<Args>(args)...) {
    }

    __attribute__((always_inline)) dist_type
    operator()(int32_t u, int32_t v) const {
        return std::apply(
            [&](auto&&... args) {
                return dist_func((const X*)this->tensor.get(u), (const X*)this->tensor.get(v), this->tensor.dim_align(),
                                 args...);
            },
            args);
    }
};

}  // namespace hnswlib
