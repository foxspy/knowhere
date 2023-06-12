#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace hnswlib {

#define ROUND_MODE_TO_NEAREST

struct bf16 {
    uint16_t x = 0;

    bf16() = default;

    explicit bf16(float f)
        : x(
#if defined(ROUND_MODE_TO_NEAREST)
              round_to_nearest(f)
#elif defined(ROUND_MODE_TO_NEAREST_EVEN)
              round_to_nearest_even(f)
#elif defined(ROUND_MODE_TRUNCATE)
              truncate(f)
#else
#error "ROUNDING_MODE must be one of ROUND_MODE_TO_NEAREST, ROUND_MODE_TO_NEAREST_EVEN, or ROUND_MODE_TRUNCATE"
#endif
          ) {
    }

    template <typename F, std::enable_if_t<std::is_convertible_v<F, float>>>
    explicit bf16(F f) : bf16(float(f)) {
    }

    explicit operator float() const {
        uint32_t buf = 0;
        std::memcpy(reinterpret_cast<char*>(&buf) + 2, &x, 2);
        auto ptr = reinterpret_cast<void*>(&buf);
        return *reinterpret_cast<float*>(ptr);
    }

    static uint32_t
    getbits(float x) {
        auto ptr = reinterpret_cast<void*>(&x);
        return *(reinterpret_cast<uint32_t*>(ptr));
    }

    static uint16_t
    round_to_nearest_even(float x) {
        return static_cast<uint16_t>((getbits(x) + ((getbits(x) & 0x00010000) >> 1)) >> 16);
    }

    static uint16_t
    round_to_nearest(float x) {
        return static_cast<uint16_t>((getbits(x) + 0x8000) >> 16);
    }

    static uint16_t
    truncate(float x) {
        return static_cast<uint16_t>((getbits(x)) >> 16);
    }
};

}  // namespace hnswlib
