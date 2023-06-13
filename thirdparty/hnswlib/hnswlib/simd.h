#pragma once

#include "common/Log.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "hnswlib/hnswlib/bf16.h"

namespace hnswlib {

inline int simd_init = [] {
#if defined(__AVX512VNNI__)
    LOG_KNOWHERE_INFO_ << "VNNI instruction enabled.";
#elif defined(__AVX512F__)
    LOG_KNOWHERE_INFO_ << "VNNI instruction disabled, AVX512 instruction enabled.";
#else
    LOG_KNOWHERE_INFO_ << "AVX512 instruction disabled. ";
#endif
    return 0;
}();

__attribute__((__always_inline__)) inline void
prefetch_L1(const void* address) {
#if defined(__AVX2__)
    _mm_prefetch((const char*)address, _MM_HINT_T0);
#else
    __builtin_prefetch(address, 0, 3);
#endif
}

__attribute__((__always_inline__)) inline void
prefetch_L2(const void* address) {
#if defined(__AVX2__)
    _mm_prefetch((const char*)address, _MM_HINT_T1);
#else
    __builtin_prefetch(address, 0, 2);
#endif
}

__attribute__((__always_inline__)) inline void
prefetch_L3(const void* address) {
#if defined(__AVX2__)
    _mm_prefetch((const char*)address, _MM_HINT_T2);
#else
    __builtin_prefetch(address, 0, 1);
#endif
}

inline void
mem_prefetch(char* ptr, const int num_lines) {
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 28:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 27:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 26:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 25:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 24:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 23:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 22:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 21:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 20:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
    }
}

#if defined(__AVX512F__)

inline float
reduce_add_f32x16(__m512 x) {
    auto sumh = _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
    auto sumhh = _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
    auto tmp1 = _mm_hadd_ps(sumhh, sumhh);
    return tmp1[0] + tmp1[1];
}

inline int32_t
reduce_add_i32x16(__m512i x) {
    auto sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(x, 0), _mm512_extracti32x8_epi32(x, 1));
    auto sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh), _mm256_extracti128_si256(sumh, 1));
    auto tmp1 = _mm_hadd_epi32(sumhh, sumhh);
    return _mm_extract_epi32(tmp1, 0) + _mm_extract_epi32(tmp1, 1);
}

#if defined(__AVX512FP16__)

inline float
reduce_add_f16x32(__m512h x) {
    return _mm512_reduce_add_ph(x);
}

#endif

inline __m512i
cvti4x64_i8x64(__m256i x) {
    auto mask = _mm256_set1_epi8(0x0f);
    auto lo = _mm256_and_si256(x, mask);
    auto hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask);
    auto loo = _mm512_cvtepu8_epi16(lo);
    auto hii = _mm512_cvtepu8_epi16(hi);
    hii = _mm512_slli_epi16(hii, 8);
    auto ret = _mm512_or_si512(loo, hii);
    ret = _mm512_slli_epi64(ret, 3);
    return ret;
}

#endif

#if defined(__AVX2__)

inline float
reduce_add_f32x8(__m256 x) {
    auto sumh = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
    auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
    return _mm_cvtss_f32(tmp2);
}

inline int32_t
reduce_add_i16x16(__m256i x) {
    auto sumh = _mm_add_epi16(_mm256_extracti128_si256(x, 0), _mm256_extracti128_si256(x, 1));
    auto tmp = _mm256_cvtepi16_epi32(sumh);
    auto sumhh = _mm_add_epi32(_mm256_extracti128_si256(tmp, 0), _mm256_extracti128_si256(tmp, 1));
    auto tmp2 = _mm_hadd_epi32(sumhh, sumhh);
    return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

#endif

inline float
L2SqrBF16(const float* x, const bf16* y, int32_t d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtepu16_epi32(zz);
        yy = _mm512_slli_epi32(yy, 16);
        auto t = _mm512_sub_ps(xx, (__m512)yy);
        sum = _mm512_fmadd_ps(t, t, sum);
    }
    return reduce_add_f32x16(sum);
#elif defined(__AVX2__)
    __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        {
            auto xx = _mm256_loadu_ps(x + i);
            auto zz = _mm_loadu_si128((__m128i*)(y + i));
            auto yy = _mm256_cvtepu16_epi32(zz);
            yy = _mm256_slli_epi32(yy, 16);
            auto t = _mm256_sub_ps(xx, (__m256)yy);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(t, t));
        }
        {
            auto xx = _mm256_loadu_ps(x + i + 8);
            auto zz = _mm_loadu_si128((__m128i*)(y + i + 8));
            auto yy = _mm256_cvtepu16_epi32(zz);
            yy = _mm256_slli_epi32(yy, 16);
            auto t = _mm256_sub_ps(xx, (__m256)yy);
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(t, t));
        }
    }
    sum1 = _mm256_add_ps(sum1, sum2);
    return reduce_add_f32x8(sum1);
#else
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        float d = x[i] - float(y[i]);
        sum += d * d;
    }
    return sum;
#endif
}

inline float
IPBF16(const float* x, const bf16* y, int32_t d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtepu16_epi32(zz);
        yy = _mm512_slli_epi32(yy, 16);
        sum = _mm512_fmadd_ps(xx, (__m512)yy, sum);
    }
    return -reduce_add_f32x16(sum);
#elif defined(__AVX2__)
    __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        {
            auto xx = _mm256_loadu_ps(x + i);
            auto zz = _mm_loadu_si128((__m128i*)(y + i));
            auto yy = _mm256_cvtepu16_epi32(zz);
            yy = _mm256_slli_epi32(yy, 16);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, (__m256)yy));
        }
        {
            auto xx = _mm256_loadu_ps(x + i + 8);
            auto zz = _mm_loadu_si128((__m128i*)(y + i + 8));
            auto yy = _mm256_cvtepu16_epi32(zz);
            yy = _mm256_slli_epi32(yy, 16);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, (__m256)yy));
        }
    }
    sum1 = _mm256_add_ps(sum1, sum2);
    return -reduce_add_f32x8(sum1);
#else
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += x[i] * float(y[i]);
    }
    return -sum;
#endif
}

inline float
L2SqrBF16Sym(const bf16* x, const bf16* y, int32_t d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xxx = _mm256_loadu_si256((__m256i*)(x + i));
        auto xx = _mm512_cvtepu16_epi32(xxx);
        xx = _mm512_slli_epi32(xx, 16);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtepu16_epi32(zz);
        yy = _mm512_slli_epi32(yy, 16);
        auto t = _mm512_sub_ps((__m512)xx, (__m512)yy);
        sum = _mm512_fmadd_ps(t, t, sum);
    }
    return reduce_add_f32x16(sum);
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        auto xxx = _mm_loadu_si128((__m128i*)(x + i));
        auto xx = _mm256_cvtepu16_epi32(xxx);
        xx = _mm256_slli_epi32(xx, 16);
        auto zz = _mm_loadu_si128((__m128i*)(y + i));
        auto yy = _mm256_cvtepu16_epi32(zz);
        yy = _mm256_slli_epi32(yy, 16);
        auto t = _mm256_sub_ps((__m256)xx, (__m256)yy);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
    }
    return reduce_add_f32x8(sum);
#else
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        float d = float(x[i]) - float(y[i]);
        sum += d * d;
    }
    return sum;
#endif
}

inline float
IPBF16Sym(const bf16* x, const bf16* y, int32_t d) {
#if defined(__AVX512BF16__)
    auto sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 32) {
        auto xx = (__m512bh)_mm512_loadu_si512(x + i);
        auto yy = (__m512bh)_mm512_loadu_si512(y + i);
        sum = _mm512_dpbf16_ps(sum, xx, yy);
    }
    return -reduce_add_f32x16(sum);
#elif defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xxx = _mm256_loadu_si256((__m256i*)(x + i));
        auto xx = _mm512_cvtepu16_epi32(xxx);
        xx = _mm512_slli_epi32(xx, 16);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtepu16_epi32(zz);
        yy = _mm512_slli_epi32(yy, 16);
        sum = _mm512_fmadd_ps((__m512)xx, (__m512)yy, sum);
    }
    return -reduce_add_f32x16(sum);
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        auto xxx = _mm_loadu_si128((__m128i*)(x + i));
        auto xx = _mm256_cvtepu16_epi32(xxx);
        xx = _mm256_slli_epi32(xx, 16);
        auto zz = _mm_loadu_si128((__m128i*)(y + i));
        auto yy = _mm256_cvtepu16_epi32(zz);
        yy = _mm256_slli_epi32(yy, 16);
        sum = _mm256_add_ps(sum, _mm256_mul_ps((__m256)xx, (__m256)yy));
    }
    return -reduce_add_f32x8(sum);
#else
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += float(x[i]) * float(y[i]);
    }
    return -sum;

#endif
}

inline int32_t L2SqrSQ8(const int8_t* x, const int8_t* y, int32_t d) {
  int32_t ans = 0;
  for (int32_t i = 0; i < d; ++i) {
    auto d = int32_t(x[i]) - int32_t(y[i]);
    ans += d * d;
  }
  return ans;
}

inline int32_t IPSQ8(const int8_t* x, const int8_t* y, int32_t d) {
  int32_t ans = 0;
  for (int32_t i = 0; i < d; ++i) {
    ans += int32_t(x[i]) * int32_t(y[i]);
  }
  return -ans;
}

inline int32_t
L2SqrSQ4Sym(const uint8_t* x, const uint8_t* y, int32_t d) {
#if defined(__AVX512VNNI__)
    __m512i sum1 = _mm512_setzero_epi32(), sum2 = _mm512_setzero_epi32();
    __m512i mask = _mm512_set1_epi8(0xf);
    for (int i = 0; i < d; i += 128) {
        auto xx = _mm512_loadu_si512((__m512i*)(x + i / 2));
        auto yy = _mm512_loadu_si512((__m512i*)(y + i / 2));
        auto xx1 = _mm512_and_si512(xx, mask);
        auto xx2 = _mm512_and_si512(_mm512_srli_epi16(xx, 4), mask);
        auto yy1 = _mm512_and_si512(yy, mask);
        auto yy2 = _mm512_and_si512(_mm512_srli_epi16(yy, 4), mask);
        auto d1 = _mm512_sub_epi8(xx1, yy1);
        auto d2 = _mm512_sub_epi8(xx2, yy2);
        d1 = _mm512_abs_epi8(d1);
        d2 = _mm512_abs_epi8(d2);
        // GCC bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94663
        // sum1 = _mm512_dpbusd_epi32(sum1, d1, d1);
        // sum2 = _mm512_dpbusd_epi32(sum2, d2, d2);
        asm("vpdpbusd %1, %2, %0" : "+x"(sum1) : "mx"(d1), "x"(d1));
        asm("vpdpbusd %1, %2, %0" : "+x"(sum2) : "mx"(d2), "x"(d2));
    }
    sum1 = _mm512_add_epi32(sum1, sum2);
    return reduce_add_i32x16(sum1);
#elif defined(__AVX2__)
    __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
    __m256i mask = _mm256_set1_epi8(0xf);
    for (int i = 0; i < d; i += 64) {
        auto xx = _mm256_loadu_si256((__m256i*)(x + i / 2));
        auto yy = _mm256_loadu_si256((__m256i*)(y + i / 2));
        auto xx1 = _mm256_and_si256(xx, mask);
        auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);
        auto yy1 = _mm256_and_si256(yy, mask);
        auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);
        auto d1 = _mm256_sub_epi8(xx1, yy1);
        auto d2 = _mm256_sub_epi8(xx2, yy2);
        d1 = _mm256_abs_epi8(d1);
        d2 = _mm256_abs_epi8(d2);
        sum1 = _mm256_add_epi16(sum1, _mm256_maddubs_epi16(d1, d1));
        sum2 = _mm256_add_epi16(sum2, _mm256_maddubs_epi16(d2, d2));
    }
    sum1 = _mm256_add_epi32(sum1, sum2);
    return reduce_add_i16x16(sum1);
#else
    int32_t ans = 0;
    for (int32_t i = 0; i < d; ++i) {
        int32_t xx = x[i / 2] >> ((i & 1) * 4) & 15;
        int32_t yy = y[i / 2] >> ((i & 1) * 4) & 15;
        auto d = xx - yy;
        ans += d * d;
    }
    return ans;
#endif
}

inline int32_t
IPSQ4Sym(const uint8_t* x, const uint8_t* y, int32_t d) {
#if defined(__AVX2__)
    __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
    __m256i mask = _mm256_set1_epi8(0xf);
    for (int i = 0; i < d; i += 64) {
        auto xx = _mm256_loadu_si256((__m256i*)(x + i / 2));
        auto yy = _mm256_loadu_si256((__m256i*)(y + i / 2));
        auto xx1 = _mm256_and_si256(xx, mask);
        auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);
        auto yy1 = _mm256_and_si256(yy, mask);
        auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);
        sum1 = _mm256_add_epi16(sum1, _mm256_maddubs_epi16(xx1, yy1));
        sum2 = _mm256_add_epi16(sum2, _mm256_maddubs_epi16(xx2, yy2));
    }
    sum1 = _mm256_add_epi32(sum1, sum2);
    return -reduce_add_i16x16(sum1);
#else
    int32_t ans = 0;
    for (int32_t i = 0; i < d; ++i) {
        int32_t xx = x[i] >> ((i & 1) * 4) & 15;
        int32_t yy = y[i] >> ((i & 1) * 4) & 15;
        ans += xx * yy;
    }
    return -ans;
#endif
}

}  // namespace hnswlib
