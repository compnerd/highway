// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Include guard (still compiled once per target)
#include <cmath>

#if defined(HIGHWAY_HWY_CONTRIB_DOT_DOT_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_DOT_DOT_INL_H_
#undef HIGHWAY_HWY_CONTRIB_DOT_DOT_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_DOT_DOT_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

struct Dot {
  // Specify zero or more of these, ORed together, as the kAssumptions template
  // argument to Compute. Each one may improve performance or reduce code size,
  // at the cost of additional requirements on the arguments.
  enum Assumptions {
    // num_elements is at least N, which may be up to HWY_MAX_LANES(T).
    kAtLeastOneVector = 1,
    // num_elements is divisible by N.
    kMultipleOfVector = 2,
    // RoundUpTo(num_elements, N) elements are accessible; their value does not
    // matter (will be treated as if they were zero).
    kPaddedToVector = 4,
    // Pointers pa and pb, respectively, are multiples of N * sizeof(T).
    // For example, aligned_allocator.h ensures this. Note that it is still
    // beneficial to arrange such alignment even if these flags are not set.
    // If not set, the pointers need only be aligned to sizeof(T).
    kVectorAlignedA = 8,
    kVectorAlignedB = 16,
  };

  // Returns sum{pa[i] * pb[i]} for float or double inputs.
  template <int kAssumptions, class D, typename T = TFromD<D>,
            HWY_IF_NOT_LANE_SIZE_D(D, 2)>
  static HWY_INLINE T Compute(const D d, const T* const HWY_RESTRICT pa,
                              const T* const HWY_RESTRICT pb,
                              const size_t num_elements) {
    static_assert(IsFloat<T>(), "MulAdd requires float type");
    using V = decltype(Zero(d));

    const size_t N = Lanes(d);
    size_t i = 0;

    constexpr bool kIsAtLeastOneVector = kAssumptions & kAtLeastOneVector;
    constexpr bool kIsMultipleOfVector = kAssumptions & kMultipleOfVector;
    constexpr bool kIsPaddedToVector = kAssumptions & kPaddedToVector;
    constexpr bool kIsAlignedA = kAssumptions & kVectorAlignedA;
    constexpr bool kIsAlignedB = kAssumptions & kVectorAlignedB;

    // Won't be able to do a full vector load without padding => scalar loop.
    if (!kIsAtLeastOneVector && !kIsMultipleOfVector && !kIsPaddedToVector &&
        HWY_UNLIKELY(num_elements < N)) {
      T sum0 = T(0);  // Only 2x unroll to avoid excessive code size for..
      T sum1 = T(0);  // this unlikely(?) case.
      for (; i + 2 <= num_elements; i += 2) {
        sum0 += pa[i + 0] * pb[i + 0];
        sum1 += pa[i + 1] * pb[i + 1];
      }
      if (i < num_elements) {
        sum1 += pa[i] * pb[i];
      }
      return sum0 + sum1;
    }

    // Compiler doesn't make independent sum* accumulators, so unroll manually.
    // Some older compilers might not be able to fit the 8 vectors in registers,
    // so manual unrolling can be helpful if you run into this issue.
    // 2 FMA ports * 4 cycle latency = 8x unrolled. For each misaligned pointer,
    // throughput is halved because load ports become the limiting factor.
    constexpr size_t kUnroll = 1ull << (kIsAlignedA + kIsAlignedB + 1);
    V sum[kUnroll];
    for (int i = 0; i < kUnroll; ++i) {
      sum[i] = Zero(d);
    }

    // Main loop: unrolled
    for (; i + kUnroll * N <= num_elements; i += kUnroll * N) {
      for (size_t j = 0; j < kUnroll; ++j) {
        const T* HWY_RESTRICT pos_a = pa + i + j * N;
        const T* HWY_RESTRICT pos_b = pb + i + j * N;
        const auto a = kIsAlignedA ? Load(d, pos_a) : LoadU(d, pos_a);
        const auto b = kIsAlignedB ? Load(d, pos_b) : LoadU(d, pos_b);
        sum[j] = MulAdd(a, b, sum[j]);
      }
    }

    // Up to kUnroll iterations of whole vectors
    for (; i + N <= num_elements; i += N) {
      const auto a = kIsAlignedA ? Load(d, pa + i) : LoadU(d, pa + i);
      const auto b = kIsAlignedB ? Load(d, pb + i) : LoadU(d, pb + i);
      sum[kUnroll - 1] = MulAdd(a, b, sum[kUnroll - 1]);
    }

    if (!kIsMultipleOfVector) {
      const size_t remaining = num_elements - i;
      if (remaining != 0) {
        if (kIsPaddedToVector) {
          const auto mask = FirstN(d, remaining);
          const auto a = kIsAlignedA ? Load(d, pa + i) : LoadU(d, pa + i);
          const auto b = kIsAlignedB ? Load(d, pb + i) : LoadU(d, pb + i);
          sum[kUnroll - 1] = MulAdd(IfThenElseZero(mask, a),
                                    IfThenElseZero(mask, b), sum[kUnroll - 1]);
        } else {
          // Unaligned load such that the last element is in the highest lane -
          // ensures we do not touch any elements outside the valid range.
          // If we get here, then num_elements >= N.
          HWY_DASSERT(i >= N);
          i += remaining - N;
          const auto discard = FirstN(d, N - remaining);
          const auto a = LoadU(d, pa + i);  // always unaligned
          const auto b = LoadU(d, pb + i);
          sum[kUnroll - 1] =
              MulAdd(IfThenZeroElse(discard, a), IfThenZeroElse(discard, b),
                     sum[kUnroll - 1]);
        }
      }
    }  // kMultipleOfVector

    // Reduction tree: sum of all accumulators by pairs into sum[0], then the
    // lanes.
    for (size_t power = 1; power < kUnroll; power *= 2) {
      for (size_t i = 0; i < kUnroll; i += 2 * power) {
        sum[i] += sum[i + power];
      }
    }

    return GetLane(SumOfLanes(d, sum[0]));
  }

  // Returns sum{pa[i] * pb[i]} for bfloat16 inputs.
  template <int kAssumptions, class D>
  static HWY_INLINE float Compute(const D d,
                                  const bfloat16_t* const HWY_RESTRICT pa,
                                  const bfloat16_t* const HWY_RESTRICT pb,
                                  const size_t num_elements) {
    const RebindToUnsigned<D> du16;
    const Repartition<float, D> df32;

    using V = decltype(Zero(df32));
    const size_t N = Lanes(d);
    size_t i = 0;

    constexpr bool kIsAtLeastOneVector = kAssumptions & kAtLeastOneVector;
    constexpr bool kIsMultipleOfVector = kAssumptions & kMultipleOfVector;
    constexpr bool kIsPaddedToVector = kAssumptions & kPaddedToVector;
    constexpr bool kIsAlignedA = kAssumptions & kVectorAlignedA;
    constexpr bool kIsAlignedB = kAssumptions & kVectorAlignedB;

    // Won't be able to do a full vector load without padding => scalar loop.
    if (!kIsAtLeastOneVector && !kIsMultipleOfVector && !kIsPaddedToVector &&
        HWY_UNLIKELY(num_elements < N)) {
      float_t sum0 = 0.0f;  // Only 2x unroll to avoid excessive code size for..
      float_t sum1 = 0.0f;  // this unlikely(?) case.
      for (; i + 2 <= num_elements; i += 2) {
        sum0 += F32FromBF16(pa[i + 0]) * F32FromBF16(pb[i + 0]);
        sum1 += F32FromBF16(pa[i + 1]) * F32FromBF16(pb[i + 1]);
      }
      if (i < num_elements) {
        sum1 += F32FromBF16(pa[i]) * F32FromBF16(pb[i]);
      }
      return sum0 + sum1;
    }

    // Compiler doesn't make independent sum* accumulators, so unroll manually.
    // Some older compilers might not be able to fit the 8 vectors in registers,
    // so manual unrolling can be helpful if you run into this issue.
    constexpr size_t kUnroll = 4;  // number of input loads
    V sum[2 * kUnroll];            // twice as many due to bf16 -> f32
    for (int i = 0; i < 2 * kUnroll; ++i) {
      sum[i] = Zero(df32);
    }

    // Main loop: unrolled
    for (; i + kUnroll * N <= num_elements; i += kUnroll * N) {
      for (size_t j = 0; j < kUnroll; ++j) {
        const bfloat16_t* HWY_RESTRICT pos_a = pa + i + j * N;
        const bfloat16_t* HWY_RESTRICT pos_b = pb + i + j * N;
        const auto a16 = kIsAlignedA ? Load(d, pos_a) : LoadU(d, pos_a);
        const auto b16 = kIsAlignedB ? Load(d, pos_b) : LoadU(d, pos_b);
        const auto aL = PromoteLowerTo(df32, a16);
        const auto aH = PromoteUpperTo(df32, a16);
        const auto bL = PromoteLowerTo(df32, b16);
        const auto bH = PromoteUpperTo(df32, b16);
        sum[2 * j + 0] = MulAdd(aL, bL, sum[2 * j + 0]);
        sum[2 * j + 1] = MulAdd(aH, bH, sum[2 * j + 1]);
      }
    }

    // Up to kUnroll iterations of whole vectors
    for (; i + N <= num_elements; i += N) {
      const auto a16 = kIsAlignedA ? Load(d, pa + i) : LoadU(d, pa + i);
      const auto b16 = kIsAlignedB ? Load(d, pb + i) : LoadU(d, pb + i);
      const auto aL = PromoteLowerTo(df32, a16);
      const auto aH = PromoteUpperTo(df32, a16);
      const auto bL = PromoteLowerTo(df32, b16);
      const auto bH = PromoteUpperTo(df32, b16);
      sum[0] = MulAdd(aL, bL, sum[0]);
      sum[1] = MulAdd(aH, bH, sum[1]);
    }

    if (!kIsMultipleOfVector) {
      const size_t remaining = num_elements - i;
      if (remaining != 0) {
        if (kIsPaddedToVector) {
          const auto mask = FirstN(du16, remaining);
          const auto va = kIsAlignedA ? Load(d, pa + i) : LoadU(d, pa + i);
          const auto vb = kIsAlignedB ? Load(d, pb + i) : LoadU(d, pb + i);
          const auto a16 = BitCast(d, IfThenElseZero(mask, BitCast(du16, va)));
          const auto b16 = BitCast(d, IfThenElseZero(mask, BitCast(du16, vb)));
          const auto aL = PromoteLowerTo(df32, a16);
          const auto aH = PromoteUpperTo(df32, a16);
          const auto bL = PromoteLowerTo(df32, b16);
          const auto bH = PromoteUpperTo(df32, b16);
          sum[0] = MulAdd(aL, bL, sum[0]);
          sum[1] = MulAdd(aH, bH, sum[1]);
        } else {
          // Unaligned load such that the last element is in the highest lane -
          // ensures we do not touch any elements outside the valid range.
          // If we get here, then num_elements >= N.
          HWY_DASSERT(i >= N);
          i += remaining - N;
          const auto discard = FirstN(du16, N - remaining);
          const auto va = LoadU(d, pa + i);  // always unaligned
          const auto vb = LoadU(d, pb + i);
          const auto a16 =
              BitCast(d, IfThenZeroElse(discard, BitCast(du16, va)));
          const auto b16 =
              BitCast(d, IfThenZeroElse(discard, BitCast(du16, vb)));
          const auto aL = PromoteLowerTo(df32, a16);
          const auto aH = PromoteUpperTo(df32, a16);
          const auto bL = PromoteLowerTo(df32, b16);
          const auto bH = PromoteUpperTo(df32, b16);
          sum[0] = MulAdd(aL, bL, sum[0]);
          sum[1] = MulAdd(aH, bH, sum[1]);
        }
      }
    }  // kMultipleOfVector

    // Reduction tree: sum of all accumulators by pairs into sum[0], then the
    // lanes.
    for (size_t power = 1; power < 2 * kUnroll; power *= 2) {
      for (size_t i = 0; i < 2 * kUnroll; i += 2 * power) {
        sum[i] += sum[i + power];
      }
    }

    return GetLane(SumOfLanes(df32, sum[0]));
  }
};

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_DOT_DOT_INL_H_
