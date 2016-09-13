/* NOLINT(build/header_guard) */
/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* template parameters: FN */

#define HistogramType FN(Histogram)

// Credit: http://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
// FIXME: copied
static BROTLI_INLINE int FN(sum8i)(__m256i x) {
  // hiQuad = ( x7, x6, x5, x4 )
  const __m128i hiQuad =
      _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(x), 1));
  // loQuad = ( x3, x2, x1, x0 )
  const __m128i loQuad =
      _mm_castps_si128(_mm256_castps256_ps128(_mm256_castsi256_ps(x)));
  // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
  const __m128i sumQuad = _mm_add_epi32(loQuad, hiQuad);
  // loDual = ( -, -, x1 + x5, x0 + x4 )
  const __m128i loDual = sumQuad;
  // hiDual = ( -, -, x3 + x7, x2 + x6 )
  const __m128i hiDual =
      _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(sumQuad),
                                     _mm_castsi128_ps(sumQuad)));
  // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
  const __m128i sumDual = _mm_add_epi32(loDual, hiDual);
  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
  const __m128i lo = sumDual;
  // hi = ( -, -, -, x1 + x3 + x5 + x7 )
  const __m128i hi =
      _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(sumDual),
                                      _mm_castsi128_ps(sumDual), 0x1));
  // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
  const __m128i sum = _mm_add_epi32(lo, hi);
  return _mm_cvtsi128_si32(sum);
}

static BROTLI_INLINE int FN(sum8)(__m256 x) {
  // hiQuad = ( x7, x6, x5, x4 )
  const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
  // loQuad = ( x3, x2, x1, x0 )
  const __m128 loQuad = _mm256_castps256_ps128(x);
  // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  // loDual = ( -, -, x1 + x5, x0 + x4 )
  const __m128 loDual = sumQuad;
  // hiDual = ( -, -, x3 + x7, x2 + x6 )
  const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
  const __m128 lo = sumDual;
  // hi = ( -, -, -, x1 + x3 + x5 + x7 )
  const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
  const __m128 sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}

static BROTLI_INLINE float FN(CostComputation)(uint32_t *depth_histo, const uint32_t *nnz_data,
                                               const size_t nnz, const float total_count,
                                               const float log2total) {
  float bits = 0;
  size_t i, j, max_depth = 1;
#ifdef __AVX2__
  // 2^(-depth - 0.5)
  __m256 pow2l = _mm256_setr_ps(
      1.0f/*0.7071067811865476f*/, 0.3535533905932738f, 0.1767766952966369f, 0.0883883476483184f,
      0.0441941738241592f, 0.0220970869120796f, 0.0110485434560398f, 0.0055242717280199f);
  __m256 pow2h = _mm256_setr_ps(
      0.0027621358640100f, 0.0013810679320050f, 0.0006905339660025f, 0.0003452669830012f,
      0.0001726334915006f, 0.0000863167457503f, 0.0000431583728752f, /*0.0000215791864376f*/0);
  __m256 ymm_tc = _mm256_set1_ps(total_count);
  __m256i search_depthl = _mm256_cvtps_epi32(_mm256_mul_ps(pow2l, ymm_tc));
  __m256i search_depthh = _mm256_cvtps_epi32(_mm256_mul_ps(pow2h, ymm_tc));
  __m256i suml = _mm256_set1_epi32(0);
  __m256i sumh = _mm256_set1_epi32(0);
  for (i = 0; i < nnz - 4; i += 4) {
    size_t j;
    for (j = 0; j < 4; j++) {
      __m256i count = _mm256_set1_epi32(nnz_data[i + j]);
      __m256i cmpl = _mm256_cmpgt_epi32(count, search_depthl);
      __m256i cmph = _mm256_cmpgt_epi32(count, search_depthh);
      __m256i mask = _mm256_set1_epi32(1);
      cmpl = _mm256_and_si256(cmpl, mask);
      cmph = _mm256_and_si256(cmph, mask);
      suml = _mm256_add_epi32(suml, cmpl);
      sumh = _mm256_add_epi32(sumh, cmph);
    }
  }
  for (; i < nnz; i++) {
    __m256i count = _mm256_set1_epi32(nnz_data[i]);
    __m256i cmpl = _mm256_cmpgt_epi32(count, search_depthl);
    __m256i cmph = _mm256_cmpgt_epi32(count, search_depthh);
    __m256i mask = _mm256_set1_epi32(1);
    cmpl = _mm256_and_si256(cmpl, mask);
    cmph = _mm256_and_si256(cmph, mask);
    suml = _mm256_add_epi32(suml, cmpl);
    sumh = _mm256_add_epi32(sumh, cmph);
  }

  // Deal with depth_histo and max_depth
  {
    uint32_t cum_sum[BROTLI_CODE_LENGTH_CODES] = { 0 };
    _mm256_storeu_si256((__m256i *) (cum_sum + 0), suml);
    _mm256_storeu_si256((__m256i *) (cum_sum + 8), sumh);
    size_t j = 0;
    for (j = 1; j < 16; j++) {
      depth_histo[j] += cum_sum[j] - cum_sum[j - 1];
      if (cum_sum[j] - cum_sum[j - 1] != 0)
        max_depth = j;
    }
  }

  i = 0;
  if (nnz > 8) {
    __m256 ymm_log2total = _mm256_set1_ps(log2total);
    __m256 bit_cum = _mm256_set1_ps(0);
    for (i = 0; i < nnz - 8; i += 8) {
      __m256 counts = _mm256_cvtepi32_ps(
          _mm256_loadu_si256((const __m256i *) (nnz_data + i)));
      __m256 log_counts = FastApproxYMMLog2(counts);
      __m256 log2p = _mm256_sub_ps(ymm_log2total, log_counts);
      bit_cum = _mm256_add_ps(bit_cum, _mm256_mul_ps(counts, log2p));
    }
    bits += FN(sum8)(bit_cum);
  }
  for (; i < nnz; i++) {
    float log2p = log2total - FastLog2(nnz_data[i]);
    bits += nnz_data[i] * log2p;
  }
#else
  for (i = 0; i < nnz; i++) {
    /* Compute -log2(P(symbol)) = -log2(count(symbol)/total_count) =
                                = log2(total_count) - log2(count(symbol)) */
    float log2p = log2total - FastLog2(nnz_data[i]);
    /* Approximate the bit depth by round(-log2(P(symbol))) */
    size_t depth = (size_t)(log2p + 0.5);
    bits += nnz_data[i] * log2p;
    if (depth > 15) {
      depth = 15;
    }
    if (depth > max_depth) {
      max_depth = depth;
    }
    ++depth_histo[depth];
  }
#endif
  /* Add the estimated encoding cost of the code length code histogram. */
  bits += (float)(18 + 2 * max_depth);
  /* Add the entropy of the code length code histogram. */
  bits += BitsEntropy(depth_histo, BROTLI_CODE_LENGTH_CODES);
  return bits;
}

float FN(BrotliPopulationCost)(const HistogramType* histogram) {
  static const float kOneSymbolHistogramCost = 12;
  static const float kTwoSymbolHistogramCost = 20;
  static const float kThreeSymbolHistogramCost = 28;
  static const float kFourSymbolHistogramCost = 37;
  const size_t data_size = FN(HistogramDataSize)();
  int count = 0;
  size_t s[5];
  float bits = 0.0;
  size_t i;
  if (histogram->total_count_ == 0) {
    return kOneSymbolHistogramCost;
  }
  for (i = 0; i < data_size; ++i) {
    if (histogram->data_[i] > 0) {
      s[count] = i;
      ++count;
      if (count > 4) break;
    }
  }
  if (count == 1) {
    return kOneSymbolHistogramCost;
  }
  if (count == 2) {
    return (kTwoSymbolHistogramCost + (float)histogram->total_count_);
  }
  if (count == 3) {
    const uint32_t histo0 = histogram->data_[s[0]];
    const uint32_t histo1 = histogram->data_[s[1]];
    const uint32_t histo2 = histogram->data_[s[2]];
    const uint32_t histomax =
        BROTLI_MAX(uint32_t, histo0, BROTLI_MAX(uint32_t, histo1, histo2));
    return (kThreeSymbolHistogramCost +
            2 * (histo0 + histo1 + histo2) - histomax);
  }
  if (count == 4) {
    uint32_t histo[4];
    uint32_t h23;
    uint32_t histomax;
    for (i = 0; i < 4; ++i) {
      histo[i] = histogram->data_[s[i]];
    }
    /* Sort */
    for (i = 0; i < 4; ++i) {
      size_t j;
      for (j = i + 1; j < 4; ++j) {
        if (histo[j] > histo[i]) {
          BROTLI_SWAP(uint32_t, histo, j, i);
        }
      }
    }
    h23 = histo[2] + histo[3];
    histomax = BROTLI_MAX(uint32_t, h23, histo[0]);
    return (kFourSymbolHistogramCost +
            3 * h23 + 2 * (histo[0] + histo[1]) - histomax);
  }

  {
    /* In this loop we compute the entropy of the histogram and simultaneously
       build a simplified histogram of the code length codes where we use the
       zero repeat code 17, but we don't use the non-zero repeat code 16. */
    size_t max_depth = 1, nnz = 0;
    uint32_t depth_histo[BROTLI_CODE_LENGTH_CODES] = {0};
    uint32_t nnz_data[data_size + 32];
    const float total_count = (float) histogram->total_count_;
    const float log2total = FastLog2(histogram->total_count_);

    for (i = 0; i < data_size;) {
      if (histogram->data_[i] > 0) {
        nnz_data[nnz] = histogram->data_[i];
        ++i;
        ++nnz;
      } else {
        /* Compute the run length of zeros and add the appropriate number of 0
           and 17 code length codes to the code length code histogram. */
        uint32_t reps = 1;
        size_t k;
        for (k = i + 1; k < data_size && histogram->data_[k] == 0; ++k) {
          ++reps;
        }
        i += reps;
        if (i == data_size) {
          /* Don't add any cost for the last zero run, since these are encoded
             only implicitly. */
          break;
        }
        if (reps < 3) {
          depth_histo[0] += reps;
        } else {
          reps -= 2;
          while (reps > 0) {
            ++depth_histo[BROTLI_REPEAT_ZERO_CODE_LENGTH];
            /* Add the 3 extra bits for the 17 code length code. */
            bits += 3;
            reps >>= 3;
          }
        }
      }
    }

    bits += FN(CostComputation)(depth_histo, nnz_data, nnz, total_count, log2total);
  }
  return bits;
}

#undef HistogramType
