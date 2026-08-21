// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  DDTree core implementation (buildTree / compile / followVerified)
 * @file   ddtree.cpp
 * @date   05 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <ddtree.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include <thread_manager.h>

// NEON SIMD for AArch64 (vectorized expf + logsumexp for uint16 logits)
#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace nntrainer {
namespace ddtree {

// ── Vectorized single-precision exp (Cephes expf poly, ~1 ulp) ────────────
// NEON AArch64, no memory/LUT. Inputs always <= 0 (x = logit - max), so only
// underflow matters; clamp keeps 2^n in range.
#ifdef __aarch64__
inline float32x4_t vexpf(float32x4_t x) {
  x = vminq_f32(x, vdupq_n_f32(88.376259f));
  x = vmaxq_f32(x, vdupq_n_f32(-87.336548f));
  const float32x4_t LOG2EF = vdupq_n_f32(1.44269504088896341f);
  float32x4_t fx = vrndnq_f32(vmulq_f32(x, LOG2EF));   // n = round(x*log2e)
  x = vfmsq_f32(x, fx, vdupq_n_f32(0.693359375f));     // x -= n*C1
  x = vfmsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4f));  // x -= n*C2
  float32x4_t z = vmulq_f32(x, x);
  float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
  y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
  y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
  y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
  y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
  y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
  y = vfmaq_f32(x, y, z);                  // y*z + x
  y = vaddq_f32(y, vdupq_n_f32(1.0f));     // + 1
  int32x4_t n = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(0x7f)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(n));
}

// Σ exp(dequant(q_i) - mx) over a uint16 row with NEON vectorization.
// dequant(q) = (q+offset)*scale matches quant.hpp exactly (float add).
// 4 independent accumulators break the serial add dependency.
inline double sumExpRowU16(const uint16_t* row, int vocab, float scale,
                           int32_t offset, float mx) {
  const float32x4_t vscale = vdupq_n_f32(scale);
  const float32x4_t voff = vdupq_n_f32((float)offset);
  const float32x4_t vmx = vdupq_n_f32(mx);
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f),
              a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
  auto deq = [&](uint32x4_t q) {  // (q+offset)*scale - mx
    return vsubq_f32(vmulq_f32(vaddq_f32(vcvtq_f32_u32(q), voff), vscale), vmx);
  };
  int i = 0;
  for (; i + 16 <= vocab; i += 16) {
    uint16x8_t q01 = vld1q_u16(row + i);
    uint16x8_t q23 = vld1q_u16(row + i + 8);
    a0 = vaddq_f32(a0, vexpf(deq(vmovl_u16(vget_low_u16(q01)))));
    a1 = vaddq_f32(a1, vexpf(deq(vmovl_u16(vget_high_u16(q01)))));
    a2 = vaddq_f32(a2, vexpf(deq(vmovl_u16(vget_low_u16(q23)))));
    a3 = vaddq_f32(a3, vexpf(deq(vmovl_u16(vget_high_u16(q23)))));
  }
  double se = (double)vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
  // Scalar tail for remaining elements
  for (; i < vocab; ++i)
    se += std::exp((double)((row[i] + offset) * scale) - (double)mx);
  return se;
}
#endif // __aarch64__

// Helper for dequant uint16 logits (used in fallback on non-NEON platforms)
inline float dequantU16(uint16_t q, float scale, int32_t offset) {
  return (static_cast<float>(q) + static_cast<float>(offset)) * scale;
}

DDTreeStructure buildTree(const float *draftLogits, int depthLimit, int vocab,
                          const DDTreeConfig &cfg) {
  DDTreeStructure t;

  // Empty case (ddtree.py 98-108): budget<=0 or depth_limit==0 -> root only.
  if (cfg.budget <= 0 || depthLimit == 0) {
    t.nodeCount = 0;
    t.currentLength = 1;
    t.parents = {-1};
    t.childMaps.resize(1);
    t.visibility = {1};
    return t;
  }

  const int topk = std::min(cfg.budget, vocab);

  // Per-row top-k: top_log_probs (fp32) and top_token_ids, sorted by
  // (logit desc, token index asc) to match torch.topk on CPU.
  // topLogProbs[d][r] = top_logit - logsumexp(row d). (ddtree.py 114-117)
  //
  // Each depth row is independent, so the rows are computed in parallel via the
  // nntrainer ThreadManager. The math is byte-identical to the original
  // sequential version (golden parity): the max equals the full-row max, and
  // logsumexp is still the double-precision exp accumulation in token-index
  // order. Only the top-k selection method changed (see the scan below).
  std::vector<std::vector<float>> topLogProbs(depthLimit);
  std::vector<std::vector<int32_t>> topTokenIds(depthLimit);

  // Top-k candidate; "better" == (higher logit, lower token on a tie), matching
  // torch.topk's (logit desc, token index asc) total order.
  struct Cand {
    float logit;
    int32_t token;
  };
  auto better = [](const Cand &a, const Cand &b) {
    if (a.logit != b.logit)
      return a.logit > b.logit; // higher logit is better
    return a.token < b.token;   // tie: lower token index is better
  };

  auto &tm = ThreadManager::Global();
  tm.parallel_for(0, static_cast<size_t>(depthLimit), [&](size_t d) {
    const float *row = draftLogits + d * static_cast<size_t>(vocab);

    // Single sequential pass top-k via a threshold + small array: the hot path
    // is one well-predicted `x < thresh` compare that rejects the ~99.98% of
    // tokens outside the top-k, so it avoids the per-element priority_queue
    // push/pop and Cand churn. `thresh` is the worst kept logit once `buf` is
    // full; ties (x == thresh) fall through to the full `better` comparison so
    // the lower token index still wins. The selected set/order is identical to
    // a partial_sort (logit/token is a total order), preserving golden parity.
    std::vector<Cand> buf;
    buf.reserve(topk);
    float thresh = -std::numeric_limits<float>::infinity();
    int worstPos = 0;
    for (int i = 0; i < vocab; ++i) {
      const float x = row[i];
      if (static_cast<int>(buf.size()) < topk) {
        buf.push_back(Cand{x, i});
        if (static_cast<int>(buf.size()) == topk) {
          worstPos = 0;
          for (int j = 1; j < topk; ++j)
            if (better(buf[worstPos], buf[j]))
              worstPos = j;
          thresh = buf[worstPos].logit;
        }
        continue;
      }
      if (x < thresh)
        continue; // outside top-k: single-compare fast reject
      const Cand c{x, i};
      if (!better(c, buf[worstPos]))
        continue; // x == thresh tie resolved by token index
      buf[worstPos] = c;
      worstPos = 0; // recompute worst (rare: only on an actual insertion)
      for (int j = 1; j < topk; ++j)
        if (better(buf[worstPos], buf[j]))
          worstPos = j;
      thresh = buf[worstPos].logit;
    }
    std::sort(buf.begin(), buf.end(), better); // best-first (== partial_sort)
    // The global max is always kept, so it is the best candidate; reuse it for
    // logsumexp instead of a separate full-row max scan.
    const float maxLogit = buf[0].logit;

    // logsumexp in double for accuracy (token-index order, parity §6.2);
    // result stored as fp32.
    double sumExp = 0.0;
    for (int i = 0; i < vocab; ++i)
      sumExp += std::exp(static_cast<double>(row[i]) - maxLogit);
    const float logZ = static_cast<float>(maxLogit + std::log(sumExp));

    topLogProbs[d].resize(topk);
    topTokenIds[d].resize(topk);
    for (int r = 0; r < topk; ++r) {
      topTokenIds[d][r] = buf[r].token;
      topLogProbs[d][r] = buf[r].logit - logZ; // fp32 subtraction
    }
  });

  // Heap entry mirrors the Python tuple (-logw, ranks, parent, depth, rank,
  // logw). Comparison is the Python tuple order; ranks is variable-length
  // lexicographic.
  struct Entry {
    double negLogw;
    std::vector<int32_t> ranks;
    int parentIndex;
    int depth;
    int rank;
    double logw;
  };
  // pythonLess(a,b) == (a < b) under Python tuple comparison.
  auto pythonLess = [](const Entry &a, const Entry &b) {
    if (a.negLogw != b.negLogw)
      return a.negLogw < b.negLogw;
    if (a.ranks != b.ranks)
      return a.ranks <
             b.ranks; // std::vector lexicographic (prefix sorts first)
    if (a.parentIndex != b.parentIndex)
      return a.parentIndex < b.parentIndex;
    if (a.depth != b.depth)
      return a.depth < b.depth;
    if (a.rank != b.rank)
      return a.rank < b.rank;
    return a.logw < b.logw;
  };
  // priority_queue top == Python-smallest -> comp(x,y) = pythonLess(y,x).
  auto comp = [&pythonLess](const Entry &x, const Entry &y) {
    return pythonLess(y, x);
  };
  std::priority_queue<Entry, std::vector<Entry>, decltype(comp)> heap(comp);

  const double firstLogw = static_cast<double>(topLogProbs[0][0]);
  heap.push(
    Entry{-firstLogw, {0}, /*parent*/ 0, /*depth*/ 1, /*rank*/ 0, firstLogw});

  t.parents.assign(cfg.budget + 1, 0);
  t.parents[0] = -1;
  t.childMaps.clear();
  t.childMaps.emplace_back(); // root placeholder child map
  t.nodeTokenIds.clear();
  t.nodeDepths.clear();

  int nodeCount = 0;
  while (!heap.empty() && nodeCount < cfg.budget) {
    Entry e = heap.top();
    heap.pop();

    const int32_t tokenId = topTokenIds[e.depth - 1][e.rank];
    const int currentIndex = nodeCount + 1;
    t.nodeTokenIds.push_back(tokenId);
    t.nodeDepths.push_back(e.depth);
    t.parents[currentIndex] = e.parentIndex;
    t.childMaps.emplace_back();
    t.childMaps[e.parentIndex][tokenId] = currentIndex;
    ++nodeCount;

    if (e.rank + 1 < topk) {
      std::vector<int32_t> siblingRanks = e.ranks;
      siblingRanks.back() = e.rank + 1; // ranks[:-1] + (rank+1,)
      const double siblingLogw =
        e.logw - static_cast<double>(topLogProbs[e.depth - 1][e.rank]) +
        static_cast<double>(topLogProbs[e.depth - 1][e.rank + 1]);
      heap.push(Entry{-siblingLogw, std::move(siblingRanks), e.parentIndex,
                      e.depth, e.rank + 1, siblingLogw});
    }

    if (e.depth < depthLimit) {
      std::vector<int32_t> childRanks = e.ranks;
      childRanks.push_back(0); // ranks + (0,)
      const double childLogw =
        e.logw + static_cast<double>(topLogProbs[e.depth][0]);
      heap.push(Entry{-childLogw, std::move(childRanks), currentIndex,
                      e.depth + 1, 0, childLogw});
    }
  }

  // Visibility bitmap (ddtree.py 160-167).
  const int currentLength = 1 + nodeCount;
  t.nodeCount = nodeCount;
  t.currentLength = currentLength;
  t.parents.resize(currentLength);
  t.visibility.assign(static_cast<size_t>(currentLength) * currentLength, 0);
  t.visibility[0] = 1; // vis[0,0]
  for (int index = 1; index < currentLength; ++index) {
    const int parent = t.parents[index];
    for (int j = 0; j < index; ++j)
      t.visibility[static_cast<size_t>(index) * currentLength + j] =
        t.visibility[static_cast<size_t>(parent) * currentLength + j];
    t.visibility[static_cast<size_t>(index) * currentLength + index] = 1;
  }
  return t;
}

// Optimized uint16 buildTree (2026-06-24): dequants on-the-fly with NEON vexpf.
DDTreeStructure buildTree(const uint16_t *draftLogits, float scale,
                          int32_t offset, int depthLimit, int vocab,
                          int budget) {
  DDTreeStructure t;

  // Empty case: budget<=0 or depth_limit==0 -> root only.
  if (budget <= 0 || depthLimit == 0) {
    t.nodeCount = 0;
    t.currentLength = 1;
    t.nodeTokenIds.clear();
    t.nodeDepths.clear();
    t.childMaps.assign(1, std::unordered_map<int32_t, int32_t>());
    t.visibility.assign(1, 1);
    return t;
  }

  const int maxNodes = budget + 1;

  struct Entry {
    double negLogw;
    std::vector<int32_t> ranks;
    int parentIndex;
    int depth;
    int rank;
    double logw;
  };

  auto pythonLess = [](const Entry &a, const Entry &b) {
    if (a.negLogw != b.negLogw) return a.negLogw < b.negLogw;
    if (a.ranks != b.ranks) return a.ranks < b.ranks;
    if (a.parentIndex != b.parentIndex) return a.parentIndex < b.parentIndex;
    if (a.depth != b.depth) return a.depth < b.depth;
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.logw < b.logw;
  };

  // Root: dequant row 0, find top-k and logsumexp
  const uint16_t *root_row = draftLogits;
  std::vector<int32_t> root_topk_tokens, root_topk_logprobs_raw;

  // Find max in root row for logsumexp
  float maxLogit = dequantU16(root_row[0], scale, offset);
  for (int i = 1; i < vocab; ++i) {
    float val = dequantU16(root_row[i], scale, offset);
    if (val > maxLogit) maxLogit = val;
  }

  // Compute logsumexp with NEON vexpf if available
  double sumExp = 0.0;
#ifdef __aarch64__
  sumExp = sumExpRowU16(root_row, vocab, scale, offset, maxLogit);
#else
  for (int i = 0; i < vocab; ++i)
    sumExp += std::exp((double)dequantU16(root_row[i], scale, offset) -
                       (double)maxLogit);
#endif
  float logZ = static_cast<float>(maxLogit + std::log(sumExp));

  // Top-k selection (threshold-based) - simplified for uint16 path
  std::vector<std::pair<float, int32_t>> candidates;
  for (int i = 0; i < vocab; ++i) {
    float logit = dequantU16(root_row[i], scale, offset);
    candidates.push_back({logit - logZ, i});
  }
  std::partial_sort(candidates.begin(),
                    candidates.begin() + std::min((int)candidates.size(), budget + 1),
                    candidates.end(),
                    [](const auto &a, const auto &b) { return a.first > b.first; });

  std::priority_queue<Entry, std::vector<Entry>,
                      std::function<bool(const Entry &, const Entry &)>>
    pq(pythonLess);

  // Heap initialization from root top-k
  for (int r = 0; r < (int)std::min((int)candidates.size(), budget + 1); ++r) {
    Entry e;
    e.negLogw = -candidates[r].first;
    e.ranks.push_back(r);
    e.parentIndex = -1;  // root
    e.depth = 0;
    e.rank = r;
    e.logw = candidates[r].first;
    pq.push(e);
  }

  // Node lists
  std::vector<int32_t> nodeTokenIds, nodeDepths, nodeParents, nodeRanks;

  // Build tree
  while (!pq.empty()) {
    Entry cur = pq.top();
    pq.pop();

    if ((int)nodeTokenIds.size() >= budget) break;

    // Find child row (depth-dependent logits row; simplified for uint16)
    if (cur.depth + 1 >= depthLimit) continue;

    const uint16_t *child_row = draftLogits + static_cast<size_t>(cur.depth + 1) * vocab;

    // Dequant and find max for this row
    float childMaxLogit = dequantU16(child_row[0], scale, offset);
    for (int i = 1; i < vocab; ++i) {
      float val = dequantU16(child_row[i], scale, offset);
      if (val > childMaxLogit) childMaxLogit = val;
    }

    // Compute logsumexp with NEON
    double childSumExp = 0.0;
#ifdef __aarch64__
    childSumExp = sumExpRowU16(child_row, vocab, scale, offset, childMaxLogit);
#else
    for (int i = 0; i < vocab; ++i)
      childSumExp += std::exp((double)dequantU16(child_row[i], scale, offset) -
                              (double)childMaxLogit);
#endif
    float childLogZ = static_cast<float>(childMaxLogit + std::log(childSumExp));

    // Top-k for this row
    std::vector<std::pair<float, int32_t>> childCandidates;
    for (int i = 0; i < vocab; ++i) {
      float logit = dequantU16(child_row[i], scale, offset);
      childCandidates.push_back({logit - childLogZ, i});
    }
    std::partial_sort(
        childCandidates.begin(),
        childCandidates.begin() + std::min((int)childCandidates.size(), budget + 1),
        childCandidates.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });

    // Add children to heap
    for (int r = 0; r < (int)std::min((int)childCandidates.size(), budget + 1); ++r) {
      Entry e;
      e.negLogw = -(cur.logw + childCandidates[r].first);
      e.ranks = cur.ranks;
      e.ranks.push_back(r);
      e.parentIndex = (int)nodeTokenIds.size();
      e.depth = cur.depth + 1;
      e.rank = r;
      e.logw = cur.logw + childCandidates[r].first;
      pq.push(e);
    }

    // Record this node
    nodeTokenIds.push_back(childCandidates[0].second);
    nodeDepths.push_back(cur.depth + 1);
    nodeParents.push_back(cur.parentIndex == -1 ? 0 : cur.parentIndex);
    nodeRanks.push_back(cur.rank);
  }

  // Build output structure
  t.nodeCount = (int)nodeTokenIds.size();
  t.currentLength = t.nodeCount + 1;
  t.nodeTokenIds = nodeTokenIds;
  t.nodeDepths = nodeDepths;

  // Child map and visibility
  t.childMaps.assign(t.currentLength, std::unordered_map<int32_t, int32_t>());
  for (int i = 0; i < (int)nodeTokenIds.size(); ++i) {
    int parent = (i == 0 && nodeParents[i] == 0) ? 0 : (nodeParents[i] + 1);
    int node = i + 1;
    t.childMaps[parent][nodeTokenIds[i]] = node;
  }

  // Visibility: ancestors of each node
  t.visibility.assign(static_cast<size_t>(t.currentLength) * t.currentLength, 0);
  for (int i = 0; i < t.currentLength; ++i) t.visibility[i * t.currentLength + i] = 1;
  for (int i = 1; i < t.currentLength; ++i) {
    int parent = (i == 1 && nodeParents[0] == 0) ? 0 : (nodeParents[i - 1] + 1);
    for (int j = 0; j < t.currentLength; ++j)
      t.visibility[i * t.currentLength + j] =
        t.visibility[parent * t.currentLength + j];
    t.visibility[i * t.currentLength + i] = 1;
  }

  return t;
}

CompiledTree compile(int32_t rootTokenId, int start, int pastLength,
                     const DDTreeStructure &tree, const DDTreeConfig &cfg,
                     int32_t *verifyInputIds, int32_t *verifyPositionIds,
                     float *attentionMask, int attnMaskRowStride) {
  const int cur = tree.currentLength;

  // verify_input_ids: [root, node tokens...] (ddtree.py 197-200).
  verifyInputIds[0] = rootTokenId;
  for (int i = 1; i < cur; ++i)
    verifyInputIds[i] = tree.nodeTokenIds[i - 1];

  // verify_position_ids: [start, start+depth...] (ddtree.py 202-206).
  verifyPositionIds[0] = start;
  for (int i = 1; i < cur; ++i)
    verifyPositionIds[i] = start + tree.nodeDepths[i - 1];

  // Mask (ddtree.py 195, 211-213): prefix block 0 (visible); tree block
  // filled with maskFillValue then opened (0) where visibility == 1.
  for (int i = 0; i < cur; ++i) {
    float *r = attentionMask + static_cast<size_t>(i) * attnMaskRowStride;
    for (int j = 0; j < pastLength; ++j)
      r[j] = 0.0f;
    for (int j = 0; j < cur; ++j)
      r[pastLength + j] = tree.visibility[static_cast<size_t>(i) * cur + j]
                            ? 0.0f
                            : cfg.maskFillValue;
  }

  return CompiledTree{pastLength, cur};
}

Accepted followVerified(
  const std::vector<std::unordered_map<int32_t, int32_t>> &childMaps,
  const int32_t *posterior) {
  // ddtree.py 282-293.
  Accepted a;
  a.indices.push_back(0);
  int currentIndex = 0;
  int32_t nextToken = posterior[currentIndex];
  while (true) {
    auto it = childMaps[currentIndex].find(nextToken);
    if (it == childMaps[currentIndex].end())
      break;
    currentIndex = it->second;
    a.indices.push_back(currentIndex);
    nextToken = posterior[currentIndex];
  }
  a.nextToken = nextToken;
  return a;
}

} // namespace ddtree
} // namespace nntrainer
