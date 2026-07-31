// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file split_remove_optimizer.h
 * @date 31 July 2026
 * @brief NNTrainer graph optimizer which removes identity split layers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __SPLIT_REMOVE_OPTIMIZER_H__
#define __SPLIT_REMOVE_OPTIMIZER_H__

#include <string>

#include <graph_optimizer.h>

namespace nntrainer {

/**
 * @brief Graph optimizer class which removes identity split layers
 * @note A split with exactly one output (split_number == 1) is a no-op that
 * just copies its input tensor. An identity split (one input, one output) is
 * removed and its producer and consumer are directly connected. This is
 * complementary to the identity layer removal pass.
 * @details Section 6.4 REMOVE_IDENTITY_SPLIT of OPTIMIZATION_METHODS.md
 *
 */
class SplitRemoveOptimizer final : public GraphOptimizer {
public:
  /**
   * @brief Construct a new Split Remove Optimizer object
   *
   */
  SplitRemoveOptimizer() = default;

  /**
   * @brief Destroy the Split Remove Optimizer object
   *
   */
  ~SplitRemoveOptimizer() = default;

  /**
   * @brief graph optimizer creates a shallow copied graph based on the
   * reference
   * @note removes identity split layers (split_number == 1) from the graph and
   * reconnects their neighbors directly
   * @param reference GraphRepresentation to be optimized
   * @return GraphRepresentation optimized graph representation
   * @throw std::out_of_range if graph is ill formed
   *
   */
  GraphRepresentation optimize(const GraphRepresentation &reference) override;

  /**
   * @copydoc GraphOptimizer::getType()
   */
  const std::string getType() const override {
    return SplitRemoveOptimizer::type;
  }

  static constexpr const char *type = "remove_split";
};

} // namespace nntrainer

#endif // __SPLIT_REMOVE_OPTIMIZER_H__
