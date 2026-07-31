// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file multiout_fold_optimizer.h
 * @date 31 July 2026
 * @brief NNTrainer graph optimizer which folds multiout layers with one
 * consumer
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __MULTIOUT_FOLD_OPTIMIZER_H__
#define __MULTIOUT_FOLD_OPTIMIZER_H__

#include <string>

#include <graph_optimizer.h>

namespace nntrainer {

/**
 * @brief Graph optimizer class which folds single-output multiout layers
 * @note A multiout layer with exactly one consumer is semantically identical
 * to an identity layer and can be removed the same way. This is complementary
 * to identity layer removal but separated since multiout serves other purposes
 * (slicing inputs for multi-branch networks) and should only be removed when it
 * genuinely has one consumer.
 * @details The multiout layer is an internal node used during graph
 * construction to support variable arity, and is safe to remove at any point
 * if it has only one output consumer.
 *
 */
class MultioutFoldOptimizer final : public GraphOptimizer {
public:
  /**
   * @brief Construct a new Multiout Fold Optimizer object
   *
   */
  MultioutFoldOptimizer() = default;

  /**
   * @brief Destroy the Multiout Fold Optimizer object
   *
   */
  ~MultioutFoldOptimizer() = default;

  /**
   * @brief graph optimizer creates a shallow copied graph based on the
   * reference
   * @note removes multiout layers with exactly one consumer from the graph and
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
    return MultioutFoldOptimizer::type;
  }

  static constexpr const char *type = "fold_multiout";
};

} // namespace nntrainer

#endif // __MULTIOUT_FOLD_OPTIMIZER_H__
