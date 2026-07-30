// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file no_op_optimizer.h
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer which returns the graph untouched
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NO_OP_OPTIMIZER_H__
#define __NO_OP_OPTIMIZER_H__

#include <string>

#include <graph_optimizer.h>

namespace nntrainer {

/**
 * @brief Graph optimizer class which performs no optimization at all
 * @details Identity pass. It exists so that the optimizer pipeline can be
 * exercised end to end before any real pass lands, and so that a new pass has
 * a minimal reference to copy from. It is also useful as a control when
 * measuring what a real pass actually buys.
 *
 */
class NoOpOptimizer final : public GraphOptimizer {
public:
  /**
   * @brief Construct a new No Op Optimizer object
   *
   */
  NoOpOptimizer() = default;

  /**
   * @brief Destroy the No Op Optimizer object
   *
   */
  ~NoOpOptimizer() = default;

  /**
   * @brief return the reference graph as is
   * @note the returned graph shares every LayerNode with the reference, so
   * this is a shallow copy of the node vector and not a clone of the graph
   *
   * @param reference GraphRepresentation to be optimized
   * @return GraphRepresentation the reference, unmodified
   */
  GraphRepresentation optimize(const GraphRepresentation &reference) override;

  /**
   * @copydoc GraphOptimizer::getType()
   */
  const std::string getType() const override { return NoOpOptimizer::type; }

  static constexpr const char *type = "no_op";
};

} // namespace nntrainer

#endif // __NO_OP_OPTIMIZER_H__
