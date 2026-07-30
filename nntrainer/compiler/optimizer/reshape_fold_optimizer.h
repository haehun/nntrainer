// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file reshape_fold_optimizer.h
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer which folds consecutive reshape layers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __RESHAPE_FOLD_OPTIMIZER_H__
#define __RESHAPE_FOLD_OPTIMIZER_H__

#include <string>

#include <graph_optimizer.h>

namespace nntrainer {

/**
 * @brief Graph optimizer class which folds a run of consecutive reshape layers
 * into the last one
 * @details ReshapeLayer's target_shape is absolute: finalize() takes the batch
 * from the input and requires the remaining feature length to match it. So in
 * `x -> reshape(A) -> reshape(B)`, B already describes the final shape and A
 * only has to agree with x on feature length, which reshape preserves. Dropping
 * every reshape but the last is therefore semantics preserving.
 * @note Only layers of type "reshape" are folded. "flatten" derives from
 * ReshapeLayer but computes its target shape from its input dimensions during
 * finalize(), so it is not interchangeable here and is left alone.
 * @note A reshape whose output has anything other than exactly one consumer is
 * kept, since its tensor is observed elsewhere.
 *
 */
class ReshapeFoldOptimizer final : public GraphOptimizer {
public:
  /**
   * @brief Construct a new Reshape Fold Optimizer object
   *
   */
  ReshapeFoldOptimizer() = default;

  /**
   * @brief Destroy the Reshape Fold Optimizer object
   *
   */
  ~ReshapeFoldOptimizer() = default;

  /**
   * @brief graph optimizer creates a shallow copied graph based on the
   * reference
   * @note folds runs of consecutive reshape layers into the last reshape of
   * each run
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
    return ReshapeFoldOptimizer::type;
  }

  static constexpr const char *type = "fold_reshape";
};

} // namespace nntrainer

#endif // __RESHAPE_FOLD_OPTIMIZER_H__
