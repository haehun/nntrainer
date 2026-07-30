// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file concat_fold_optimizer.h
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer which folds nested concat layers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __CONCAT_FOLD_OPTIMIZER_H__
#define __CONCAT_FOLD_OPTIMIZER_H__

#include <string>

#include <graph_optimizer.h>

namespace nntrainer {

/**
 * @brief Graph optimizer class which folds a concat feeding another concat
 * along the same axis into its consumer
 * @details Concatenation along one axis is associative, so
 * `concat(a, concat(b, c), d)` and `concat(a, b, c, d)` produce the same
 * tensor. Flattening the nest saves an intermediate buffer and a copy.
 * @note Both layers must carry an explicit axis property. ConcatLayer defaults
 * an unset axis to channel or width depending on the first input's channel
 * count, which is only known during finalize(), so an implicit axis cannot be
 * compared here and the pair is left alone. This is what keeps the pass off
 * the concat layers RecurrentRealizer emits, which set no axis.
 * @note An inner concat whose output has anything other than exactly one
 * consumer is kept, since its tensor is observed elsewhere.
 *
 */
class ConcatFoldOptimizer final : public GraphOptimizer {
public:
  /**
   * @brief Construct a new Concat Fold Optimizer object
   *
   */
  ConcatFoldOptimizer() = default;

  /**
   * @brief Destroy the Concat Fold Optimizer object
   *
   */
  ~ConcatFoldOptimizer() = default;

  /**
   * @brief graph optimizer creates a shallow copied graph based on the
   * reference
   * @note flattens nested concat layers which share an explicit axis
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
    return ConcatFoldOptimizer::type;
  }

  static constexpr const char *type = "fold_concat";
};

} // namespace nntrainer

#endif // __CONCAT_FOLD_OPTIMIZER_H__
