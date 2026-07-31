// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file permute_fold_optimizer.h
 * @date 31 July 2026
 * @brief NNTrainer graph optimizer which folds consecutive permute layers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __PERMUTE_FOLD_OPTIMIZER_H__
#define __PERMUTE_FOLD_OPTIMIZER_H__

#include <string>

#include <graph_optimizer.h>

namespace nntrainer {

/**
 * @brief Graph optimizer class which collapses a run of permute layers into one
 * @details Section 2.2 SQUASH_MULTIPLE_PERMUTE and FOLD_MULTIPLE_TRANSPOSE of
 * OPTIMIZATION_METHODS.md. Permutations compose, so a run of permute layers is
 * one permute whose direction is the composition. When the composition turns
 * out to be the identity the whole run is dropped, which is the case worth
 * having: a transpose there and back costs two full tensor copies and moves no
 * data anywhere.
 *
 * @note PermuteLayer's direction is a fixed three element property covering
 * channel, height and width, and it is read straight off the node, so unlike
 * the concat and split passes nothing here depends on the tensor dimensions.
 *
 */
class PermuteFoldOptimizer final : public GraphOptimizer {
public:
  /**
   * @brief Construct a new Permute Fold Optimizer object
   *
   */
  PermuteFoldOptimizer() = default;

  /**
   * @brief Destroy the Permute Fold Optimizer object
   *
   */
  ~PermuteFoldOptimizer() = default;

  /**
   * @brief graph optimizer creates a shallow copied graph based on the
   * reference
   * @note collapses runs of permute layers into a single composed permute, and
   * removes the run entirely when it composes to the identity
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
    return PermuteFoldOptimizer::type;
  }

  static constexpr const char *type = "fold_permute";
};

} // namespace nntrainer

#endif // __PERMUTE_FOLD_OPTIMIZER_H__
