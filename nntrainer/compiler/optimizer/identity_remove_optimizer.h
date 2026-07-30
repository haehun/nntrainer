// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file identity_remove_optimizer.h
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer which removes identity layers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __IDENTITY_REMOVE_OPTIMIZER_H__
#define __IDENTITY_REMOVE_OPTIMIZER_H__

#include <string>

#include <graph_optimizer.h>

namespace nntrainer {

/**
 * @brief Graph optimizer class which removes identity layers from the graph
 * @note An identity layer is removed only when it has exactly one input
 * connection and exactly one consumer. A terminal identity (no consumer) is
 * kept because it may be the model's designated output, and a fanned-out
 * identity is kept because redistributing several consumers onto one source
 * tensor is MultioutRealizer's job, not this pass'.
 * @note Consumers are counted over the whole graph rather than read off the
 * producer, since output connections exist only after compile() and several
 * consumers reading the same producer index collapse into one slot.
 *
 */
class IdentityRemoveOptimizer final : public GraphOptimizer {
public:
  /**
   * @brief Construct a new Identity Remove Optimizer object
   *
   */
  IdentityRemoveOptimizer() = default;

  /**
   * @brief Destroy the Identity Remove Optimizer object
   *
   */
  ~IdentityRemoveOptimizer() = default;

  /**
   * @brief graph optimizer creates a shallow copied graph based on the
   * reference
   * @note removes eligible identity layers from GraphRepresentation and
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
    return IdentityRemoveOptimizer::type;
  }

  static constexpr const char *type = "remove_identity";
};

} // namespace nntrainer

#endif // __IDENTITY_REMOVE_OPTIMIZER_H__
