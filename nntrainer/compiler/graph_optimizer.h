// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file graph_optimizer.h
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer which rewrites a graph representation
 * without changing its semantics
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 * @details
 * A graph optimizer takes a GraphRepresentation and returns a
 * GraphRepresentation which computes the exact same function, but is expected
 * to be cheaper to execute (fused operators, dropped no-op nodes, reordered
 * layouts, ...).
 *
 * This is intentionally kept apart from GraphRealizer. A realizer *lowers* the
 * user facing description into the form the engine can run (expanding
 * recurrent scopes, materializing activations, ...) and may well change what
 * nodes mean. An optimizer runs after that, on an already lowered graph, and
 * must preserve observable behavior.
 *
 *    +-------------------+       +-------------------+
 *    |GraphRepresentation|       |GraphRepresentation|
 *    |    (as written)   |       |    (as written)   |
 *    +---------+---------+       +---------+---------+
 *              |                           |
 *    realize() | (lowering)      realize() | (lowering)
 *              v                           v
 *    +-------------------+       +-------------------+
 *    |GraphRepresentation|       |GraphRepresentation|
 *    |    (lowered)      |       |    (lowered)      |
 *    +-------------------+       +---------+---------+
 *                                          |
 *                              optimize()  | (semantics preserving)
 *                                          v
 *                                +-------------------+
 *                                |GraphRepresentation|
 *                                |    (optimized)    |
 *                                +-------------------+
 *
 * Usage:
 * @code
 * std::vector<std::unique_ptr<nntrainer::GraphOptimizer>> optimizers;
 * optimizers.emplace_back(new nntrainer::NoOpOptimizer());
 *
 * for (auto &optimizer : optimizers) {
 *   graph = optimizer->optimize(graph);
 * }
 * @endcode
 */
#ifndef __GRAPH_OPTIMIZER_H__
#define __GRAPH_OPTIMIZER_H__

#include <memory>
#include <string>
#include <vector>

#include <compiler_fwd.h>

namespace nntrainer {

/**
 * @brief Graph optimizer class
 *
 */
class GraphOptimizer {
public:
  /**
   * @brief Destroy the Graph Optimizer object
   *
   */
  virtual ~GraphOptimizer() = default;

  /**
   * @brief optimize the given graph representation
   * @note the returned graph must be functionally equivalent to the reference.
   * Nodes which the pass does not touch are expected to be shared (shallow
   * copied) with the reference rather than cloned.
   *
   * @param reference GraphRepresentation to be optimized
   * @return GraphRepresentation optimized graph representation
   * @throw std::invalid_argument if the graph is ill formed for this pass
   */
  virtual GraphRepresentation
  optimize(const GraphRepresentation &reference) = 0;

  /**
   * @brief get the type of the optimizer, used for logging and for selecting a
   * pass by name
   *
   * @return const std::string type of the optimizer
   */
  virtual const std::string getType() const = 0;
};

} // namespace nntrainer

#endif // __GRAPH_OPTIMIZER_H__
