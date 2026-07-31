// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file optimizer_utils.h
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer common helpers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __OPTIMIZER_UTILS_H__
#define __OPTIMIZER_UTILS_H__

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <compiler_fwd.h>

namespace nntrainer {

class LayerNode;

/**
 * @brief read a property off a layer as it would be serialized
 * @note LayerNode::getProperty() only reaches the node's own properties, and
 * layers are not required to override Layer::getProperty(), which returns an
 * empty string by default. Reading through the exporter works for any layer.
 * An unset property is skipped while exporting, so a missing value means the
 * layer left it to be resolved during finalize() and it cannot be reasoned
 * about at this stage.
 *
 * @param node node to read from
 * @param key property key to look for
 * @return std::optional<std::string> value, or nullopt when unset
 */
std::optional<std::string> getExportedProperty(const LayerNode &node,
                                               const std::string &key);

/**
 * @brief count how many input connections in the graph refer to each node name
 * @note a fan-out cannot be read off the producer: output connections exist
 * only after NetworkGraph::compile() and are keyed by the producer's output
 * index, so several consumers reading the same index collapse into one slot.
 * Counting input connections is the reliable signal, and a count of zero
 * identifies a node whose output nothing consumes.
 *
 * @param reference graph to scan
 * @return std::unordered_map<std::string, unsigned> node name -> consumer count
 */
std::unordered_map<std::string, unsigned>
countConsumers(const GraphRepresentation &reference);

/**
 * @brief drop nodes from the graph, reconnecting each dropped node's producer
 * directly to its consumers
 * @note every dropped node must have exactly one input connection and must be
 * a passthrough with respect to the tensor it forwards, since consumers are
 * repointed at the dropped node's producer. Chains of dropped nodes are walked
 * through, so a whole run collapses in one call.
 *
 * @param reference graph to rewrite
 * @param dropped nodes to remove, as raw pointers into @a reference
 * @return GraphRepresentation graph without the dropped nodes
 * @throw std::out_of_range if a dropped node refers to a producer absent from
 * the graph
 */
GraphRepresentation
spliceOutNodes(const GraphRepresentation &reference,
               const std::unordered_set<LayerNode *> &dropped);

} // namespace nntrainer

#endif // __OPTIMIZER_UTILS_H__
