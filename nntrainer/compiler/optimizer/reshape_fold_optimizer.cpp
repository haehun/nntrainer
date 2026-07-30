// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file reshape_fold_optimizer.cpp
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer which folds consecutive reshape layers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <reshape_fold_optimizer.h>

#include <layer_node.h>
#include <optimizer_utils.h>
#include <reshape_layer.h>

#include <nntrainer_log.h>

#include <unordered_map>

namespace nntrainer {

static constexpr unsigned SINGLE_INOUT_IDX = 0;

GraphRepresentation
ReshapeFoldOptimizer::optimize(const GraphRepresentation &reference) {
  std::unordered_map<std::string, LayerNode *> existing_nodes;
  for (auto &node : reference) {
    existing_nodes.emplace(node->getName(), node.get());
  }

  auto consumer_count = countConsumers(reference);

  auto is_foldable = [&consumer_count](const LayerNode *node) {
    return istrequal(node->getType(), ReshapeLayer::type) &&
           node->getNumInputConnections() == 1 &&
           consumer_count[node->getName()] == 1;
  };

  /// @note a reshape is dropped when its consumer is another reshape, so the
  /// last reshape of a run always survives and its absolute target_shape wins
  std::unordered_set<LayerNode *> dropped;
  for (auto &node : reference) {
    if (!istrequal(node->getType(), ReshapeLayer::type) ||
        node->getNumInputConnections() != 1) {
      continue;
    }

    auto producer =
      existing_nodes.find(node->getInputConnectionName(SINGLE_INOUT_IDX));
    if (producer != existing_nodes.end() && is_foldable(producer->second)) {
      dropped.insert(producer->second);
    }
  }

  ml_logd("[%s] folding away %zu reshape node(s) out of %zu", getType().c_str(),
          dropped.size(), reference.size());

  return spliceOutNodes(reference, dropped);
}

} // namespace nntrainer
