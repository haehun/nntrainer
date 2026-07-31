// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file multiout_fold_optimizer.cpp
 * @date 31 July 2026
 * @brief NNTrainer graph optimizer which folds multiout layers with one
 * consumer
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <multiout_fold_optimizer.h>

#include <layer_node.h>
#include <multiout_layer.h>
#include <optimizer_utils.h>

#include <nntrainer_log.h>

#include <unordered_set>

namespace nntrainer {

GraphRepresentation
MultioutFoldOptimizer::optimize(const GraphRepresentation &reference) {
  auto consumer_count = countConsumers(reference);

  std::unordered_set<LayerNode *> dropped;
  for (auto &node : reference) {
    if (istrequal(node->getType(), MultiOutLayer::type) &&
        node->getNumInputConnections() == 1 &&
        consumer_count[node->getName()] == 1) {
      dropped.insert(node.get());
    }
  }

  ml_logd("[%s] folding %zu multiout node(s) out of %zu", getType().c_str(),
          dropped.size(), reference.size());

  return spliceOutNodes(reference, dropped);
}

} // namespace nntrainer
