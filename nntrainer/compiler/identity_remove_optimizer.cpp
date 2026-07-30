// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file identity_remove_optimizer.cpp
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer which removes identity layers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <identity_remove_optimizer.h>

#include <identity_layer.h>
#include <layer_node.h>
#include <optimizer_utils.h>

#include <nntrainer_log.h>

namespace nntrainer {

GraphRepresentation
IdentityRemoveOptimizer::optimize(const GraphRepresentation &reference) {
  auto consumer_count = countConsumers(reference);

  std::unordered_set<LayerNode *> dropped;
  for (auto &node : reference) {
    if (istrequal(node->getType(), IdentityLayer::type) &&
        node->getNumInputConnections() == 1 &&
        consumer_count[node->getName()] == 1) {
      dropped.insert(node.get());
    }
  }

  ml_logd("[%s] removing %zu identity node(s) out of %zu", getType().c_str(),
          dropped.size(), reference.size());

  return spliceOutNodes(reference, dropped);
}

} // namespace nntrainer
