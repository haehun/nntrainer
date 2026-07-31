// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file split_remove_optimizer.cpp
 * @date 31 July 2026
 * @brief NNTrainer graph optimizer which removes identity split layers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <split_remove_optimizer.h>

#include <common_properties.h>
#include <layer_node.h>
#include <optimizer_utils.h>
#include <split_layer.h>

#include <nntrainer_log.h>

#include <unordered_set>

namespace nntrainer {

GraphRepresentation
SplitRemoveOptimizer::optimize(const GraphRepresentation &reference) {
  auto consumer_count = countConsumers(reference);

  std::unordered_set<LayerNode *> dropped;
  for (auto &node : reference) {
    if (!istrequal(node->getType(), SplitLayer::type) ||
        node->getNumInputConnections() != 1 ||
        consumer_count[node->getName()] != 1) {
      continue;
    }

    /// @note an unset split_number is resolved from the input dimensions during
    /// finalize(), so it cannot be reasoned about here
    auto split_number = getExportedProperty(*node, props::SplitNumber::key);
    if (split_number.has_value() && split_number.value() == "1") {
      dropped.insert(node.get());
    }
  }

  ml_logd("[%s] removing %zu split node(s) out of %zu", getType().c_str(),
          dropped.size(), reference.size());

  return spliceOutNodes(reference, dropped);
}

} // namespace nntrainer
