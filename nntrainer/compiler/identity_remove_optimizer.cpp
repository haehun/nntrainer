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

#include <connection.h>
#include <identity_layer.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nntrainer {

static constexpr unsigned SINGLE_INOUT_IDX = 0;

GraphRepresentation
IdentityRemoveOptimizer::optimize(const GraphRepresentation &reference) {
  std::unordered_map<std::string, LayerNode *> existing_nodes;
  std::unordered_map<std::string, unsigned> consumer_count;

  for (auto &node : reference) {
    existing_nodes.emplace(node->getName(), node.get());
  }

  /// @note a fan-out cannot be detected from the producer side: compile() keys
  /// output connections by the producer's output index, so two consumers both
  /// reading slot 0 collapse into a single output connection. Counting input
  /// connections over the whole graph is the reliable signal, and it also
  /// leaves a terminal identity (zero consumers) alone since that may be the
  /// model's designated output.
  for (auto &node : reference) {
    for (unsigned i = 0, num = node->getNumInputConnections(); i < num; ++i) {
      ++consumer_count[node->getInputConnectionName(i)];
    }
  }

  std::unordered_set<LayerNode *> removed;
  std::vector<LayerNode *> removable;
  for (auto &node : reference) {
    if (istrequal(node->getType(), IdentityLayer::type) &&
        node->getNumInputConnections() == 1 &&
        consumer_count[node->getName()] == 1) {
      removable.push_back(node.get());
      removed.insert(node.get());
    }
  }

  /// name of a removed identity -> the connection that replaces it
  std::unordered_map<std::string, Connection> replacement;

  for (auto *node : removable) {
    /// @note walk to the first surviving producer so that a chain of
    /// identities collapses in a single pass, whatever order the nodes appear
    /// in the vector
    auto *source = node;
    unsigned source_index = SINGLE_INOUT_IDX;
    while (removed.find(source) != removed.end()) {
      source_index = source->getInputConnectionIndex(SINGLE_INOUT_IDX);
      source =
        existing_nodes.at(source->getInputConnectionName(SINGLE_INOUT_IDX));
    }

    replacement.emplace(node->getName(),
                        Connection(source->getName(), source_index));
  }

  for (auto &node : reference) {
    if (removed.find(node.get()) != removed.end()) {
      continue;
    }

    for (unsigned i = 0, num = node->getNumInputConnections(); i < num; ++i) {
      auto found = replacement.find(node->getInputConnectionName(i));
      if (found != replacement.end()) {
        node->setInputConnectionName(i, found->second.getName());
        node->setInputConnectionIndex(i, found->second.getIndex());
      }
    }

    /// @note output connections are only populated once the graph has been
    /// compiled, and they are keyed by the producer's output index, so the
    /// slot is reused while only the consumer name is redirected
    for (unsigned i = 0, num = node->getNumOutputConnections(); i < num; ++i) {
      auto *out = node->getOutputConnection(i);
      if (out == nullptr) {
        continue;
      }

      auto name = out->getName();
      auto index = out->getIndex();
      bool redirected = false;

      while (removed.find(existing_nodes.at(name)) != removed.end()) {
        auto *identity = existing_nodes.at(name);
        name = identity->getOutputConnection(SINGLE_INOUT_IDX)->getName();
        index = identity->getOutputConnection(SINGLE_INOUT_IDX)->getIndex();
        redirected = true;
      }

      if (redirected) {
        node->setOutputConnection(i, name, index);
      }
    }
  }

  ml_logd("[%s] removed %zu identity node(s) out of %zu", getType().c_str(),
          removable.size(), reference.size());

  GraphRepresentation processed;
  for (auto &node : reference) {
    if (removed.find(node.get()) == removed.end()) {
      processed.push_back(node);
    }
  }

  return processed;
}

} // namespace nntrainer
