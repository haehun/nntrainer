// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file optimizer_utils.cpp
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer common helpers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <optimizer_utils.h>

#include <connection.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <node_exporter.h>

#include <stdexcept>

namespace nntrainer {

static constexpr unsigned SINGLE_INOUT_IDX = 0;

std::optional<std::string> getExportedProperty(const LayerNode &node,
                                               const std::string &key) {
  Exporter exporter;
  node.exportTo(exporter, ml::train::ExportMethods::METHOD_STRINGVECTOR);

  auto result =
    exporter.getResult<ml::train::ExportMethods::METHOD_STRINGVECTOR>();
  if (result == nullptr) {
    return std::nullopt;
  }

  for (auto &[prop_key, prop_value] : *result) {
    if (istrequal(prop_key, key)) {
      return prop_value;
    }
  }

  return std::nullopt;
}

std::unordered_map<std::string, unsigned>
countConsumers(const GraphRepresentation &reference) {
  std::unordered_map<std::string, unsigned> consumer_count;

  for (auto &node : reference) {
    for (unsigned i = 0, num = node->getNumInputConnections(); i < num; ++i) {
      ++consumer_count[node->getInputConnectionName(i)];
    }
  }

  return consumer_count;
}

GraphRepresentation
spliceOutNodes(const GraphRepresentation &reference,
               const std::unordered_set<LayerNode *> &dropped) {
  if (dropped.empty()) {
    return reference;
  }

  std::unordered_map<std::string, LayerNode *> existing_nodes;
  for (auto &node : reference) {
    existing_nodes.emplace(node->getName(), node.get());
  }

  /// name of a dropped node -> the connection that takes its place
  std::unordered_map<std::string, Connection> replacement;

  for (auto *node : dropped) {
    auto *source = node;
    unsigned source_index = SINGLE_INOUT_IDX;

    while (dropped.find(source) != dropped.end()) {
      source_index = source->getInputConnectionIndex(SINGLE_INOUT_IDX);
      source =
        existing_nodes.at(source->getInputConnectionName(SINGLE_INOUT_IDX));
    }

    replacement.emplace(node->getName(),
                        Connection(source->getName(), source_index));
  }

  for (auto &node : reference) {
    if (dropped.find(node.get()) != dropped.end()) {
      continue;
    }

    for (unsigned i = 0, num = node->getNumInputConnections(); i < num; ++i) {
      auto found = replacement.find(node->getInputConnectionName(i));
      if (found != replacement.end()) {
        node->setInputConnectionName(i, found->second.getName());
        node->setInputConnectionIndex(i, found->second.getIndex());
      }
    }

    /// @note output connections only exist on a compiled graph, and they are
    /// keyed by the producer's output index, so the slot is kept while only the
    /// consumer it names is redirected forward past the dropped nodes
    for (unsigned i = 0, num = node->getNumOutputConnections(); i < num; ++i) {
      auto *out = node->getOutputConnection(i);
      if (out == nullptr) {
        continue;
      }

      auto name = out->getName();
      auto index = out->getIndex();
      bool redirected = false;

      while (dropped.find(existing_nodes.at(name)) != dropped.end()) {
        auto *target = existing_nodes.at(name);
        auto *next = target->getOutputConnection(SINGLE_INOUT_IDX);
        NNTR_THROW_IF(next == nullptr, std::invalid_argument)
          << "dropped node has no output connection to redirect to: " << name;
        name = next->getName();
        index = next->getIndex();
        redirected = true;
      }

      if (redirected) {
        node->setOutputConnection(i, name, index);
      }
    }
  }

  GraphRepresentation processed;
  for (auto &node : reference) {
    if (dropped.find(node.get()) == dropped.end()) {
      processed.push_back(node);
    }
  }

  return processed;
}

} // namespace nntrainer
