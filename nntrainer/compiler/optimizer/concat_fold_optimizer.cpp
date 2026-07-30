// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file concat_fold_optimizer.cpp
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer which folds nested concat layers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <concat_fold_optimizer.h>

#include <concat_layer.h>
#include <connection.h>
#include <layer_node.h>
#include <node_exporter.h>
#include <optimizer_utils.h>

#include <nntrainer_log.h>

#include <functional>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace nntrainer {

/**
 * @brief read a property off a layer as it would be serialized
 * @note LayerNode::getProperty() only reaches the node's own properties and
 * ConcatLayer does not override Layer::getProperty(), so the layer's own
 * properties have to be read back through the exporter. An unset property is
 * skipped while exporting, which is exactly how an implicit axis is detected.
 *
 * @param node node to read from
 * @param key property key to look for
 * @return std::optional<std::string> value, or nullopt when unset
 */
static std::optional<std::string> getExportedProperty(const LayerNode &node,
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

GraphRepresentation
ConcatFoldOptimizer::optimize(const GraphRepresentation &reference) {
  std::unordered_map<std::string, LayerNode *> existing_nodes;
  for (auto &node : reference) {
    existing_nodes.emplace(node->getName(), node.get());
  }

  auto consumer_count = countConsumers(reference);

  /// explicit axis of each concat, absent when the layer leaves it implicit
  std::unordered_map<LayerNode *, std::string> concat_axis;
  for (auto &node : reference) {
    if (!istrequal(node->getType(), ConcatLayer::type)) {
      continue;
    }
    auto axis = getExportedProperty(*node, props::ConcatDimension::key);
    if (axis.has_value()) {
      concat_axis.emplace(node.get(), axis.value());
    }
  }

  /// @note an inner concat is dropped only when its single consumer is a concat
  /// sharing its explicit axis, so the outermost concat of a nest survives
  std::unordered_set<LayerNode *> dropped;
  for (auto &node : reference) {
    auto outer = concat_axis.find(node.get());
    if (outer == concat_axis.end()) {
      continue;
    }

    for (unsigned i = 0, num = node->getNumInputConnections(); i < num; ++i) {
      auto producer = existing_nodes.find(node->getInputConnectionName(i));
      if (producer == existing_nodes.end()) {
        continue;
      }

      auto inner = concat_axis.find(producer->second);
      if (inner != concat_axis.end() && inner->second == outer->second &&
          consumer_count[producer->first] == 1) {
        dropped.insert(producer->second);
      }
    }
  }

  if (dropped.empty()) {
    return reference;
  }

  /**
   * @brief expand one input connection into the connections that replace it,
   * walking through a whole nest of dropped concats
   */
  std::function<void(const Connection &, std::vector<Connection> &)> expand =
    [&](const Connection &connection, std::vector<Connection> &into) {
      auto found = existing_nodes.find(connection.getName());
      if (found == existing_nodes.end() ||
          dropped.find(found->second) == dropped.end()) {
        into.push_back(connection);
        return;
      }

      auto *inner = found->second;
      for (unsigned i = 0, num = inner->getNumInputConnections(); i < num;
           ++i) {
        expand(Connection(inner->getInputConnectionName(i),
                          inner->getInputConnectionIndex(i)),
               into);
      }
    };

  for (auto &node : reference) {
    if (dropped.find(node.get()) != dropped.end() ||
        concat_axis.find(node.get()) == concat_axis.end()) {
      continue;
    }

    std::vector<Connection> folded;
    for (unsigned i = 0, num = node->getNumInputConnections(); i < num; ++i) {
      expand(Connection(node->getInputConnectionName(i),
                        node->getInputConnectionIndex(i)),
             folded);
    }

    if (folded.size() == node->getNumInputConnections()) {
      continue;
    }

    /// @note the input connection list can only be replaced wholesale, since
    /// setInputConnectionName() addresses existing slots only. from_string()
    /// for a vector property clears it first, so this is a replace.
    std::stringstream ss;
    for (auto iter = folded.begin(); iter != folded.end(); ++iter) {
      if (iter != folded.begin()) {
        ss << ',';
      }
      ss << iter->toString();
    }
    node->setProperty({"input_layers=" + ss.str()});

    /// @note every producer of this concat now sits at a different input slot,
    /// and output connections are keyed by the consumer's slot, so they are
    /// rebuilt the way NetworkGraph::setOutputConnections() would. A producer
    /// with no output connections at all means the graph is not compiled yet,
    /// in which case there is nothing to keep in sync.
    for (unsigned i = 0, num = node->getNumInputConnections(); i < num; ++i) {
      auto producer = existing_nodes.find(node->getInputConnectionName(i));
      if (producer == existing_nodes.end() ||
          producer->second->getNumOutputConnections() == 0) {
        continue;
      }
      producer->second->setOutputConnection(node->getInputConnectionIndex(i),
                                            node->getName(), i);
    }
  }

  ml_logd("[%s] folding away %zu concat node(s) out of %zu", getType().c_str(),
          dropped.size(), reference.size());

  GraphRepresentation processed;
  for (auto &node : reference) {
    if (dropped.find(node.get()) == dropped.end()) {
      processed.push_back(node);
    }
  }

  return processed;
}

} // namespace nntrainer
