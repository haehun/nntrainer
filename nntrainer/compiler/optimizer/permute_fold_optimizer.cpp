// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file permute_fold_optimizer.cpp
 * @date 31 July 2026
 * @brief NNTrainer graph optimizer which folds consecutive permute layers
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <permute_fold_optimizer.h>

#include <layer_node.h>
#include <optimizer_utils.h>
#include <permute_layer.h>

#include <nntrainer_log.h>

#include <array>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nntrainer {

static constexpr unsigned SINGLE_INOUT_IDX = 0;
static constexpr unsigned NUM_AXES = 3;

/// direction covers channel, height and width as one-based axis numbers
using Direction = std::array<unsigned, NUM_AXES>;

static constexpr Direction IDENTITY = {1, 2, 3};

/**
 * @brief read a permute layer's direction
 * @note PermuteLayer keeps direction as a three element array property, so the
 * exporter joins it into "c,h,w". A node whose direction was never set throws
 * while being exported; such a node is ill formed and finalize() would reject
 * it, so it is reported as unreadable and left alone here.
 *
 * @param node node to read from
 * @return std::optional<Direction> direction, or nullopt when unreadable
 */
static std::optional<Direction> readDirection(const LayerNode &node) {
  std::optional<std::string> raw;
  try {
    raw = getExportedProperty(node, props::PermuteDims::key);
  } catch (const std::exception &) {
    return std::nullopt;
  }

  if (!raw.has_value()) {
    return std::nullopt;
  }

  Direction direction{};
  std::stringstream ss(raw.value());
  std::string token;

  for (unsigned i = 0; i < NUM_AXES; ++i) {
    if (!std::getline(ss, token, ',')) {
      return std::nullopt;
    }

    try {
      direction[i] = std::stoul(token);
    } catch (const std::exception &) {
      return std::nullopt;
    }
  }

  if (std::getline(ss, token, ',')) {
    return std::nullopt;
  }

  /// reject anything that is not a permutation of the three axes
  unsigned seen = 0;
  for (auto axis : direction) {
    if (axis < 1 || axis > NUM_AXES) {
      return std::nullopt;
    }
    seen |= 1u << (axis - 1);
  }

  return seen == 0b111 ? std::optional<Direction>(direction) : std::nullopt;
}

/**
 * @brief compose two permutations
 * @note a permute maps out[p] = in[direction[p]] over one-based axes, so
 * applying @a first and then @a second gives
 * out[q] = mid[second[q]] = in[first[second[q]]]
 *
 * @param first permutation applied first
 * @param second permutation applied second
 * @return Direction the single permutation equivalent to both
 */
static Direction compose(const Direction &first, const Direction &second) {
  Direction composed{};
  for (unsigned i = 0; i < NUM_AXES; ++i) {
    composed[i] = first[second[i] - 1];
  }
  return composed;
}

/**
 * @brief render a direction the way the property parser expects it
 *
 * @param direction direction to render
 * @return std::string "c,h,w"
 */
static std::string toPropertyValue(const Direction &direction) {
  std::stringstream ss;
  ss << direction[0] << ',' << direction[1] << ',' << direction[2];
  return ss.str();
}

GraphRepresentation
PermuteFoldOptimizer::optimize(const GraphRepresentation &reference) {
  std::unordered_map<std::string, LayerNode *> existing_nodes;
  for (auto &node : reference) {
    existing_nodes.emplace(node->getName(), node.get());
  }

  auto consumer_count = countConsumers(reference);

  std::unordered_map<const LayerNode *, Direction> directions;
  for (auto &node : reference) {
    if (istrequal(node->getType(), PermuteLayer::type) &&
        node->getNumInputConnections() == 1) {
      if (auto direction = readDirection(*node); direction.has_value()) {
        directions.emplace(node.get(), direction.value());
      }
    }
  }

  if (directions.empty()) {
    return reference;
  }

  /// the only consumer of each node that has exactly one
  std::unordered_map<const LayerNode *, LayerNode *> single_consumer;
  for (auto &node : reference) {
    for (unsigned i = 0, num = node->getNumInputConnections(); i < num; ++i) {
      auto &producer_name = node->getInputConnectionName(i);
      if (consumer_count[producer_name] != 1) {
        continue;
      }
      auto producer = existing_nodes.find(producer_name);
      if (producer != existing_nodes.end()) {
        single_consumer.emplace(producer->second, node.get());
      }
    }
  }

  /**
   * @brief whether a node hands its tensor to exactly one permute that this
   * pass can absorb it into
   */
  auto links_forward = [&](const LayerNode *node) {
    if (directions.find(node) == directions.end()) {
      return false;
    }
    auto consumer = single_consumer.find(node);
    return consumer != single_consumer.end() &&
           directions.find(consumer->second) != directions.end();
  };

  std::unordered_set<LayerNode *> dropped;

  for (auto &node : reference) {
    if (directions.find(node.get()) == directions.end()) {
      continue;
    }

    /// only start from the head of a run, so each run is walked once
    auto producer =
      existing_nodes.find(node->getInputConnectionName(SINGLE_INOUT_IDX));
    if (producer != existing_nodes.end() && links_forward(producer->second)) {
      continue;
    }

    auto *current = node.get();
    auto composed = directions.at(current);

    while (links_forward(current)) {
      auto *next = single_consumer.at(current);
      composed = compose(composed, directions.at(next));
      dropped.insert(current);
      current = next;
    }

    if (current == node.get()) {
      continue;
    }

    /// @note a run composing to the identity moves no data, so the surviving
    /// node goes too when it can be spliced out like any other passthrough
    if (composed == IDENTITY && current->getNumInputConnections() == 1 &&
        consumer_count[current->getName()] == 1) {
      dropped.insert(current);
      continue;
    }

    current->setProperty(
      {std::string(props::PermuteDims::key) + "=" + toPropertyValue(composed)});
  }

  ml_logd("[%s] folding away %zu permute node(s) out of %zu", getType().c_str(),
          dropped.size(), reference.size());

  return spliceOutNodes(reference, dropped);
}

} // namespace nntrainer
