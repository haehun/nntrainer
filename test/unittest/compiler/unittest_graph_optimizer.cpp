// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file unittest_graph_optimizer.cpp
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer related tests
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <flatten_realizer.h>
#include <graph_optimizer.h>
#include <no_op_optimizer.h>
#include <realizer.h>

#include <compiler_test_util.h>
#include <nntrainer_test_util.h>

using namespace nntrainer;

/**
 * @brief check optimize and equal
 *
 * @param optimizer optimizer to use
 * @param input input
 * @param expected expected output
 */
static void optimizeAndEqual(GraphOptimizer &optimizer,
                             const std::vector<LayerRepresentation> &input,
                             const std::vector<LayerRepresentation> &expected) {
  auto processed = optimizer.optimize(makeGraph(input));
  auto expected_graph = makeGraph(expected);
  graphEqual(processed, expected_graph);
}

TEST(NoOpOptimizer, type_p) {
  NoOpOptimizer opt;
  EXPECT_EQ(opt.getType(), "no_op");
}

TEST(NoOpOptimizer, empty_graph_p) {
  NoOpOptimizer opt;
  EXPECT_NO_THROW(optimizeAndEqual(opt, {}, {}));
}

TEST(NoOpOptimizer, single_node_p) {
  NoOpOptimizer opt;

  LayerRepresentation input1 = {"fully_connected", {"name=layer1", "unit=1"}};

  EXPECT_NO_THROW(optimizeAndEqual(opt, {input1}, {input1}));
}

TEST(NoOpOptimizer, multi_node_p) {
  NoOpOptimizer opt;

  std::vector<LayerRepresentation> graph = {
    {"input", {"name=in", "input_shape=1:1:2"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=in"}},
    {"fully_connected", {"name=fc2", "unit=1", "input_layers=fc1"}},
  };

  EXPECT_NO_THROW(optimizeAndEqual(opt, graph, graph));
}

/**
 * @brief the pass must hand back the very same LayerNodes, not clones
 *
 */
TEST(NoOpOptimizer, nodes_are_shared_p) {
  NoOpOptimizer opt;

  auto reference = makeGraph({
    {"input", {"name=in", "input_shape=1:1:2"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=in"}},
  });

  auto optimized = opt.optimize(reference);

  ASSERT_EQ(optimized.size(), reference.size());
  for (unsigned int i = 0; i < reference.size(); ++i) {
    EXPECT_EQ(optimized[i].get(), reference[i].get());
  }
}

/**
 * @brief usage example: lower the graph with realizers first, then run the
 * optimizer pipeline on the lowered graph
 *
 */
TEST(GraphOptimizer, pipeline_after_realize_p) {
  std::vector<std::unique_ptr<GraphRealizer>> realizers;
  realizers.emplace_back(new FlattenRealizer());

  std::vector<std::unique_ptr<GraphOptimizer>> optimizers;
  optimizers.emplace_back(new NoOpOptimizer());

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:2"}},
    {"fully_connected",
     {"name=fc1", "unit=2", "input_layers=in", "flatten=true"}},
  });

  for (auto &realizer : realizers) {
    graph = realizer->realize(graph);
  }

  auto realized_size = graph.size();

  for (auto &optimizer : optimizers) {
    graph = optimizer->optimize(graph);
  }

  /// no_op keeps whatever the realizers produced
  EXPECT_EQ(graph.size(), realized_size);
}
