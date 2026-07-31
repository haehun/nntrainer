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

#include <concat_fold_optimizer.h>
#include <flatten_realizer.h>
#include <graph_optimizer.h>
#include <identity_remove_optimizer.h>
#include <multiout_fold_optimizer.h>
#include <no_op_optimizer.h>
#include <realizer.h>
#include <reshape_fold_optimizer.h>
#include <split_remove_optimizer.h>

#include <network_graph.h>

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

/**
 * @brief compile then check optimize and equal
 * @note output connections are only populated by NetworkGraph::compile(), so a
 * pass that inspects getNumOutputConnections() has to be tested on a compiled
 * graph
 *
 * @param optimizer optimizer to use
 * @param input input
 * @param expected expected output
 */
static void
compileAndOptimizeAndEqual(GraphOptimizer &optimizer,
                           const std::vector<LayerRepresentation> &input,
                           const std::vector<LayerRepresentation> &expected) {
  std::vector<std::unique_ptr<GraphRealizer>> realizers;
  auto processed = optimizer.optimize(makeCompiledGraph(input, realizers));
  auto expected_graph = makeCompiledGraph(expected, realizers);
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

TEST(IdentityRemoveOptimizer, type_p) {
  IdentityRemoveOptimizer opt;
  EXPECT_EQ(opt.getType(), "remove_identity");
}

TEST(IdentityRemoveOptimizer, remove_single_identity_p) {
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc1"}},
    {"identity", {"name=id1", "input_layers=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=id1"}},
  };
  std::vector<LayerRepresentation> after = {
    {"fully_connected", {"name=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=fc1"}},
  };

  IdentityRemoveOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, before, after));
}

TEST(IdentityRemoveOptimizer, remove_consecutive_identity_p) {
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc1"}},
    {"identity", {"name=id1", "input_layers=fc1"}},
    {"identity", {"name=id2", "input_layers=id1"}},
    {"activation", {"name=ac1", "activation=relu", "input_layers=id2"}},
    {"fully_connected", {"name=fc2", "input_layers=ac1"}},
  };
  std::vector<LayerRepresentation> after = {
    {"fully_connected", {"name=fc1"}},
    {"activation", {"name=ac1", "activation=relu", "input_layers=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=ac1"}},
  };

  IdentityRemoveOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, before, after));
}

/**
 * @brief a terminal identity may be the model's designated output, so it must
 * survive the pass
 *
 */
TEST(IdentityRemoveOptimizer, keep_terminal_identity_p) {
  std::vector<LayerRepresentation> graph = {
    {"fully_connected", {"name=fc1"}},
    {"identity", {"name=id_out", "input_layers=fc1"}},
  };

  IdentityRemoveOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

/**
 * @brief an identity consumed by more than one node is not spliced by this
 * pass
 *
 */
TEST(IdentityRemoveOptimizer, keep_fanned_out_identity_p) {
  std::vector<LayerRepresentation> graph = {
    {"fully_connected", {"name=fc1"}},
    {"identity", {"name=id1", "input_layers=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=id1"}},
    {"fully_connected", {"name=fc3", "input_layers=id1"}},
    {"addition", {"name=add1", "input_layers=fc2,fc3"}},
  };

  IdentityRemoveOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

TEST(IdentityRemoveOptimizer, no_identity_is_noop_p) {
  std::vector<LayerRepresentation> graph = {
    {"fully_connected", {"name=fc1"}},
    {"activation", {"name=ac1", "activation=relu", "input_layers=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=ac1"}},
  };

  IdentityRemoveOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

TEST(IdentityRemoveOptimizer, empty_graph_p) {
  IdentityRemoveOptimizer opt;
  EXPECT_NO_THROW(optimizeAndEqual(opt, {}, {}));
}

TEST(ReshapeFoldOptimizer, type_p) {
  ReshapeFoldOptimizer opt;
  EXPECT_EQ(opt.getType(), "fold_reshape");
}

TEST(ReshapeFoldOptimizer, fold_two_reshapes_p) {
  std::vector<LayerRepresentation> before = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"reshape", {"name=rs1", "target_shape=1:4:2", "input_layers=in"}},
    {"reshape", {"name=rs2", "target_shape=1:1:8", "input_layers=rs1"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=rs2"}},
  };
  std::vector<LayerRepresentation> after = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"reshape", {"name=rs2", "target_shape=1:1:8", "input_layers=in"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=rs2"}},
  };

  ReshapeFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, before, after));
}

TEST(ReshapeFoldOptimizer, fold_reshape_run_p) {
  std::vector<LayerRepresentation> before = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"reshape", {"name=rs1", "target_shape=1:4:2", "input_layers=in"}},
    {"reshape", {"name=rs2", "target_shape=1:8:1", "input_layers=rs1"}},
    {"reshape", {"name=rs3", "target_shape=1:1:8", "input_layers=rs2"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=rs3"}},
  };
  std::vector<LayerRepresentation> after = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"reshape", {"name=rs3", "target_shape=1:1:8", "input_layers=in"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=rs3"}},
  };

  ReshapeFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, before, after));
}

TEST(ReshapeFoldOptimizer, keep_isolated_reshape_p) {
  std::vector<LayerRepresentation> graph = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"reshape", {"name=rs1", "target_shape=1:1:8", "input_layers=in"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=rs1"}},
  };

  ReshapeFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

/**
 * @brief a reshape observed by another consumer as well is not folded away
 *
 */
TEST(ReshapeFoldOptimizer, keep_fanned_out_reshape_p) {
  std::vector<LayerRepresentation> graph = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"reshape", {"name=rs1", "target_shape=1:4:2", "input_layers=in"}},
    {"reshape", {"name=rs2", "target_shape=1:1:8", "input_layers=rs1"}},
    {"fully_connected", {"name=fc1", "unit=8", "input_layers=rs1"}},
    {"addition", {"name=add1", "input_layers=rs2,fc1"}},
  };

  ReshapeFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

/**
 * @brief flatten computes its target shape from its input during finalize(),
 * so it is not interchangeable with reshape and must survive
 *
 */
TEST(ReshapeFoldOptimizer, keep_flatten_p) {
  std::vector<LayerRepresentation> graph = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"flatten", {"name=ft1", "input_layers=in"}},
    {"reshape", {"name=rs1", "target_shape=1:1:8", "input_layers=ft1"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=rs1"}},
  };

  ReshapeFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

TEST(ReshapeFoldOptimizer, empty_graph_p) {
  ReshapeFoldOptimizer opt;
  EXPECT_NO_THROW(optimizeAndEqual(opt, {}, {}));
}

TEST(ConcatFoldOptimizer, type_p) {
  ConcatFoldOptimizer opt;
  EXPECT_EQ(opt.getType(), "fold_concat");
}

TEST(ConcatFoldOptimizer, fold_nested_concat_p) {
  std::vector<LayerRepresentation> before = {
    {"input", {"name=in1", "input_shape=1:2:4"}},
    {"input", {"name=in2", "input_shape=1:2:4"}},
    {"input", {"name=in3", "input_shape=1:2:4"}},
    {"concat", {"name=ct_in", "axis=3", "input_layers=in2,in3"}},
    {"concat", {"name=ct_out", "axis=3", "input_layers=in1,ct_in"}},
  };
  std::vector<LayerRepresentation> after = {
    {"input", {"name=in1", "input_shape=1:2:4"}},
    {"input", {"name=in2", "input_shape=1:2:4"}},
    {"input", {"name=in3", "input_shape=1:2:4"}},
    {"concat", {"name=ct_out", "axis=3", "input_layers=in1,in2,in3"}},
  };

  ConcatFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, before, after));
}

/**
 * @brief the inner concat's inputs must land at its own position, not be
 * appended at the end
 *
 */
TEST(ConcatFoldOptimizer, fold_preserves_input_order_p) {
  std::vector<LayerRepresentation> before = {
    {"input", {"name=in1", "input_shape=1:2:4"}},
    {"input", {"name=in2", "input_shape=1:2:4"}},
    {"input", {"name=in3", "input_shape=1:2:4"}},
    {"input", {"name=in4", "input_shape=1:2:4"}},
    {"concat", {"name=ct_in", "axis=3", "input_layers=in2,in3"}},
    {"concat", {"name=ct_out", "axis=3", "input_layers=in1,ct_in,in4"}},
  };
  std::vector<LayerRepresentation> after = {
    {"input", {"name=in1", "input_shape=1:2:4"}},
    {"input", {"name=in2", "input_shape=1:2:4"}},
    {"input", {"name=in3", "input_shape=1:2:4"}},
    {"input", {"name=in4", "input_shape=1:2:4"}},
    {"concat", {"name=ct_out", "axis=3", "input_layers=in1,in2,in3,in4"}},
  };

  ConcatFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, before, after));
}

TEST(ConcatFoldOptimizer, fold_concat_nest_p) {
  std::vector<LayerRepresentation> before = {
    {"input", {"name=in1", "input_shape=1:2:4"}},
    {"input", {"name=in2", "input_shape=1:2:4"}},
    {"input", {"name=in3", "input_shape=1:2:4"}},
    {"concat", {"name=ct_a", "axis=3", "input_layers=in2,in3"}},
    {"concat", {"name=ct_b", "axis=3", "input_layers=ct_a"}},
    {"concat", {"name=ct_c", "axis=3", "input_layers=in1,ct_b"}},
  };
  std::vector<LayerRepresentation> after = {
    {"input", {"name=in1", "input_shape=1:2:4"}},
    {"input", {"name=in2", "input_shape=1:2:4"}},
    {"input", {"name=in3", "input_shape=1:2:4"}},
    {"concat", {"name=ct_c", "axis=3", "input_layers=in1,in2,in3"}},
  };

  ConcatFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, before, after));
}

/**
 * @brief concatenating along different axes is not associative
 *
 */
TEST(ConcatFoldOptimizer, keep_mismatched_axis_p) {
  std::vector<LayerRepresentation> graph = {
    {"input", {"name=in1", "input_shape=2:2:4"}},
    {"input", {"name=in2", "input_shape=2:2:4"}},
    {"input", {"name=in3", "input_shape=2:2:4"}},
    {"concat", {"name=ct_in", "axis=1", "input_layers=in2,in3"}},
    {"concat", {"name=ct_out", "axis=3", "input_layers=in1,ct_in"}},
  };

  ConcatFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

/**
 * @brief an unset axis resolves from the input dimensions during finalize(),
 * so it cannot be compared here and the nest is left alone
 *
 */
TEST(ConcatFoldOptimizer, keep_implicit_axis_p) {
  std::vector<LayerRepresentation> graph = {
    {"input", {"name=in1", "input_shape=1:2:4"}},
    {"input", {"name=in2", "input_shape=1:2:4"}},
    {"input", {"name=in3", "input_shape=1:2:4"}},
    {"concat", {"name=ct_in", "input_layers=in2,in3"}},
    {"concat", {"name=ct_out", "input_layers=in1,ct_in"}},
  };

  ConcatFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

/**
 * @brief an inner concat observed by another consumer as well is not folded
 *
 */
TEST(ConcatFoldOptimizer, keep_fanned_out_concat_p) {
  std::vector<LayerRepresentation> graph = {
    {"input", {"name=in1", "input_shape=1:2:4"}},
    {"input", {"name=in2", "input_shape=1:2:4"}},
    {"input", {"name=in3", "input_shape=1:2:4"}},
    {"concat", {"name=ct_in", "axis=3", "input_layers=in2,in3"}},
    {"concat", {"name=ct_out", "axis=3", "input_layers=in1,ct_in"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=ct_in"}},
  };

  ConcatFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

TEST(ConcatFoldOptimizer, no_nesting_is_noop_p) {
  std::vector<LayerRepresentation> graph = {
    {"input", {"name=in1", "input_shape=1:2:4"}},
    {"input", {"name=in2", "input_shape=1:2:4"}},
    {"concat", {"name=ct1", "axis=3", "input_layers=in1,in2"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=ct1"}},
  };

  ConcatFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

TEST(ConcatFoldOptimizer, empty_graph_p) {
  ConcatFoldOptimizer opt;
  EXPECT_NO_THROW(optimizeAndEqual(opt, {}, {}));
}

/**
 * @brief graphEqual only compares exported properties, which do not carry
 * output connections, so the slot renumbering is asserted directly here
 *
 */
TEST(ConcatFoldOptimizer, fold_rebuilds_output_connections_p) {
  std::vector<std::unique_ptr<GraphRealizer>> realizers;
  auto graph = makeCompiledGraph(
    {
      {"input", {"name=in1", "input_shape=1:2:4"}},
      {"input", {"name=in2", "input_shape=1:2:4"}},
      {"input", {"name=in3", "input_shape=1:2:4"}},
      {"concat", {"name=ct_in", "axis=3", "input_layers=in2,in3"}},
      {"concat", {"name=ct_out", "axis=3", "input_layers=in1,ct_in"}},
    },
    realizers);

  ConcatFoldOptimizer opt;
  auto processed = opt.optimize(graph);

  auto find = [&processed](const std::string &name) -> LayerNode * {
    for (auto &node : processed) {
      if (node->getName() == name) {
        return node.get();
      }
    }
    return nullptr;
  };

  auto *concat = find("ct_out");
  ASSERT_NE(concat, nullptr);
  ASSERT_EQ(concat->getNumInputConnections(), 3u);
  EXPECT_EQ(find("ct_in"), nullptr);

  /// each producer must name the surviving concat, at the slot it now occupies
  const char *producers[] = {"in1", "in2", "in3"};
  for (unsigned i = 0; i < 3; ++i) {
    EXPECT_EQ(concat->getInputConnectionName(i), producers[i]);

    auto *producer = find(producers[i]);
    ASSERT_NE(producer, nullptr);
    ASSERT_EQ(producer->getNumOutputConnections(), 1u);

    auto *out = producer->getOutputConnection(0);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->getName(), "ct_out");
    EXPECT_EQ(out->getIndex(), i);
  }
}

/**
 * @brief folding an uncompiled graph must leave something NetworkGraph can
 * still compile, which also covers the branch where no output connections
 * exist yet
 *
 */
TEST(ConcatFoldOptimizer, folded_graph_is_compilable_p) {
  auto graph = makeGraph({
    {"input", {"name=in1", "input_shape=1:2:4"}},
    {"input", {"name=in2", "input_shape=1:2:4"}},
    {"input", {"name=in3", "input_shape=1:2:4"}},
    {"concat", {"name=ct_in", "axis=3", "input_layers=in2,in3"}},
    {"concat", {"name=ct_out", "axis=3", "input_layers=in1,ct_in"}},
  });

  ConcatFoldOptimizer opt;
  auto processed = opt.optimize(graph);
  EXPECT_EQ(processed.size(), 4u);

  auto model_graph = NetworkGraph();
  for (auto &node : processed) {
    model_graph.addLayer(node);
  }
  EXPECT_EQ(model_graph.compile(""), ML_ERROR_NONE);
}

TEST(SplitRemoveOptimizer, type_p) {
  SplitRemoveOptimizer opt;
  EXPECT_EQ(opt.getType(), "remove_split");
}

TEST(SplitRemoveOptimizer, remove_identity_split_p) {
  std::vector<LayerRepresentation> before = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"split", {"name=sp", "split_number=1", "input_layers=in"}},
    {"fully_connected", {"name=fc", "unit=2", "input_layers=sp"}},
  };
  std::vector<LayerRepresentation> after = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
  };

  SplitRemoveOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, before, after));
}

TEST(SplitRemoveOptimizer, keep_multi_split_p) {
  std::vector<LayerRepresentation> graph = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"split", {"name=sp", "split_number=2", "input_layers=in"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=sp(0)"}},
    {"fully_connected", {"name=fc2", "unit=2", "input_layers=sp(1)"}},
  };

  SplitRemoveOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

TEST(SplitRemoveOptimizer, empty_graph_p) {
  SplitRemoveOptimizer opt;
  EXPECT_NO_THROW(optimizeAndEqual(opt, {}, {}));
}

TEST(MultioutFoldOptimizer, type_p) {
  MultioutFoldOptimizer opt;
  EXPECT_EQ(opt.getType(), "fold_multiout");
}

TEST(MultioutFoldOptimizer, fold_single_consumer_multiout_p) {
  std::vector<LayerRepresentation> before = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"multiout", {"name=mo", "input_layers=in"}},
    {"fully_connected", {"name=fc", "unit=2", "input_layers=mo"}},
  };
  std::vector<LayerRepresentation> after = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"fully_connected", {"name=fc", "unit=2", "input_layers=in"}},
  };

  MultioutFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, before, after));
}

/**
 * @brief a multiout that genuinely fans out is what the layer exists for
 *
 */
TEST(MultioutFoldOptimizer, keep_fanned_out_multiout_p) {
  std::vector<LayerRepresentation> graph = {
    {"input", {"name=in", "input_shape=1:2:4"}},
    {"multiout", {"name=mo", "input_layers=in"}},
    {"fully_connected", {"name=fc1", "unit=2", "input_layers=mo(0)"}},
    {"fully_connected", {"name=fc2", "unit=2", "input_layers=mo(1)"}},
  };

  MultioutFoldOptimizer opt;
  EXPECT_NO_THROW(compileAndOptimizeAndEqual(opt, graph, graph));
}

TEST(MultioutFoldOptimizer, empty_graph_p) {
  MultioutFoldOptimizer opt;
  EXPECT_NO_THROW(optimizeAndEqual(opt, {}, {}));
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
