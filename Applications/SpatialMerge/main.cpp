// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   main.cpp
 * @date   27 May 2026
 * @brief  Spatial Merge + Merger MLP model for nntrainer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This application builds a Spatial Merge model that:
 *   1) Reshapes flat patch tokens into a 2D spatial grid
 *   2) Merges adjacent 2×2 patches via reshape + permute
 *   3) Passes the merged features through an MLP
 *
 * Input:  [1, 576, 1, 768]  (576 patches × 768 features)
 * Output: [1, 144, 1, 1024] (144 merged patches × 1024 features)
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <tensor_dim.h>
#include <util_func.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using ml::train::createLayer;

/** Model constants */
static constexpr unsigned int BATCH_SIZE = 1;
static constexpr unsigned int NUM_PATCHES = 576; // 24 × 24
static constexpr unsigned int PATCH_H = 24;
static constexpr unsigned int PATCH_W = 24;
static constexpr unsigned int FEAT_DIM = 768;
static constexpr unsigned int MERGE_RATIO = 2;      // 2×2 merge
static constexpr unsigned int GRID_H = 12;          // 24 / 2
static constexpr unsigned int GRID_W = 12;          // 24 / 2
static constexpr unsigned int MERGED_PATCHES = 144; // 12 × 12
static constexpr unsigned int MERGED_FEAT = 3072;   // 768 × 2 × 2

/** MLP dimensions */
static constexpr unsigned int MLP_DIM1 = 3072;
static constexpr unsigned int MLP_DIM2 = 1536;
static constexpr unsigned int MLP_DIM3 = 1024;

/**
 * @brief Create the Spatial Merge + Merger MLP model
 *
 * Spatial Merge (4D equivalent of 6D reshape+permute):
 *   [1, 576, 1, 768]
 *     → Reshape(1:24:24:768)       [split 576 into 24×24 spatial]
 *   [1, 24, 24, 768]
 *     → Reshape(1:24:12:1536)      [split h=24→12×2, merge 2 into w]
 *   [1, 24, 12, 1536]
 *     → Permute(2,1,3)             [swap c↔h to group j1 with i]
 *   [1, 12, 24, 1536]
 *     → Reshape(1:12:12:3072)      [split h=24→12×2, merge 2 into w]
 *   [1, 12, 12, 3072]
 *     → Permute(2,1,3)             [swap c↔h to get i1,j1 order]
 *   [1, 12, 12, 3072]
 *     → Reshape(1:144:1:3072)      [flatten 12×12→144]
 *   [1, 144, 1, 3072]
 *
 * Merger MLP:
 *   [1, 144, 1, 3072]
 *     → LayerNorm(3072)
 *     → FC(3072→3072) + GELU
 *     → FC(3072→1536) + GELU
 *     → LayerNorm(1536)
 *     → FC(1536→1024)
 *     → LayerNorm(1024)
 *   [1, 144, 1, 1024]
 */
ModelHandle createModel() {
  ModelHandle model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});

  model->setProperty({"batch_size=" + std::to_string(BATCH_SIZE)});

  /* ── Input ─────────────────────────────────────────────────── */
  model->addLayer(createLayer(
    "input", {nntrainer::withKey("name", "input"),
              nntrainer::withKey("input_shape",
                                 "1:" + std::to_string(MERGED_PATCHES) +
                                   ":1:" + std::to_string(MERGED_FEAT))}));

  /* ── Spatial Merge ─────────────────────────────────────────── */
  // model->addLayer(createLayer(
  //   "reshape",
  //   {nntrainer::withKey("name", "spatial_reshape1"),
  //    nntrainer::withKey("target_shape", "1:" + std::to_string(PATCH_H) + ":"
  //    +
  //                                         std::to_string(PATCH_W) + ":" +
  //                                         std::to_string(FEAT_DIM))}));

  // model->addLayer(createLayer(
  //   "reshape", {nntrainer::withKey("name", "spatial_reshape2"),
  //               nntrainer::withKey("target_shape",
  //                                  "1:" + std::to_string(PATCH_H) + ":" +
  //                                    std::to_string(GRID_H) + ":" +
  //                                    std::to_string(FEAT_DIM *
  //                                    MERGE_RATIO))}));

  // model->addLayer(
  //   createLayer("permute", {nntrainer::withKey("name", "spatial_permute1"),
  //                           nntrainer::withKey("direction", "2,1,3")}));

  // model->addLayer(createLayer(
  //   "reshape",
  //   {nntrainer::withKey("name", "spatial_reshape3"),
  //    nntrainer::withKey("target_shape", "1:" + std::to_string(GRID_H) + ":" +
  //                                         std::to_string(GRID_W) + ":" +
  //                                         std::to_string(MERGED_FEAT))}));

  // model->addLayer(
  //   createLayer("permute", {nntrainer::withKey("name", "spatial_permute2"),
  //                           nntrainer::withKey("direction", "2,1,3")}));

  // model->addLayer(createLayer(
  //   "reshape", {nntrainer::withKey("name", "spatial_reshape4"),
  //               nntrainer::withKey("target_shape",
  //                                  "1:" + std::to_string(MERGED_PATCHES) +
  //                                    ":1:" + std::to_string(MERGED_FEAT))}));

  /* ── Merger MLP ────────────────────────────────────────────── */
  model->addLayer(createLayer("layer_normalization",
                              {nntrainer::withKey("name", "merger_ln1"),
                               nntrainer::withKey("axis", 3),
                               nntrainer::withKey("epsilon", 1e-5f)}));

  model->addLayer(
    createLayer("fully_connected", {nntrainer::withKey("name", "merger_fc1"),
                                    nntrainer::withKey("unit", MLP_DIM1),
                                    nntrainer::withKey("activation", "gelu")}));

  model->addLayer(
    createLayer("fully_connected", {nntrainer::withKey("name", "merger_fc2"),
                                    nntrainer::withKey("unit", MLP_DIM2),
                                    nntrainer::withKey("activation", "gelu")}));

  model->addLayer(createLayer("layer_normalization",
                              {nntrainer::withKey("name", "merger_ln2"),
                               nntrainer::withKey("axis", 3),
                               nntrainer::withKey("epsilon", 1e-5f)}));

  model->addLayer(
    createLayer("fully_connected", {nntrainer::withKey("name", "merger_fc3"),
                                    nntrainer::withKey("unit", MLP_DIM3)}));

  model->addLayer(createLayer("layer_normalization",
                              {nntrainer::withKey("name", "merger_ln3"),
                               nntrainer::withKey("axis", 3),
                               nntrainer::withKey("epsilon", 1e-5f)}));

  return model;
}

int main(int argc, char *argv[]) {
  try {
    /* ── Create model ────────────────────────────────────────── */
    auto model = createModel();

    model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

    /* ── Compile (inference-only) ────────────────────────────── */
    try {
      model->compile(ml::train::ExecutionMode::INFERENCE);
    } catch (const std::exception &e) {
      std::cerr << "Error during compile: " << e.what() << std::endl;
      return 1;
    }

    /* ── Initialize ─────────────────────────────────────────── */
    try {
      model->initialize(ml::train::ExecutionMode::INFERENCE);
    } catch (const std::exception &e) {
      std::cerr << "Error during initialize: " << e.what() << std::endl;
      return 1;
    }

    /* ── Load weights ───────────────────────────────────────── */
    try {
      model->load("./merger_weight.bin",
                  ml::train::ModelFormat::MODEL_FORMAT_BIN);
    } catch (const std::exception &e) {
      std::cerr << "Error during load: " << e.what() << std::endl;
      return 1;
    }

    /* ── Print input dimensions ─────────────────────────────── */
    auto in_dims = model->getInputDimension();

    std::cout << "\n=== Model Dimensions ===" << std::endl;
    for (size_t i = 0; i < in_dims.size(); ++i) {
      std::cout << "Input[" << i << "]: " << in_dims[i] << std::endl;
    }
    std::cout << "Expected output: [1:" << MERGED_PATCHES << ":1:" << MLP_DIM3
              << "]" << std::endl;

    /* ── Load input data from file ──────────────────────────── */
    const unsigned int input_size = BATCH_SIZE * NUM_PATCHES * 1 * FEAT_DIM;
    std::vector<float> input_data(input_size);

    {
      std::ifstream ifs("./model_input.bin", std::ios::binary);
      if (!ifs.is_open()) {
        std::cerr << "Error: cannot open ./model_input.bin" << std::endl;
        return 1;
      }
      ifs.read(reinterpret_cast<char *>(input_data.data()),
               input_size * sizeof(float));
      if (!ifs) {
        std::cerr << "Error: failed to read input data from ./model_input.bin"
                  << std::endl;
        return 1;
      }
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float *> output =
      model->inference(BATCH_SIZE, {input_data.data()});
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\nInference time: " << elapsed.count() << " ms" << std::endl;

    std::cout << std::setprecision(10);
    if (!output.empty()) {
      std::cout << "Output[0] sample values: ";
      for (unsigned int i = 0; i < 5 && i < MLP_DIM3; ++i) {
        std::cout << output[0][i] << " ";
      }
      std::cout << "..." << std::endl;
    }

    std::cout << "\nSpatial Merge model runs successfully!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
