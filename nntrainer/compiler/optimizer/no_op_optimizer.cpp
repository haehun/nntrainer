// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file no_op_optimizer.cpp
 * @date 30 July 2026
 * @brief NNTrainer graph optimizer which returns the graph untouched
 * @see	https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <no_op_optimizer.h>

#include <nntrainer_log.h>

namespace nntrainer {

GraphRepresentation
NoOpOptimizer::optimize(const GraphRepresentation &reference) {
  ml_logd("[%s] passing through %zu nodes", getType().c_str(),
          reference.size());

  return reference;
}

} // namespace nntrainer
