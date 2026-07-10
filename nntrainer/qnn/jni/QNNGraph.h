// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   QNNGraph.h
 * @date   10 Jan 2025
 * @brief  This is QNN Graph Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNTR_QNNGRAPH_H__
#define __NNTR_QNNGRAPH_H__

#include <iostream>
#include <layer_impl.h>
#include <qnn_context_var.h>
#include <qnn_properties.h>
#include <qnn_rpc_manager.h>

namespace nntrainer {

/** @brief QNN Graph layer that wraps Qualcomm Neural Network graph execution.
 */
class QNNGraph : public LayerImpl {
public:
  /** @brief Variant over the raw tensor buffer pointer types QNNGraph can
   * bind as QNN graph input/output. */
  using BufferTypePtr =
    std::variant<std::monostate, uint8_t *, uint16_t *, float *>;

  /**
   * @brief     Constructor of QNNGraph
   */
  QNNGraph();

  /**
   * @brief     Destructor of QNNGraph. Frees the QNN graph context and, if
   * this was the last user, the cached context entry for its binary path.
   */
  ~QNNGraph();

  inline static const std::string type = "qnn_graph";

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return QNNGraph::type; };

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override{};

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief Deserialize (or reuse the cached) QNN context binary for this
   * graph's bin_path.
   */
  StatusCode makeContext(RunLayerContext &context);

  /**
   * @brief Free the QNN graph context created by makeContext().
   */
  StatusCode freeContext(RunLayerContext &context);

  /**
   * @copydoc Layer::read()
   */
  void read(std::ifstream &file, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType defineWeightDataType, bool fsu = false,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

  /**
   * @brief Append a BufferTypePtr for T to buffers, tagged by T's data type
   * (UINT4/UINT8, UINT16, or FP32) so it can be bound to a QNN tensor.
   */
  void updateBufferType(std::vector<BufferTypePtr> &buffers, Tensor &T);

  /**
   * @brief Populate and register QNN tensor T with the given buffer,
   * dispatching on the buffer's held pointer type.
   */
  void populateTensor(std::shared_ptr<QNNVar> qc_var,
                      Qnn_Context_Graph_t &context_i, BufferTypePtr buffer,
                      Qnn_Tensor_t *T);

private:
  std::tuple<std::vector<props::TensorDimension>,
             std::vector<props::TensorDataType>, std::vector<props::TensorType>,
             props::FilePath, std::vector<props::InputQuantParam>,
             std::vector<props::OutputQuantParam>>
    graph_props;

  std::vector<unsigned int> weight_idx;
  std::vector<unsigned int> tensor_idx;

  Qnn_ContextHandle_t m_context = nullptr;
  std::string bin_path;
  bool m_isContextCreated;

  iotensor::InputDataType m_inputDataType;

  std::vector<props::TensorDataType> t_dtype;
  std::vector<TensorDim> t_dims;
  std::vector<props::TensorType> t_type;

  std::vector<BufferTypePtr> currentInputBuffers;
  std::vector<BufferTypePtr> currentOutputBuffers;

  sample_app::QnnFunctionPointers m_qnnFunctionPointers;

  int counter;

  // One-shot guard: log how long the very first forward waits to acquire the
  // QNN context (i.e. how much of the background preload it had to block on).
  // ~0 ms means the async preload fully hid the deserialization.
  bool context_wait_logged_ = false;
};

} // namespace nntrainer

#endif
