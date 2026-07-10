// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@samsung.com>
 *
 * @file    qnn_timing.h
 * @date    10 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Haehun Yang <haehun.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Lightweight wall-clock timing helpers for QNN initialization
 *          profiling. All elapsed times are logged in milliseconds under the
 *          "QNN-INIT" tag so they can be filtered with `logcat | grep QNN-INIT`
 *          (Android) or a grep on stderr (host).
 *
 *          Two ways to use it:
 *            - QNN_TIME_SCOPE("label")      : RAII timer, logs on scope exit.
 *            - QNN_TIME_SINCE("label", tp)  : logs (now - tp); use for calls
 *                                             embedded in an if-condition where
 *                                             a scope block is awkward.
 *
 *          steady_clock is used deliberately: it is monotonic and unaffected by
 *          wall-clock adjustments, so it is the correct source for measuring
 *          durations. Define QNN_TIMING_DISABLE to compile the timers out.
 */
#ifndef __QNN_TIMING_H__
#define __QNN_TIMING_H__

#include <chrono>
#include <string>
#include <utility>

#ifdef __ANDROID__
#include <android/log.h>
#define QNN_TIMING_LOG(...)                                                    \
  __android_log_print(ANDROID_LOG_INFO, "QNN-INIT", __VA_ARGS__)
#else
#include <cstdio>
#define QNN_TIMING_LOG(fmt, ...)                                               \
  fprintf(stderr, "[QNN-INIT] " fmt "\n", ##__VA_ARGS__)
#endif

namespace nntrainer {

/**
 * @brief Milliseconds elapsed since the given time point, as a long long for
 * %lld logging.
 */
inline long long
qnn_ms_since(const std::chrono::steady_clock::time_point &since) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::steady_clock::now() - since)
    .count();
}

/**
 * @brief RAII timer: captures the current time on construction and logs the
 * elapsed duration under the "QNN-INIT" tag when it goes out of scope.
 */
class QnnScopedTimer {
public:
  explicit QnnScopedTimer(std::string label) :
    label_(std::move(label)), start_(std::chrono::steady_clock::now()) {}
  ~QnnScopedTimer() {
    QNN_TIMING_LOG("%s: %lld ms", label_.c_str(), qnn_ms_since(start_));
  }
  QnnScopedTimer(const QnnScopedTimer &) = delete;
  QnnScopedTimer &operator=(const QnnScopedTimer &) = delete;

private:
  std::string label_;
  std::chrono::steady_clock::time_point start_;
};

} // namespace nntrainer

#define QNN_TIMER_CONCAT_(a, b) a##b
#define QNN_TIMER_CONCAT(a, b) QNN_TIMER_CONCAT_(a, b)

#ifndef QNN_TIMING_DISABLE
#define QNN_TIME_SCOPE(label)                                                  \
  ::nntrainer::QnnScopedTimer QNN_TIMER_CONCAT(_qnn_scoped_timer_,             \
                                               __LINE__)(label)
#define QNN_TIME_SINCE(label, since)                                           \
  QNN_TIMING_LOG("%s: %lld ms", label, ::nntrainer::qnn_ms_since(since))
#else
#define QNN_TIME_SCOPE(label) ((void)0)
#define QNN_TIME_SINCE(label, since) ((void)0)
#endif

#endif /* __QNN_TIMING_H__ */
