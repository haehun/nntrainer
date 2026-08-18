#! /bin/bash
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
#
# @file prepare_ml-api-common.sh
# @date 10 June 2021
# @brief This file is a helper tool to ml-api-common dependency for android build
# @author Parichay Kapoor <pk.kapoor@samsung.com>
#
# usage: ./prepare_ml-api-common.sh target

set -e
TARGET=$1
# Note: zip name can be nnstreamer-native-*.zip but this file is heavier to download
FILE_PREFIX=nnstreamer-lite-native
ZIP_NAME=${FILE_PREFIX}.zip
URL="https://nnstreamer-release.s3-ap-northeast-2.amazonaws.com/nnstreamer/latest/android/"

echo "PREPARING ml_api at ${TARGET}"

[ ! -d ${TARGET} ] && mkdir -p ${TARGET}

pushd ${TARGET}

function _download_ml_api {
  [ -f $ZIP_NAME ] && echo "${ZIP_NAME} exists, skip downloading" && return 0
  echo "[ml_api] downloading ${ZIP_NAME}\n"
  if ! wget -r -l1 -nH --cut-dirs=3 ${URL}${ZIP_NAME} -O ${ZIP_NAME} --no-check-certificate ; then
    echo "[ml_api] Download failed, please check url\n"
    exit $?
  fi
  echo "[ml_api] Finish downloading ml_api\n"
}

function _extract_ml_api {
  echo "[ml_api] unzip ml_api\n"
  unzip -q ${ZIP_NAME} -d ${FILE_PREFIX}
  rm -f ${ZIP_NAME}
}

function _cleanup_ml_api {
  echo "[ml_api] cleanup ml_api \n"
  # move include to the target location
  mv -f ${FILE_PREFIX}/main/jni/nnstreamer/include .
  mv -f ${FILE_PREFIX}/main/jni/nnstreamer/lib .
  # remove all untarred directories/files
  rm -rf ${FILE_PREFIX}
  # cleanup all files other than ml_api and tizen_error
  find include ! \( -name '*.h' \) -type f -exec rm -f {} +
  find lib ! \( -name 'libnnstreamer-native.so' -or -name 'libgstreamer_android.so' \) -type f -exec rm -f {} +
}

# Try copying from a sibling nntrainer builddir first (S3 download often fails
# behind corporate proxies). The sibling builddir is at <repo_root>/../nntrainer
# relative to this repo's nntrainer, i.e. the same nntrainer/builddir but from a
# different checkout, or simply a pre-populated copy.
_COPY_SRC="$(cd "$(dirname "$0")/../.." && pwd)/../nntrainer/builddir/ml-api-inference"
if [ ! -d "${TARGET}/include" ] && [ -d "${_COPY_SRC}/include" ]; then
  echo "[ml_api] copying from ${_COPY_SRC}"
  cp -r "${_COPY_SRC}/include" "${TARGET}/include"
  cp -r "${_COPY_SRC}/lib" "${TARGET}/lib" 2>/dev/null || true
fi

# Fallback: download from S3 if still not present
[ ! -d "${TARGET}/include" ] && _download_ml_api && _extract_ml_api \
  && _cleanup_ml_api

popd

