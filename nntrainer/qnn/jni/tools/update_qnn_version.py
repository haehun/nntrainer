import os
import re
import sys
import shutil

VENDOR_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vendor'))
SAMPLEAPP_SRC_PREFIX = 'examples/QNN/SampleApp/SampleApp/src'
GENIE_SRC_PREFIX = 'examples/Genie/Genie/src'

def is_from_qnn(file_path):
	if not file_path.endswith(('.cpp', '.hpp', '.h')):
		return False
	with open(file_path, 'r') as f:
		return 'Qualcomm Technologies, Inc.' in f.read()

def recursive_overwrite(src, dest, ignore=None):
	if os.path.isdir(src):
		if not os.path.isdir(dest):
			os.makedirs(dest)
		files = os.listdir(src)
		ignored = ignore(src, files) if ignore else set()
		for f in files:
			if f not in ignored:
				recursive_overwrite(os.path.join(src, f), os.path.join(dest, f), ignore)
	else:
		shutil.copyfile(src, dest)

def find_replace_in_file(file_path, old_text, new_text):
	with open(file_path, 'r') as f:
		file_data = f.read()
	file_data = file_data.replace(old_text, new_text)
	with open(file_path, 'w') as f:
		f.write(file_data)

def remove_path(path):
	if os.path.isdir(path):
		shutil.rmtree(path)
	elif os.path.exists(path):
		os.remove(path)

# ── 1. Copy from SDK ──────────────────────────────────────────────────────

# Maps target dirs (under VENDOR_DIR) to source dirs (under QNN_SDK_ROOT)
target_src_dirs = {
	'Log':            os.path.join(SAMPLEAPP_SRC_PREFIX, 'Log'),
	'PAL':            os.path.join(SAMPLEAPP_SRC_PREFIX, 'PAL'),
	'QNN':            'include/QNN',
	'Utils':          os.path.join(SAMPLEAPP_SRC_PREFIX, 'Utils'),
	'WrapperUtils':   os.path.join(SAMPLEAPP_SRC_PREFIX, 'WrapperUtils'),
	'qnn-api':        os.path.join(GENIE_SRC_PREFIX, 'qualla/engines/qnn-api'),
}

# Additional individual files to copy: (src_rel_path, target_rel_path)
extra_files = [
	(os.path.join(SAMPLEAPP_SRC_PREFIX, 'SampleApp.hpp'),
	 os.path.join(VENDOR_DIR, 'QNN.hpp')),
	(os.path.join(SAMPLEAPP_SRC_PREFIX, 'QnnTypeMacros.hpp'),
	 os.path.join(VENDOR_DIR, 'QnnTypeMacros.hpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'resource-manager/include/ResourceManager.hpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/ResourceManager.hpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'resource-manager/src/ResourceManager.cpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/ResourceManager.cpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'qualla/include/qualla/detail/Log.hpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/qualla/detail/Log.hpp')),
	(os.path.join(GENIE_SRC_PREFIX, 'qualla/include/qualla/detail/dlOpenWrapper.hpp'),
	 os.path.join(VENDOR_DIR, 'qnn-api/qualla/detail/dlOpenWrapper.hpp')),
]

if __name__ == '__main__':
	# Resolve SDK root: --qnn-sdk-root=PATH flag takes priority, then env var.
	qnn_root = None
	for arg in sys.argv[1:]:
		if arg.startswith('--qnn-sdk-root='):
			qnn_root = arg[len('--qnn-sdk-root='):]
			break
	if not qnn_root:
		qnn_root = os.environ.get('QNN_SDK_ROOT', '')
	if not qnn_root:
		sys.exit(
			'Set --qnn-sdk-root=<path> or the QNN_SDK_ROOT env variable to your '
			'Qualcomm QNN SDK root (e.g. .../qairt/2.47.0.x)'
		)

	# Copy directories
	for target_name, src_rel in target_src_dirs.items():
		src_path = os.path.join(qnn_root, src_rel)
		target_path = os.path.join(VENDOR_DIR, target_name)
		if not os.path.exists(src_path):
			print(f'WARNING: {src_rel} does not exist in SDK, skipping')
			continue
		# Remove old Qualcomm files from target
		for root, dirs, files in os.walk(target_path):
			for f in files:
				cur = os.path.join(root, f)
				if is_from_qnn(cur):
					os.remove(cur)
		recursive_overwrite(src_path, target_path)

	# Copy extra files
	for src_rel, target_rel in extra_files:
		src_path = os.path.join(qnn_root, src_rel)
		if os.path.exists(src_path):
			os.makedirs(os.path.dirname(target_rel), exist_ok=True)
			shutil.copyfile(src_path, target_rel)

	# ── 1b. Pin QNN API version ───────────────────────────────────────────
	_qnn_common_h = os.path.join(VENDOR_DIR, 'QNN/QnnCommon.h')
	if not os.path.exists(_qnn_common_h):
		sys.exit(f'Expected {_qnn_common_h} to exist after SDK copy — check SDK layout.')
	# Known-good QNN API minor versions: 36 = SDK 2.47.x, 38 = SDK 2.49.x.
	# The vendor tree layout is identical across these; only the API headers move.
	_supported_minors = {36: '2.47.x', 38: '2.49.x'}
	_found_minor = None
	with open(_qnn_common_h, 'r') as _fh:
		for _line in _fh:
			_m = re.match(r'\s*#define\s+QNN_API_VERSION_MINOR\s+(\d+)', _line)
			if _m:
				_found_minor = int(_m.group(1))
				break
	if _found_minor is None:
		sys.exit(f'{_qnn_common_h} does not define QNN_API_VERSION_MINOR — unexpected SDK layout.')
	if _found_minor not in _supported_minors:
		sys.exit(
			f'Unsupported QNN API version: found MINOR={_found_minor}, '
			f'supported are {sorted(_supported_minors)} '
			f'({", ".join(f"{k}=SDK {v}" for k, v in sorted(_supported_minors.items()))}). '
			f'Add the version to _supported_minors in this script if you intend to bump the SDK.'
		)
	print(f'QNN API MINOR={_found_minor} (SDK {_supported_minors[_found_minor]})')

	# ── 2. Remove unused files/directories ────────────────────────────────

	# Unused QNN backend subdirs (we only use HTP and System)
	for d in ['CPU', 'DSP', 'GPU', 'GPU.unused', 'HTA', 'Saver', 'IR',
	          'LPAI', 'TFLiteDelegate', 'GenAiTransformer',
	          'LoraAdapterBinUpdater', 'HTPQEMU']:
		remove_path(os.path.join(VENDOR_DIR, 'QNN', d))

	# HTP/core is internal headers not needed for our build
	remove_path(os.path.join(VENDOR_DIR, 'QNN', 'HTP', 'core'))

	# PAL windows (not needed for Android)
	remove_path(os.path.join(VENDOR_DIR, 'PAL/src/windows'))

	# Unused files from qnn-api with unresolvable dependencies
	for f in ['ClientBuffer.hpp', 'ClientBuffer.cpp',
	          'DmaBufAllocator.hpp', 'DmaBufAllocator.cpp', 'IBufferAlloc.hpp',
	          'IOTensor.hpp', 'IOTensor.cpp', 'qnn-utils.hpp', 'qnn-utils.cpp',
	          'QnnApi.hpp', 'QnnApi.cpp', 'QnnApiUtils.hpp', 'QnnApiUtils.cpp',
	          'RpcMem.hpp', 'RpcMem.cpp', 'QnnWrapperUtils.hpp',
	          'BufferUtils.hpp', 'BufferUtils.cpp',
	          'QnnTypeUtils.hpp', 'QnnTypeUtils.cpp',
	          'QnnTypeMacros.hpp']:
		remove_path(os.path.join(VENDOR_DIR, 'qnn-api', f))

	# Unused qnn-api subdirs with unresolvable dependencies
	for d in ['buffer', 'config', 'PAL']:
		remove_path(os.path.join(VENDOR_DIR, 'qnn-api', d))

	# Orphaned qualla/ at vendor root (duplicate of qnn-api/qualla/)
	remove_path(os.path.join(VENDOR_DIR, 'qualla'))

	# ── 3. Apply compatibility patches ─────────────────────────────────────

	# SDK uses SampleApp.hpp, our build uses QNN.hpp
	find_replace_in_file(os.path.join(VENDOR_DIR, 'Utils/DynamicLoadUtil.hpp'),
		'SampleApp.hpp', 'QNN.hpp')
	find_replace_in_file(os.path.join(VENDOR_DIR, 'Utils/QnnSampleAppUtils.hpp'),
		'SampleApp.hpp', 'QNN.hpp')

	# IOTensor.hpp: make members public for our usage
	find_replace_in_file(os.path.join(VENDOR_DIR, 'Utils/IOTensor.hpp'),
		'private', 'public')

	# dlOpenWrapper.hpp: fix const-correctness
	dl_open_wrapper = os.path.join(VENDOR_DIR, 'qnn-api/qualla/detail/dlOpenWrapper.hpp')
	find_replace_in_file(dl_open_wrapper, 'static const int s_anchor', 'static int s_anchor')
	find_replace_in_file(dl_open_wrapper, 'reinterpret_cast<const void*>', 'reinterpret_cast<void*>')

	# SDK 2.49+ adds qnn-api/IOTensor/IOTensorTypes.hpp, pulled in by IBackend.hpp.
	# It redefines StatusCode / OutputDataType / InputDataType, which Utils/IOTensor.hpp
	# already declares in the same qnn::tools::iotensor namespace, so any TU seeing
	# both fails to compile. IBackend.hpp only needs GraphInputInfo from that header,
	# so drop the duplicate enums and take them from Utils/IOTensor.hpp instead.
	_io_types = os.path.join(VENDOR_DIR, 'qnn-api/IOTensor/IOTensorTypes.hpp')
	if os.path.exists(_io_types):
		with open(_io_types, 'r') as _fh:
			_data = _fh.read()
		_dup_enums = re.compile(
			r'enum class StatusCode \{.*?\};\s*'
			r'enum class OutputDataType \{[^}]*\};\s*'
			r'enum class InputDataType \{[^}]*\};\s*',
			re.DOTALL)
		_data, _n = _dup_enums.subn('', _data)
		if _n != 1:
			sys.exit(
				f'{_io_types}: expected exactly 1 duplicate-enum block, found {_n}. '
				f'The SDK layout changed — review this patch.'
			)
		# Utils/ is on the include path; IOTensor.hpp declares the three enums.
		_data = _data.replace('#pragma once', '#pragma once\n\n#include "IOTensor.hpp"', 1)
		with open(_io_types, 'w') as _fh:
			_fh.write(_data)
		print('Patched IOTensorTypes.hpp: removed enums duplicated by Utils/IOTensor.hpp')

	print('QNN vendor files updated successfully.')
