optimizer_sources = [
  'optimizer_utils.cpp',
  'no_op_optimizer.cpp',
  'identity_remove_optimizer.cpp',
  'reshape_fold_optimizer.cpp',
  'concat_fold_optimizer.cpp',
  'split_remove_optimizer.cpp',
  'multiout_fold_optimizer.cpp',
  'permute_fold_optimizer.cpp',
]

foreach s : optimizer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach
