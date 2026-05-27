app_sources = [
  'main.cpp',
]

app_dependencies = [
  nntrainer_dep,
  nntrainer_ccapi_dep,
]

e = executable('nntrainer_spatial_merge',
  app_sources,
  dependencies: app_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
