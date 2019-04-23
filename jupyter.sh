docker run \
  --rm \
  -p 8888:8888 \
  -e JUPYTER_ENABLE_LAB=yes \
  -v "$PWD":/home/jovyan/work \
  jupyter/scipy-notebook:7d427e7a4dde

