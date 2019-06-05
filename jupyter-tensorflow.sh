docker run \
  --rm \
  -p 8888:8888 \
  -e JUPYTER_ENABLE_LAB=yes \
  -v "$PWD":/home/jovyan/work \
  jupyter/tensorflow-notebook:d4cbf2f80a2a

