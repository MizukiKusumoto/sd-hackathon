docker run \
--gpus all \
-it \
-p  8000:8000 \
--rm \
-v /home/common/mosaic-art-maker:/code/app/ \
--name mizuki \
mizuki:latest \
bash
# 8888 は jupyter