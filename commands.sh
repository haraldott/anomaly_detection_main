#!/bin/sh
# Normal learning no finetune
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=2 clip=1.0 -experiment="-hiddenunits=128 -hiddenlayers=2 clip=1.0"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=2 clip=1.1 -experiment="-hiddenunits=128 -hiddenlayers=2 clip=1.1"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=2 clip=1.2 -experiment="-hiddenunits=128 -hiddenlayers=2 clip=1.2"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=2 clip=1.3 -experiment="-hiddenunits=128 -hiddenlayers=2 clip=1.3"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=2 clip=1.4 -experiment="-hiddenunits=128 -hiddenlayers=2 clip=1.4"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=2 clip=1.5 -experiment="-hiddenunits=128 -hiddenlayers=2 clip=1.5"

CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=3 clip=1.0 -experiment="-hiddenunits=128 -hiddenlayers=3 clip=1.0"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=3 clip=1.1 -experiment="-hiddenunits=128 -hiddenlayers=3 clip=1.1"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=3 clip=1.2 -experiment="-hiddenunits=128 -hiddenlayers=3 clip=1.2"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=3 clip=1.3 -experiment="-hiddenunits=128 -hiddenlayers=3 clip=1.3"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=3 clip=1.4 -experiment="-hiddenunits=128 -hiddenlayers=3 clip=1.4"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=128 -hiddenlayers=3 clip=1.5 -experiment="-hiddenunits=128 -hiddenlayers=3 clip=1.5"

CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=2 clip=1.0 -experiment="-hiddenunits=256 -hiddenlayers=2 clip=1.0"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=2 clip=1.1 -experiment="-hiddenunits=256 -hiddenlayers=2 clip=1.1"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=2 clip=1.2 -experiment="-hiddenunits=256 -hiddenlayers=2 clip=1.2"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=2 clip=1.3 -experiment="-hiddenunits=256 -hiddenlayers=2 clip=1.3"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=2 clip=1.4 -experiment="-hiddenunits=256 -hiddenlayers=2 clip=1.4"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=2 clip=1.5 -experiment="-hiddenunits=256 -hiddenlayers=2 clip=1.5"

CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=3 clip=1.0 -experiment="-hiddenunits=256 -hiddenlayers=3 clip=1.0"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=3 clip=1.1 -experiment="-hiddenunits=256 -hiddenlayers=3 clip=1.1"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=3 clip=1.2 -experiment="-hiddenunits=256 -hiddenlayers=3 clip=1.2"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=3 clip=1.3 -experiment="-hiddenunits=256 -hiddenlayers=3 clip=1.3"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=3 clip=1.4 -experiment="-hiddenunits=256 -hiddenlayers=3 clip=1.4"
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -anomaly_type="random_lines" -hiddenunits=256 -hiddenlayers=3 clip=1.5 -experiment="-hiddenunits=256 -hiddenlayers=3 clip=1.5"