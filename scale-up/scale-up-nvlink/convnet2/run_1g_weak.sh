#1-GPU
export CUDA_VISIBLE_DEVICES=0
python convnet.py --data-provider cifar --test-range 5 --train-range 1-4 --data-path CIFAR-10/cifar-10-py-colmajor/ --inner-size 24 --save-path ./tmp --epochs 50 --gpu 0 --layer-def layers/cifar10-1gpu-weak.cfg --layer-params layers/layer-params-cifar10-11pct-1gpu.cfg

