#1-GPU
#python convnet.py --data-provider cifar --test-range 5 --train-range 1-4 --data-path CIFAR-10/cifar-10-py-colmajor/ --inner-size 24 --save-path ./tmp --epochs 50 --gpu 0 --layer-def layers/layers-cifar10-11pct-1gpu.cfg --layer-params layers/layer-params-cifar10-11pct-1gpu.cfg

#2-GPU
#python convnet.py --data-provider cifar --test-range 5 --train-range 1-4 --data-path CIFAR-10/cifar-10-py-colmajor/ --inner-size 24 --save-path ./tmp --epochs 50 --gpu 0,1 --layer-def layers/layers-cifar10-11pct-2gpu.cfg --layer-params layers/layer-params-cifar10-11pct-2gpu.cfg

#4-GPU
python convnet.py --data-provider cifar --test-range 5 --train-range 1-4 --data-path CIFAR-10/cifar-10-py-colmajor/ --inner-size 24 --save-path ./tmp --epochs 50 --gpu 0,1,2,3 --layer-def layers/layers-cifar10-11pct-4gpu.cfg --layer-params layers/layer-params-cifar10-11pct-4gpu.cfg

#8-GPU
#python convnet.py --data-provider cifar --test-range 5 --train-range 1-4 --data-path CIFAR-10/cifar-10-py-colmajor/ --inner-size 24 --save-path ./tmp --gpu 0,1,2,3,4,5,6,7 --layer-def layers/layers-cifar10-11pct-8gpu.cfg --layer-params layers/layer-params-cifar10-11pct-8gpu.cfg
