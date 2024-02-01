# Custom CNN Architecture on CIFAR-10 and MNIST dataset

- Computation Unit is Analog MAC (AMAC)
- Support configurable dynamic quantization on each layers

## Command
- python train.py --arch=mnist --dataset=mnist --quant=1
- python eval.py --arch=mnist --dataset=mnist --quant=0
