import torch

def quant_signed_15(original, bit=5):
    bit = bit - 2
    original = original.clamp(max=1.875,min=-1.875)
    #print("Activation result quantization")
    torch.set_printoptions(precision=bit)
    return ((original * (2**bit)).int()) / (2**bit)


def quant_signed_05(original, bit=5):
    bit = bit - 1
    original = original.clamp(max=0.9375,min=-0.9375)
    torch.set_printoptions(precision=bit)
    return ((original * (2**bit)).int()) / (2**bit)


# Quant with 1 integer bit
def quant_signed_1(original, bit=6):
    bit = bit - 2
    original = original.clamp(max=1.9375,min=-1.9375)
    torch.set_printoptions(precision=bit)
    return ((original * (2**bit)).int()) / (2**bit)


# Quant with no integer bit
def quant_signed_0(original, bit=6):
    bit = bit - 1
    original = original.clamp(max=0.96875,min=-0.96875)
    torch.set_printoptions(precision=bit)
    return ((original * (2**bit)).int()) / (2**bit)

