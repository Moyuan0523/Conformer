import sys, torch, torchvision, PIL, numpy, sklearn, timm
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("Pillow", PIL.__version__)
print("numpy", numpy.__version__, "scikit-learn", sklearn.__version__)
print("timm", timm.__version__)