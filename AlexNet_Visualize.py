import tvm
import os

import numpy as np
from tvm import relay
from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from tvm_walk_through.visualize import RelayVisualizer

def auto_optimize(mod, target, params, filepath):
    print("Export OPTIMIZED Graph as json format. Please Go to https://netron.app/ to plot Graph")
    mod, params = relay.optimize(mod, target=target, params=params)
    visualizer = RelayVisualizer()
    visualizer.visualize(mod, path=filepath)
    return mod, params

## Create Directory
os.makedirs("visualizes/", exist_ok=True)
######################################################################
# Load a pretrained PyTorch model
# -------------------------------
model_name = "alexnet"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

######################################################################
# Load a test image
# -----------------
# Classic cat example!
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list) ## Converting pytorch expression to Relay

print("Export UNOPTIMIZED Graph as json format. Please Go to https://netron.app/ to plot Graph")
visualizer = RelayVisualizer()
visualizer.visualize(mod, path="visualizes/AlexNet_relay_before_optimized.prototxt")

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    mod, _ = auto_optimize(mod, target, params, filepath="visualizes/AlexNet_relay_after_optimized.prototxt")
