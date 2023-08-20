import torch

from midas_sources.midas.model_loader import load_model
from midas_sources.run import process as midas_process


# model = transform = net_w = net_h, device, _model_type = None

def prepare_model(model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
                  square=False):
    global model, transform, net_w, net_h, device, _model_type, _optimize
    _model_type = model_type
    _optimize = optimize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)


def process(img):
    global model, transform, net_w, net_h, device, _model_type, _optimize
    image = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        prediction = midas_process(device, model, _model_type, image, (net_w, net_h), img.shape[1::-1],
                                   _optimize, False)

    depth_min = prediction.min()
    depth_max = prediction.max()
    normalized_depth = 255 * (prediction - depth_min) / (depth_max - depth_min)
    return normalized_depth