# arrange an instance segmentation model for test
# from sahi.utils.yolov8 import (
#     download_yolov8s_model,
# )

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

# will be used for detectron2 fasterrcnn model zoo name
# from sahi.utils.detectron2 import Detectron2TestConstants
#
# # import required functions, classes
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction, predict, get_prediction
# from sahi.utils.file import download_from_url
# from sahi.utils.cv import read_image
# from IPython.display import Image

# yolo
yolov8_model_path = "rim3.pt"

model_type = "yolov8"
model_path = yolov8_model_path
model_device = "cuda:0" # or 'cuda:0'
model_confidence_threshold = 0.2

slice_height = 1028
slice_width = 1028
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "Filtered/CircleSmooth"
# source_image_dir = "Filtered/Sobel2"

predict(
    model_type=model_type,
    model_path=model_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)


# detectron2
# model_type = "detectron2"
# model_path = "detectron2.pth"
# model_config_path = model_path
# model_device = "cuda:0" # or 'cuda:0'
# model_confidence_threshold = 0.5
#
# slice_height = 480
# slice_width = 480
# overlap_height_ratio = 0.2
# overlap_width_ratio = 0.2
#
# source_image_dir = "NG"
#
# predict(
#     model_type=model_type,
#     model_path=model_path,
#     model_config_path=model_path,
#     model_device=model_device,
#     model_confidence_threshold=model_confidence_threshold,
#     source=source_image_dir,
#     slice_height=slice_height,
#     slice_width=slice_width,
#     overlap_height_ratio=overlap_height_ratio,
#     overlap_width_ratio=overlap_width_ratio,
# )