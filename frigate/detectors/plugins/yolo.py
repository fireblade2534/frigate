import logging

import numpy as np
import torch
from pydantic import Field
from typing_extensions import Literal
import os

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import (
    BaseDetectorConfig,
    ModelTypeEnum,
)

torch.cuda.set_device(0)
from frigate.util.model import get_ort_providers

logger = logging.getLogger(__name__)

DETECTOR_KEY = "yolo"


class YOLODetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default="AUTO", title="Device Type")


class YOLODetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: YOLODetectorConfig):
        try:
            import ultralytics

            logger.info("YOLO: loaded ultralytics module")
        except ModuleNotFoundError:
            logger.error(
                "YOLO: module loading failed, need 'pip install ultralytics'?!?"
            )
            raise

        path = detector_config.model.path
        logger.info(f"YOLO: loading {detector_config.model.path}")

        #providers, options = get_ort_providers(
        #    detector_config.device == "CPU", detector_config.device
        #)
        ModelName="yolo11s.pt"
        Paths=f"/config/model_cache/yolo/{ModelName}"
        """
        if os.path.isfile(Paths):

        else:
            self.model = ultralytics.YOLO(f"{ModelName}")
            os.replace(f"{ModelName}",f"config/model_cache/yolo/{ModelName}")
        """
        self.model = ultralytics.YOLO(Paths)
        self.h = detector_config.model.height
        self.w = detector_config.model.width
        self.onnx_model_type = detector_config.model.model_type
        self.onnx_model_px = detector_config.model.input_pixel_format
        self.onnx_model_shape = detector_config.model.input_tensor
        path = detector_config.model.path

        logger.info(f"YOLO: {path} loaded")

    def detect_raw(self, tensor_input: np.ndarray):
        # print(tensor_input,type(tensor_input),tensor_input.shape,tensor_input.dtype)
        # tensor_input = tensor_input.astype(self.onnx_model_shape)
        tensor_input = tensor_input.astype(np.float32)
        tensor_input = torch.from_numpy(tensor_input)
        tensor_input /= 255.0
        # print(tensor_input,type(tensor_input))
        # model_input_name = self.model.get_inputs()[0].name
        tensor_output = self.model(tensor_input, verbose=False, device=0)
        # print(tensor_output)
        if self.onnx_model_type == ModelTypeEnum.yolo:
            prediction = tensor_output[0]
            # print(prediction.names)
            detections = np.zeros((20, 6), np.float32)
            Count = 0
            for P in prediction.boxes:
                # print(prediction.boxes,type(prediction.boxes))

                if Count == 20:
                    break
                NormSize = P.xywhn[0]
                Valid = [0, 2, 3, 5, 7]
                if int(P.cls) in Valid:
                    # when running in GPU mode, empty predictions in the output have class_id of -1
                    detections[Count] = [
                        int(P.cls),
                        float(P.conf),
                        float(NormSize[1]),
                        float(NormSize[0]),
                        float(NormSize[3]),
                        float(NormSize[2]),
                    ]
                    Count += 1
            return detections
        else:
            raise Exception(
                f"{self.onnx_model_type} is currently not supported for rocm. See the docs for more info on supported models."
            )
