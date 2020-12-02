import cv2
import torch
import logging

from pathlib import Path
from copy import deepcopy

class BaseExporter :
    logger = logging.getLogger(__name__)
    def __init__(self, filename: str, image_size: int, input_dtype: str='uint8', n_channels=3, n_batch=1) :
        if isinstance(image_size, int) :
            image_size = (image_size, image_size)
        if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            image_size = (n_batch, *image_size, n_channels)
        assert len(image_size) == 4
        self.image_size = image_size
        self.input_type = input_dtype
        self.filename = filename

    def export(self, predictor, example_input, class_names, output_format, additional_inputs) :
        raise NotImplementedError

    def __call__(self, predictor, class_names=None, example_image_path=None) :
        predictor = deepcopy(predictor).eval()

        if example_image_path is None:
            example_input = torch.rand(*self.image_size)
            if self.input_type == 'uint8' :
                example_input = (example_input * 255).long().type(torch.uint8)
        elif isinstance(example_image_path, (Path, str, list)):
            ## using example-input means infer type from image, warn (?)
            type(self).logger.warning('using {} as example input'.format(example_image_path))
            example_image_paths = example_image_path \
                if isinstance(example_image_path, list) else [example_image_path]
            if len(example_image_paths)==1 :
                example_image_paths = example_image_paths * self.image_size[0]
            assert len(example_image_paths)==self.image_size[0]
            example_inputs = []
            for example_image_path in example_image_paths :
                ## opencv2 doesn't check if file exists
                assert Path(example_image_path).exists()
                example_input = cv2.imread(example_image_path)
                example_input = cv2.resize(example_input, self.image_size[1:-1])
                example_inputs.append(torch.from_numpy(example_input).unsqueeze(0))
            example_input = torch.cat(example_inputs, dim=0)
        else:
            raise RuntimeError("Unknown 'example_image_path' type of {} with value {}, "
                "expected [Path, str, list[str]]".format(type(example_image_path), example_image_path))

        class_names = [] if class_names is None else class_names
        assert all(isinstance(class_name, str) for class_name in class_names)
        additional_inputs = tuple()
        if hasattr(predictor.postprocess, 'additional_inputs'):
            additional_inputs = predictor.postprocess.additional_inputs
            ## typecheck
            assert isinstance(additional_inputs, tuple)
            assert len(additional_inputs)
            assert all(isinstance(additional_input, tuple) for additional_input in additional_inputs)
            ## TODO : check additional_input[0][0] is str, additional_input[0][1] is tuple of int
        output_format = predictor.output_format
        class_names = dict(map(lambda x: (x[1],x[0]), enumerate(class_names)))
        return self.export(
            predictor=predictor, 
            example_input=example_input,
            class_names=class_names,
            output_format=output_format,
            additional_inputs=additional_inputs,
        )
