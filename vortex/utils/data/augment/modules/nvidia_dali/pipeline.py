import nvidia.dali.ops as ops
import copy
import nvidia.dali.types as types
from ....augment import create_transform
from easydict import EasyDict
from nvidia.dali.pipeline import Pipeline



class DALIExternalSourcePipeline(Pipeline):
    def __init__(self, dataset_iterator, 
                       batch_size, 
                       num_threads, 
                       device_id, 
                       dali_augments = None,
                       normalize = True,
                       seed = 12345):
        super().__init__(
                        batch_size,
                        num_threads,
                        device_id,
                        seed=seed)
        self.original_data_layout = copy.copy(dataset_iterator.data_layout)
        self.pipeline_output_data_layout = copy.copy(dataset_iterator.data_layout)
        self.source = ops.ExternalSource(source = dataset_iterator, 
                                         num_outputs = len(dataset_iterator.data_layout))
        self.image_decode = ops.ImageDecoder(device = "mixed", output_type = types.BGR)
        self.label_cast = ops.Cast(device="cpu",
                             dtype=types.FLOAT)
        self.labels_pad_value = -99

        # Nvidia DALI augmentations from experiment file
        self.dali_augments = dali_augments

        # Modify pipeline output data layout to prepare placeholder for required information
        # that need to be passed from DALI pipeline to DALI loader
        
        ## Information placeholder for flip augmentation flag used for asymmetric information
        flag_count = 0
        if self.dali_augments:
            for augment in self.dali_augments.compose:

                # Somehow, VerticalFlip in DALI didn't break the sequence for asymm landmarks
                if type(augment).__name__ == 'HorizontalFlip' or type(augment).__name__ == 'VerticalFlip':
                    self.pipeline_output_data_layout.append('flip_flag_'+str(flag_count))
                    flag_count+=1

        ## Information placeholder for standard augment used for coordinates fixing
        self.pipeline_output_data_layout.append('pre_padded_image_shape')
        # Standard Augments Resize and Pad to Square
        preprocess_args = dataset_iterator.dataset.preprocess_args
        if 'mean' not in preprocess_args.input_normalization:
            preprocess_args.input_normalization.mean=[0.,0.,0.]
        if 'std' not in preprocess_args.input_normalization:
            preprocess_args.input_normalization.mean=[1.,1.,1.]
        if 'scaler' not in preprocess_args.input_normalization:
            preprocess_args.input_normalization.scaler=255

        standard_augment = EasyDict()
        standard_augment.data_format = dataset_iterator.dataset.data_format
        standard_augment.transforms = [
            {'transform': 'StandardAugment','args':{'scaler' : preprocess_args.input_normalization.scaler,
                                                    'mean' : preprocess_args.input_normalization.mean,
                                                    'std' : preprocess_args.input_normalization.std,
                                                    'input_size' : preprocess_args.input_size,
                                                    'image_pad_value' : 0,
                                                    'labels_pad_value' : self.labels_pad_value,
                                                    'normalize' : normalize
                                                    }
            }
        ]
        self.standard_augments = create_transform(
            'nvidia_dali', **standard_augment)

    def define_graph(self):
        pipeline_input = self.source()

        # Prepare pipeline input data
        data = EasyDict()
        graph_data=dict(zip(self.original_data_layout, pipeline_input))
        data.images = self.image_decode(graph_data['images'])
        data.labels = EasyDict()
        for layout in self.original_data_layout:
            if layout!='images':
                data.labels[layout]=self.label_cast(graph_data[layout])
        data.additional_info=EasyDict()
        data.data_layout=self.original_data_layout
        # Custom Augmentation Start

        if self.dali_augments:
            data = self.dali_augments(**data)

        # Custom Augmentation End
        data = self.standard_augments(**data)

        # Prepare pipeline output tuple
        pipeline_output = []

        # Iterate new layout
        new_layout = data.data_layout

        for component in new_layout:
            if component == 'images':
                pipeline_output.append(data[component])
            elif component in data.labels:
                pipeline_output.append(data.labels[component])
            elif component in data.additional_info:
                pipeline_output.append(data.additional_info[component])
        return pipeline_output