import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple, Union, List, Dict, Sequence

import vortex.runtime as vrt

# TODO: move this to another directory
class Visual:
    """Helper class for various drawing routine accepting formated result
    """
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 127, 127), (127, 255, 127), (127, 127, 255),
        (255, 0, 255), (0, 255, 255), (255, 255, 255),
    ] * 115 ## >= imagenet
    font, font_scale, line_type = cv2.FONT_HERSHEY_SIMPLEX, 1, 2

    def __init__(self, class_names):
        self.class_names = class_names

    @classmethod
    def draw_bbox(cls, vis: np.ndarray, tl: Tuple[int,int], rb: Tuple[int,int], color: Tuple[int,int,int]=colors[0]) :
        """draw bounding box on `vis`

        Args:
            vis (np.ndarray): [input output] array in which bounding box to be visualized
            tl (Tuple[int,int]): top-left
            rb (Tuple[int,int]): bottom-right
            color (Tuple[int,int,int], optional): desired color of the rectangle. Defaults to colors[0].

        Returns:
            np.ndarray: image with bbox visualized
        """        
        cv2.rectangle(vis, tl, rb, color, 1)
        return vis

    @classmethod
    def draw_bboxes(cls, vis: np.ndarray, bboxes, classes, confidences, color_map=None, class_names=None) :
        """draw multiple bounding box on `vis`

        Args:
            vis (np.ndarray): array in which bounding box is to be visualized
            bboxes (iterable of tuple): bounding boxes to be visualized
            classes (iterable of int): list of classes corresponding to each bounding box
            confidences (iterable of float): list of confidences corresponding to each bounding box
            color_map (mapping, optional): mapping from class to color. Defaults to None.
            class_names (mapping, optional): mapping from class to str represnting human-readable class name. Defaults to None.

        Returns:
            np.ndarray: array with visualization
        """        
        color_map = color_map if color_map else cls.colors
        for bbox, label, confidence in zip(bboxes, classes, confidences) :
            label = int(label)
            color = color_map[label]
            x1, y1, x2, y2 = bbox
            vis = cls.draw_bbox(vis, (x1,y1), (x2,y2), color=color)
            vis = cls.draw_label(vis, label, confidence, (x1,y1), color, class_names=class_names)
        return vis

    @classmethod
    def draw_landmarks(cls, vis: np.ndarray, landmarks, color: Tuple[int,int,int]=None, radius=2, thickness=-1) :
        """draw multiple landmarks on `vis`

        Args:
            vis (np.ndarray): array in which landmarks are to be visualized
            landmarks (np.ndarray): landmarks to be visualized
            color (Tuple[int,int,int], optional): desired color of point for visualization. Defaults to None.
            radius (int, optional): desired radius of point for visualization. Defaults to 2.
            thickness (int, optional): desired thickness of point for visualization. Defaults to -1.

        Returns:
            np.ndarray: array with visualization
        """        
        color = color if color else cls.colors[0]
        for landmark in landmarks :
            assert (len(landmark) % 2) == 0
            xpts, ypts = landmark[0::2], landmark[1::2]
            for x, y in zip(xpts, ypts) :
                pt = int(x), int(y)
                cv2.circle(vis, pt, radius=radius, color=color, thickness=thickness)
        return vis

    @classmethod
    def draw_label(cls, vis, obj_class, confidence, bl, color, class_names=None) :
        """draw single label on `vis`

        Args:
            vis (np.ndarray): array in which label is to be visualized
            obj_class (integer): object class/label
            confidence (scalar): object confidence
            bl (tuple): bottom-left point to visualize label
            color (tuple): desired color to visualize label
            class_names (mapping, optional): mapping from int label to human-readable str. Defaults to None.

        Returns:
            np.ndarray: array with visualiazation
        """        
        obj_class = obj_class.item() if isinstance(obj_class, np.ndarray) else int(obj_class)
        confidence = confidence.item() if isinstance(confidence, np.ndarray) else float(confidence)
        class_name = class_names[obj_class] if class_names else 'class_{}'.format(obj_class)
        class_name = '{0} : {1:.2f}'.format(class_name, confidence)
        cv2.putText(vis, class_name, bl, cls.font, cls.font_scale, color, cls.line_type)
        return vis

    @classmethod
    def draw_labels(cls, vis, obj_classes, confidences, bls : Sequence[Tuple[int,int]], color_map=colors, class_names=None) :
        """draw multiple labels on `vis`

        Args:
            vis (np.ndarray): array in which label is to be visualized
            obj_classes (iterable): object classes/labels
            confidences (iterable): object confidences
            bls (Sequence[Tuple[int,int]]): bottom-left points corresponding to each label to be visualized
            color_map (mapping, optional): mapping from int to colors. Defaults to colors.
            class_names (mapping, optional): mapping from int label to human-readable str. Defaults to None.

        Returns:
            np.ndarray: array with visualiazation
        """        
        for obj_class, confidence, bl in zip(obj_classes, confidences, bls) :
            color = color_map[int(obj_class)]
            vis = cls.draw_label(vis, obj_class, confidence, bl, color, class_names=class_names)
        return vis
    
    @classmethod
    def draw(cls, result: dict, vis: np.ndarray, class_names=None, color_map=colors):
        """draw single prediction result on `vis`

        Args:
            result (dict): single prediction result
            vis (np.ndarray): array in which prediction result is to be visualized
            class_names (mapping, optional): mapping from int label to human-readable str. Defaults to None.
            color_map (mapping, optional): mapping from int to colors. Defaults to colors.

        Returns:
            np.ndarray: array with visualiazation
        """        
        class_label = None
        if 'class_label' in result:
            class_label = result['class_label']
        else:
            class_label = np.zeros((result['class_confidence'].shape[0], 1))
        class_confidence = result['class_confidence']

        if 'bounding_box' in result:
            assert 'class_confidence' in result
            bounding_box = result['bounding_box']
            if bounding_box is not None:
                vis = cls.draw_bboxes(vis, bounding_box, class_label, class_confidence, 
                                class_names=class_names, color_map=color_map)
        else:
            label_pts = np.asarray([[0, int(vis.shape[0]*0.95)]] * len(class_label))
            label_pts = [tuple(label_pt) for label_pt in label_pts.tolist()]
            vis = cls.draw_labels(vis, class_label.astype(np.int), class_confidence, label_pts, 
                              color_map=color_map, class_names=class_names)

        if 'landmarks' in result:
            landmarks = result['landmarks']
            if landmarks is not None:
                vis = cls.draw_landmarks(vis, landmarks)
        return vis

    @classmethod
    def visualize_result(cls, vis: np.ndarray, results: List[Dict[str,np.ndarray]] , class_names=None, color_map=colors):
        """draw single-batch prediction result on `vis`

        Args:
            vis (np.ndarray): array in which prediction result is to be visualized
            results (List[Dict[str,np.ndarray]]): single-batch prediction result
            class_names (mapping, optional): mapping from int label to human-readable str. Defaults to None.
            color_map (mapping, optional): mapping from int to colors. Defaults to colors.

        Returns:
            np.ndarray: array with visualiazation
        """
        for result in results:
            vis = cls.draw(result, vis, class_names=class_names, color_map=color_map)
        return vis

    @classmethod
    def visualize(cls, batch_vis: List, batch_results: List, class_names=None) -> List:
        """draw batched prediction result on `vis`

        Args:
            batch_vis (List): batch image for visualization
            batch_results (List): batched prediction result
            class_names (mapping, optional): mapping from int label to human-readable str. Defaults to None.

        Returns:
            np.ndarray: array with visualiazation
        """
        f = lambda vis, results: cls.visualize_result(vis=vis, results=[results], class_names=class_names)
        result_vis = list(map(f, batch_vis, batch_results))
        return result_vis
    
    def __call__(self, batch_vis: List, batch_results: List):
        return self.visualize(batch_vis, batch_results, self.class_names)

# TODO: move this to another directory
class InferenceHelper:
    """Helper class for to load images, inference, and visualization for convinience
    """
    def __init__(self, model):
        self.model = model
        self.class_names = model.class_names
    
    @staticmethod
    def create_runtime_model(**kwargs):
        """helper method to instantiate model

        Returns:
            InferenceHelper: wrapped model
        """        
        model = vrt.create_runtime_model(**kwargs)
        return InferenceHelper(model)

    @classmethod
    def run_inference(cls, model, batch_imgs, **kwargs):
        """run inference on batched (possibly non-uniform size) image

        Args:
            model (runtime): vortex rt model
            batch_imgs (list): list image for inference

        Returns:
            list of dict: list of dictionary corresponding to each image
        """
        input_shape = model.input_specs['input']['shape']
        n = input_shape[0]
        # TODO allows batch > n, by running multiple times
        assert len(batch_imgs) <= n, "expects 'images' <= n batch ({}) got {}".format(n, len(batch_imgs))

        # stretch resize input
        batch_imgs = type(model).resize_batch([mat.copy() for mat in batch_imgs], input_shape)
        results = model(batch_imgs, **kwargs)

        return results

    @classmethod
    def load_images(cls, images):
        """load images from list of files

        Args:
            images (list): list of files

        Raises:
            TypeError: unknown type passed

        Returns:
            list: list of loaded np.ndarray
        """
        # Check image availability if image path is provided
        if isinstance(images[0],str) or isinstance(images[0],Path):
            image_paths = [Path(image) for image in images]
            for image_path in image_paths :
                assert image_path.exists(), "image {} doesn't exist".format(str(image_path))
            # TODO make sure to load with 3 channel
            batch_mat = [cv2.imread(image) for image in images]
        elif isinstance(images[0],np.ndarray):
            batch_mat = images
            assert all(len(image.shape) == 3 for image in batch_mat), \
                "Provided 'images' list member in numpy ndarray must be of dim 3, [h , w , c]"
        else:
            raise TypeError("'images' arguments must be provided with list of image path or list of "
                "numpy ndarray, found {}".format(type(images[0])))
        return batch_mat
    
    @classmethod
    def adjust_coordinates(cls, batch_vis, batch_results, coordinate_fmt='relative'):
        """adjust prediction results for visualization

        Args:
            batch_vis (list): list of image for visualization
            batch_results (list): prediction results to be transformed
            coordinate_fmt (str, optional): output coordinat format. Defaults to 'relative'.

        Returns:
            list: list of transformed prediction results
        """
        known_coordinate_fmt = ['relative', 'absolute']
        assert coordinate_fmt in known_coordinate_fmt, \
            f"available 'output_coordinate_format': {known_coordinate_fmt}"
        def adjust_coordinate(vis, result):
            # assume HWC format
            im_h, im_w, im_c = vis.shape
            if 'bounding_box' in result:
                bounding_box = result['bounding_box']
                if bounding_box is not None:
                    bounding_box[...,0::2] *= im_w
                    bounding_box[...,1::2] *= im_h
                    result['bounding_box'] = bounding_box
            if 'landmarks' in result:
                landmarks = result['landmarks']
                if landmarks is not None:
                    landmarks[...,0::2] *= im_w
                    landmarks[...,1::2] *= im_h
                    result['landmarks'] = landmarks
            return result
        if coordinate_fmt == 'relative':
            batch_results = list(map(adjust_coordinate, batch_vis, batch_results))
        return batch_results

    @classmethod
    def save_images(cls, filenames: List, batch_vis: List, output_dir: Union[str,Path]='.', output_file_prefix='prediction'):
        """save images

        Args:
            filenames (List): original filenames
            batch_vis (List): visualization results to be saved
            output_dir (Union[str,Path], optional): output directory. Defaults to '.'.
            output_file_prefix (str, optional): prefix to be added to output filenames. Defaults to 'prediction'.

        Returns:
            list: output filenames
        """
        filename_fmt = "{filename}.{suffix}"
        def make_filename(image):
            filename = Path(image)
            suffix = filename.suffix.replace('.','')
            filename = Path(output_dir) / filename_fmt.format_map(
                dict(filename='_'.join([output_file_prefix, filename.stem]),suffix=suffix)
            )
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True)
            return filename
        filenames_ = list(map(make_filename, filenames))
        write_image = lambda filename, vis: cv2.imwrite(str(filename),vis)
        # note that cv2.imwrite doesnt return status
        _ = list(map(write_image, filenames_, batch_vis))
        return filenames_

    @classmethod
    def run_and_visualize(cls, model,
            images: Union[List[str],np.ndarray],
            output_coordinate_format: str='relative',
            visualize: bool=False,
            dump_visual: bool=False,
            output_dir: Union[str,Path]='.',
            class_names=None,
            **kwargs) -> dict:
        """run inference on model with given images paths

        Args:
            model (vrt.BaseRuntime): vorted rt model
            images (Union[List[str],np.ndarray]): list of image's path
            output_coordinate_format (str, optional): output coordinate format. Defaults to 'relative'.
            visualize (bool, optional): visualize output. Defaults to False.
            dump_visual (bool, optional): save images. Defaults to False.
            output_dir (Union[str,Path], optional): output directory. Defaults to '.'.

        Returns:
            dict: prediction results
        """    
        if isinstance(images, (str,Path)):
            images = [images]

        # load images
        batch_mat = cls.load_images(images)

        # copy image for visualization
        batch_vis = [mat.copy() for mat in batch_mat]
        batch_imgs = batch_mat

        # simple measurement
        start_time = time.time()
        # actually run inference
        results = cls.run_inference(model, batch_imgs, **kwargs)
        end_time = time.time()
        dt = end_time - start_time

        # Transform coordinate-based result from relative coordinates to absolute value
        results = cls.adjust_coordinates(
            batch_vis=batch_vis, batch_results=results,
            coordinate_fmt=output_coordinate_format
        )

        results = dict(
            prediction=results,
            runtime=dt,
        )

        # Visualize prediction
        if visualize:
            result_vis = Visual.visualize(batch_vis=batch_vis, class_names=class_names, batch_results=results['prediction'])
            results['visualization'] = result_vis
            # Dump prediction only support when given image is list of filenames
            if dump_visual and isinstance(images,list):
                saved_images = cls.save_images(images, batch_vis)
                saved_images = [str(img) for img in saved_images]
                print('prediction saved to {}'.format(str(', '.join(saved_images))))

        return results
    
    def __call__(self, *args, **kwargs):
        return self.run_and_visualize(self.model, class_names=self.class_names, *args, **kwargs)

def main(args):
    vrt.check_available_runtime()
    kwargs = dict(
        model_path=args.model,
        runtime=args.runtime,
    )
    model = InferenceHelper.create_runtime_model(**kwargs)
    print(model.model)
    results = model(args.input,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        visualize=True, dump_visual=True
    )
    print("prediction results\n", results['prediction'])

    if args.show:
        print(type(results['visualization']))
        vis = np.vstack(results['visualization'])
        cv2.imshow("prediction results", vis)
        cv2.waitKey()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--runtime', type=str, help='runtime engine')
    parser.add_argument('--input', type=str, help='input image')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument("--score_threshold", default=0.9, type=float,
                        help='score threshold for detection, only used if model is detection, ignored otherwise')
    parser.add_argument("--iou_threshold", default=0.2, type=float,
                        help='iou threshold for nms, only used if model is detection, ignored otherwise')
    args = parser.parse_args()
    main(args)