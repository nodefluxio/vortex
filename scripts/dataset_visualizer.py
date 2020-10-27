import vortex.development.utils.data.dataset.wrapper as vortex_wrapper
import vortex.development.utils.visual as vortex_visual
import vortex.development.utils.data.dataset.dataset as vortex_dataset
from vortex.development.core.factory import (
    create_dataset, create_dataloader
)
import yaml
from easydict import EasyDict

import tkinter
import ttk
import numpy as np

from typing import List, Union

import PIL
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from easydict import EasyDict
from typing import List, Union, Dict
from math import ceil, sqrt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

# TODO: move to vortex visual
def draw_xyxy_bboxes(vis: np.ndarray, bboxes, classes, confidences, color_map=vortex_visual.colors, class_names=None) :
    """draw bounding boxes from x1y1x2y fromat

    Args:
        vis (np.ndarray): image in which bounding boxes to be drawn
        bboxes (iterable): list of bounding box
        classes (iterable): corresponding class for each bounding boxes
        confidences (iterable): corresponding confidences for each bounding boxes
        color_map (mapping, optional): mapping from classes to actual color (tuple). Defaults to vortex_visual.colors.
        class_names (mapping, optional): mapping from classes to class name (str). Defaults to None.

    Returns:
        vis: image with bounding box drawn
    """    
    for bbox, label, confidence in zip(bboxes, classes, confidences) :
        label = int(label)
        color = color_map[label]
        x1, y1, x2, y2 = bbox
        vis = vortex_visual.draw_bbox(vis, (x1,y1), (x2,y2), color=color)
        vis = vortex_visual.draw_label(vis, label, confidence, (x1,y1), color, class_names=class_names)
    return vis

# TODO: move to vortex visual
def draw_xywh_bboxes(vis: np.ndarray, bboxes, classes, confidences, color_map=vortex_visual.colors, class_names=None):
    """draw bounding boxes from xywh fromat

    Args:
        vis (np.ndarray): image in which bounding boxes to be drawn
        bboxes (iterable): list of bounding box
        classes (iterable): corresponding class for each bounding boxes
        confidences (iterable): corresponding confidences for each bounding boxes
        color_map (mapping, optional): mapping from classes to actual color (tuple). Defaults to vortex_visual.colors.
        class_names (mapping, optional): mapping from classes to class name (str). Defaults to None.

    Returns:
        np.ndarray: image with bounding box drawn
    """    
    for bbox, label, confidence in zip(bboxes, classes, confidences) :
        label = int(label)
        color = color_map[label]
        x1, y1, w, h = bbox
        vis = vortex_visual.draw_bbox(vis, (x1,y1), (x1+w,y1+h), color=color)
        vis = vortex_visual.draw_label(vis, label, confidence, (x1,y1), color, class_names=class_names)
    return vis

# TODO: move to vortex visual
def draw_bboxes(vis: np.ndarray, bboxes, classes, confidences, color_map=vortex_visual.colors, class_names=None, format='xyxy'):
    """draw bounding boxes either from xyxy or xywh format

    Args:
        vis (np.ndarray): image in which bounding boxes to be drawn
        bboxes (iterable): list of bounding box
        classes (iterable): corresponding class for each bounding boxes
        confidences (iterable): corresponding confidences for each bounding boxes
        color_map (mapping, optional): mapping from classes to actual color (tuple). Defaults to vortex_visual.colors.
        class_names (mapping, optional): mapping from classes to class name (str). Defaults to None.

    Returns:
        np.ndarray: image with bounding box drawn
    """    
    if format == 'xyxy':
        return draw_xyxy_bboxes(vis,bboxes,classes,confidences,color_map,class_names)
    elif format == 'xywh':
        return draw_xywh_bboxes(vis,bboxes,classes,confidences,color_map,class_names)


def visualize(data_pair, data_format, figure: Figure=None, show_img_stats=True, bins: Union[str,int]='auto', class_names=None):
    """visualize dataset entry to matplotlib

    Args:
        data_pair (tuple-like): pair of image, label
        data_format (dict): vortex dataset's data_format
        figure (Figure, optional): matplotlib figure for drawing. Defaults to None.
        show_img_stats (bool, optional): show boxplot and histograms. Defaults to True.
        bins (Union[str,int], optional): number of bins. Defaults to 'auto'.
        class_names (list, optional): vortex dataset's class_names. Defaults to None.

    Returns:
        dict: mapping from string to matplotlib axes
    """
    if not figure:
        his_fig, his_ax = plt.subplots()
        box_fig, box_ax = plt.subplots()
        img_fig, img_ax = plt.subplots()
    elif show_img_stats:
        gs = figure.add_gridspec(2,4)
        his_fig, his_ax = figure, figure.add_subplot(gs[0,2:])
        box_fig, box_ax = figure, figure.add_subplot(gs[1,2:])
        img_fig, img_ax = figure, figure.add_subplot(gs[0:,0:2])
    else:
        gs = figure.add_gridspec(1,1)
        his_fig, his_ax = None, None
        box_fig, box_ax = None, None
        img_fig, img_ax = figure, figure.add_subplot(gs[0:,0:2])

    image, labels = data_pair
    # since this visualization script read the dataset (instead of wrapper/dataloader)
    # there is a chance that PIL image is given
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    h, w, c = image.shape

    # tensor across observation for histogram and boxplot
    tensor: np.ndarray = np.asarray(image).flatten()

    if his_ax:
        histogram = np.histogram(tensor,bins=bins)
        his_ax.hist(tensor, bins=bins)
    if box_ax:
        red_square = dict(markerfacecolor='r', marker='s')
        box_ax.boxplot(tensor, vert=False, flierprops=red_square)

    # dont plot here, let the caller decide
    # plt.show()

    # TODO: do not hard code here, read from standard module (if any) instead
    classification_format = ['class_label']
    detection_format = ['bounding_box', 'class_label']

    data_format_keys = list(data_format.keys())

    # TODO: matplotlib expect RGB, torchvisiondataset use RGB, vortex assume opencv (BGR)
    # image = cv2.cvtColor(image, cv2.BGR2RGB)

    if data_format_keys == classification_format:
        # TODO: use np.take to slice labels, add class_names if any
        bl = int(0), int(0.9 * h)
        class_label = labels[0] if isinstance(labels, list) else int(labels)
        color = vortex_visual.colors[class_label]
        # TODO: do not change global variable, pass as param instead
        # vortex_visual.font_scale = 0.25
        image = vortex_visual.draw_label(
            image, obj_class=class_label,
            confidence=1, bl=bl, color=color,
            class_names=class_names
        )
    elif data_format_keys == detection_format:
        # TODO: dont do this here, move to vortex_visual
        # TODO: use np.take to slice labels, add class_names if any
        bbox_fmt = data_format['bounding_box']
        clss_fmt = data_format['class_label']
        bounding_boxes = np.take(labels, bbox_fmt['indices'], bbox_fmt['axis'])
        class_labels   = np.take(labels, clss_fmt['indices'], clss_fmt['axis'])
        confidences    = [1]*len(class_labels)
        bounding_boxes[:,0] *= w
        bounding_boxes[:,1] *= h
        bounding_boxes[:,2] *= w
        bounding_boxes[:,3] *= h
        image = draw_bboxes(
            image, bboxes=bounding_boxes.astype(int),
            classes=class_labels, confidences=confidences,
            format='xywh', class_names=class_names
        )

    img_ax.imshow(image)

    visualization = EasyDict(
        histogram=dict(
            figure=his_fig,
            axes=his_ax,
        ),
        boxplot=dict(
            feature=box_fig,
            axes=box_ax,
        ),
        image=dict(
            figure=img_fig,
            axes=img_ax,
        )
    )
    return visualization

class EazyViz:
    """Visualization helper
    """    
    def __init__(self, title, dataset, show_img_stats=False):
        """initialize visualization

        Args:
            title (str): title of visualizer
            dataset (object): vortex dataset
            show_img_stats (bool, optional): show histogram & boxplot. Defaults to False.
        """        
        # dataset
        # observed_nodes = list(dataset.keys())
        self.dataset = dataset

        root = tkinter.Tk()
        root.title(title)
        # Add a grid
        mainframe = tkinter.Frame(root)
        sticky = (tkinter.N,tkinter.W,tkinter.E,tkinter.S)
        mainframe.grid(column=0, row=0, sticky=sticky)
        mainframe.columnconfigure(0, weight=1)
        mainframe.rowconfigure(0, weight=1)
        mainframe.pack(pady=10, padx=100)
        self.root = root
        self.mainframe = mainframe
        # Create a Tkinter variable
        tkvar = tkinter.StringVar(root)
        # Dictionary with options
        choices = list([str(i) for i in range(len(dataset))])
        tkvar.set('1') # set the default option
        label = ttk.Label(mainframe, text="Index").grid(row=1, column=1)
        # note that width is not automatically adjusted
        combobox = ttk.Combobox(mainframe, textvariable=tkvar, values=choices, width=90)
        combobox.grid(row=2, column=1)
        next_button = ttk.Button(mainframe, text ="Next", command=self.next).grid(row=2, column=2)
        prev_button = ttk.Button(mainframe, text ="Prev", command=self.prev).grid(row=2, column=0)
        # link function to change dropdown
        tkvar.trace('w', self.change_dropdown)
        self.label = label
        self.tkvar = tkvar
        self.combobox = combobox
        self.next_button = next_button
        self.prev_button = prev_button

        figure = plt.figure()
        # observer = dataset[observed_nodes[0]]
        self.show_img_stats = show_img_stats
        self.class_names = dataset.class_names if hasattr(dataset, 'class_names') else None
        visualization = visualize(dataset[0], dataset.data_format, figure, show_img_stats, class_names=self.class_names)
        figure.tight_layout()
        self.figure = figure
        # visualization.histogram.figure.show()

        # plotter
        canvas = FigureCanvasTkAgg(self.figure, master=root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        self.canvas = canvas
        self.toolbar = toolbar

        button = tkinter.Button(master=root, text="Quit", command=self.quit)
        button.pack(side=tkinter.BOTTOM)
        self.button = button
    
    def quit(self):
        """quit gui program
        """        
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent

    def change_dropdown(self,*args):
        """callback to be called when user change dropdown menu
        """        
        key = self.tkvar.get()
        self.figure.clear()
        data = self.dataset[int(key)]
        visualization = visualize(data, dataset.data_format, self.figure, self.show_img_stats, class_names=self.class_names)
        self.figure.tight_layout()
        self.canvas.draw()
    
    def next(self):
        """callback to be called when user click next button
        """        
        key = int(self.tkvar.get()) + 1
        key = max(0, key)
        key = min(key, len(self.dataset)-1)
        self.tkvar.set(str(key))
    
    def prev(self):
        """callback to be called when uesr click prev button
        """        
        key = int(self.tkvar.get()) - 1
        key = max(0, key)
        key = min(key, len(self.dataset)-1)
        self.tkvar.set(str(key))

    def mainloop(self):
        """blocking call to actual gui render
        """        
        self.root.mainloop()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="path to config file")
    parser.add_argument('--stats', default=False, action='store_true', help="path to config file")
    args = parser.parse_args()
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))
    # TODO: use dataset wrapper / loader
    dataset = vortex_dataset.get_base_dataset(config.dataset.train.name, config.dataset.train.args)
    ezviz = EazyViz(config.dataset.train.name, dataset, args.stats)
    ezviz.mainloop()