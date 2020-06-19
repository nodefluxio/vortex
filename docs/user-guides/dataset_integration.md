# Dataset Integration

To integrate external datasets into vortex, you need to follow several standards. This section will describe those standards.

---

## Directory Structure Standards

- The dataset directory must be placed under directory `external/datasets` .  By default, Vortex will search this folder in the current working directory. However, should you place this directory elsewhere, you can set the environment variable VORTEX_DATASET_ROOT so Vortex can find it.

    Example :

        export VORTEX_DATASET_ROOT = /home/alvinprayuda

        /*
        Means that the dataset is in /home/alvinprayuda/external/datasets
        */

    So, the directory structure will looked like this

        external/
            datasets/
                {dataset-directory}

- Each dataset directory must provide a python module `/utils/dataset.py` which will act as interface

        {dataset-directory}/
            utils/
                dataset.py

---

## Python Module Standards

- The python module can have several dataset interface classes. However it must be noted that a class represents a dataset. E.g. :

        class VOC0712DetectionDataset :
            def __init__(self):
                """
                Implement something here
                """
                pass

- All available dataset classes in the `utils/dataset.py` module must be listed in a variable named `supported_datasets`. E.g. :

        supported_dataset = [
            'VOC0712DetectionDataset'
        ]

        class VOC0712DetectionDataset :
            def __init__(self):
                """
                Implement something here
                """
                pass

- The `utils/dataset.py` interface python module must implement a function `create_dataset`. This function will receive args from the experiment file. E.g. :

        class VOC0712DetectionDataset :
            def __init__(self,*args,**kwargs):
                """
                Implement something here
                """
                pass

        def create_dataset(*args, **kwargs) :
            return VOC0712DetectionDataset(*args, **kwargs)

---

## Dataset Class Standards

Each dataset class must implement several mandatory method and attributes:

- Method `__getitem__` and `__len__` similar to [Pytorch dataset implementation](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class):

    - `__len__` function must return the number (int) of dataset e.g. the number of images. E.g. :

            class VOC0712DetectionDataset :
                def __init__(self,*args,**kwargs):
                    self.images = ['images/image1.jpg','images/image2.jpg','images/image3.jpg']

                def __len__(self):
                    return len(self.images)

    - `__getitem__` function returned value must be a tuple of image path (str) and its annotations (numpy array). E.g.

        - Classification task, if you choose not to use [torchvision's ImageFolder](../modules/builtin_dataset.md), the returned target's array dimension is 1

                class ImageClassificationDataset :
                    def __init__(self,*args,**kwargs):
                        self.images = ['images/image1.jpg','images/image2.jpg','images/image3.jpg']
                        self.labels = [[0],[2],[1]]

                    def __getitem__(self, index):
                        img_path = self.images[index]
                        target = np.array(self.labels[index])
                        return img_path, target

                dataset = ImageClassificationDataset()
                print(dataset[0][0])
                """
                'images/image1.jpg'
                """
                    
                print(dataset[0][1])
                """
                array([0], dtype=float32)
                """

        - Detection task, the returned target's array dimension is 2

                import numpy as np

                class VOC0712DetectionDataset :
                    def __init__(self,*args,**kwargs):
                        self.images = ['images/image1.jpg','images/image2.jpg','images/image3.jpg']
                        self.labels = [
                            [[0. , 0.5 , 0.5 , 0.3 , 0.2],[0. , 0.2 , 0.3 , 0.4 , 0.5]],
                            [[0. , 0.1 , 0.2 , 0.3 , 0.4]],
                            [[1. , 0.7 , 0.5 , 0.2 , 0.3],[2. , 0.4 , 0.4 , 0.3 , 0.3]],
                        ]

                    def __getitem__(self, index):
                        img_path = self.images[index]
                        target = np.array(self.labels[index])
                        return img_path, target

                dataset = VOC0712DetectionDataset()
                print(dataset[0][0])
                """
                'images/image1.jpg'
                """
                
                print(dataset[0][1])
                """
                array([[0. , 0.5, 0.5, 0.3, 0.2],
                        [0. , 0.2, 0.3, 0.4, 0.5]], dtype=float32)
                    """
    
- Attribute `self.class_names` and `self.data_format`

    -  `self.class_names` contains information about class index to string . The value must be a list with string members which the sequence corresponds to its integer class index. The returned class labels in dataset's target must correspond to this list. E.g. 

            class ImageClassificationDataset :
                def __init__(self,*args,**kwargs):
                    self.images = ['images/image1.jpg','images/image2.jpg','images/image3.jpg']
                    self.labels = [[0],[2],[1]]
                    self.class_names = ['cat','dog','bird']


                def __getitem__(self, index):
                    img_path = self.images[index]
                    target = np.array(self.labels[index])
                    return img_path, target

            dataset = ImageClassificationDataset()
            print(dataset[0][0])
            """
            'images/image1.jpg'
            """

            print(dataset[0][1])
            """
            array([0], dtype=float32)
            """

            class_label = dataset[0][1]
            class_label_string_name = dataset.class_names[class_label[0]]
            print(class_label_string_name)
            """
            'cat'

            This means that class_label = 0 correspond to string 'cat' in the self.class_names
            """

    - `self.data_format` which explains the format of dataset's target array and will be used to extract information from it. This attribute is specifically different between different tasks. Vortex utilizes [numpy.take](https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html) to slice the information from the dataset's target array. E.g.

            """
            Example

            self.data_format = {
                'bounding_box' : {
                    'indices' : [0, 1, 2, 3],
                    'axis' : 1
                },
                'class_label' : {
                    'indices' : [4],
                    'axis' : 1
                }
            }

            'indices' : [0, 1, 2, 3] -> indicate x,y,w,h index of bounding box notation
            from labels array
            'axis' : 1 -> specify the axis in which we slice the labels array

            --Example
                                        'x'   'y'    'w'   'h'  'class'
            target_array=np.array([[ 0.75 , 0.6  , 0.1 , 0.2 ,  8 ]
                                    [ 0.5  , 0.22 , 0.3 , 0.4 ,  7 ]])

            Using above data format we can slice the array to get only the bounding boxes coords

            bbox_array = np.array([[ 0.75 , 0.6 , 0.1 , 0.28]
                                    [ 0.5 , 0.22 , 0.3 , 0.4]])
            
            class_array = np.array([[8]
                                    [7]])
            """


        - Classification Task

            - Class Label Data Format

                    """
                    Class label data format
                    """
                    self.data_format = {'class_label' : None}

                    """
                    Because the annotations array size only 1, no need to specify indices and axis
                    However, `self.data_format` is still mandatory
                    """

        - Detection Task

            - Class Label Data Format

                    """
                    Class label data format
                    """
                    """
                    Option 1

                    Indicate a single class notation for object detection
                    """
                    self.data_format = {'class_label' : None} 

                    """
                    Option 2

                    Indicate a single-category multi-class notation
                    """
                    self.data_format = {
                        'class_label' : {
                            'indices' : [4],
                            'axis' : 1
                        }
                    }

                    """
                    Option 3 (FUTURE-PLAN,NOT SUPPORTED YET)

                    Indicate a multi-category multi-class notation
                    """
                    self.data_format = {
                        'class_label' : {
                            'indices' : [4,5,6],
                            'axis' : 1
                        }
                    }

                    """
                    Option 4 (FUTURE-PLAN,NOT SUPPORTED YET)

                    Indicate a multi-category multi-class notation with sequential long indexes
                    """
                    self.data_format = {
                        'class_label' : {
                            'indices' : {
                                'start' : 4,
                                'end' : 6
                            },
                            'axis' : 1
                        }
                    }

                    """
                    Explanation

                    'indices' with dict format and keys 'start' and 'end' will be converted to 
                    indices sequence internally
                    """
            
            - Bounding Box Data Format

                It must be noted that VORTEX utilize **\[x,y,w,h\]** bounding box format in a **normalized style (range 0 - 1 , [x] and [w] are normalized to image’s width, whereas [y] and [h] normalized to image’s height )**

                    """
                    Bounding box data format
                    """
                    self.data_format = {
                        'bounding_box' : {
                            'indices' : [0, 1, 2, 3],
                            'axis' : 1
                        },
                    }

            - Landmarks (Key Points) Data Format (OPTIONAL)

                This data format is 'optional' in the sense that not all detection models that support landmark (key points) prediction. Thus if you want to utilize model that predict landmarks, such as RetinaFace, this data format is mandatory

                Landmarks annotation is presented as a 1-dimensional array which has **an even length**. E.g. 

                    [ x1,y1, x2,y2, x3,y3, x4,y4, x5,y5 ]

                The given example means that we have 5 landmarks with the coordinates of **(x1,y1),(x2,y2),(x3,y3),(x4,y4), and (x5,y5) and also in normalized style (range 0 - 1 , [x] are normalized to image’s width, whereas [y] normalized to image’s height )**

                    """
                    Landmarks data format
                    """
                    """
                    Option 1

                    Standard implementation
                    """
                    self.data_format = {
                        'landmarks' : {
                            'indices' : [7,8,9,10,11,12,13,14,15,16],
                            'axis' : 1
                        }
                    }

                    """
                    Option 2

                    With asymmetric keypoint declaration
                    """
                    self.data_format = {
                        'landmarks' : {
                            'indices' : [7,8,9,10,11,12,13,14,15,16],
                            'asymm_pairs' : [[0,1],[3,4]],
                            'axis' : 1
                        }
                    }

                    """
                    Option 3

                    Implementation with long sequences
                    """
                    self.data_format = {
                        'landmarks' : {
                            'indices' : {
                                'start' : 7,
                                'end' : 16
                            }
                            'asymm_pairs' : [[0,1],[3,4]],
                            'axis' : 1
                        }
                    }

                    """
                    Explanation
                    
                    'indices' : [7,8,9,10,11,12,13,14,15,16] or
                    'indices' : {
                        'start' : 7,
                        'end' : 16
                    }

                    The implementatiom above indicates a sequence of x,y coordinates 
                    (e.g index 7,9,11,13,15 -> x coordinates ,
                    index 8,10,12,14,16 -> y coordinates)
                    Indices length must be even number

                    'asymm_pairs' : [[0,1],[3,4]]

                    Indicates asymmetric key points which can be affected by vertical/ horizontal-flip 
                    data augmentation

                    For example :

                    Internally, indices [7,8,9,10,11,12,13,14,15,16] will be converted to
                    [(7,8),(9,10),(11,12),(13,14),(15,16)] which means that the key points indexes are :

                    keypoint 0 -> (7,8)
                    keypoint 1 -> (9,10)
                    keypoint 2 -> (11,12)
                    keypoint 3 -> (13,14)
                    keypoint 4 -> (15,16)

                    In this example, we follow 5 facial landmarks example in which left and right landmarks
                    sequence is crucial

                    keypoint 0 -> (7,8) -> left eye
                    keypoint 1 -> (9,10) -> right eye
                    keypoint 2 -> (11,12) -> nose
                    keypoint 3 -> (13,14) -> left mouth
                    keypoint 4 -> (15,16) -> right mouth


                    To handle this, the data format should specify which key points index have asymmetric
                    relation, in this case keypoint 0-1 and keypoint 3-4,
                    so we annotate them as in a list as [[0,1],[3,4]]
                    """