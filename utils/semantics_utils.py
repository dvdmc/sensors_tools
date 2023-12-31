import numpy as np

# A function that obtains the different colors of an image
def get_class_colors(rgb_image, prev_class_colors=[]):
    """A function that obtains the different colors of an image as integers
    Returns:
        class_colors: a list of colors
    """
    class_colors = prev_class_colors
    rgb_np = np.array(rgb_image)
    for row in rgb_np:
        for pixel in row:
            if pixel.tolist() not in class_colors:
                class_colors.append(pixel.tolist())
    return class_colors

def get_color_map(dataset_name: str, bgr: bool = True) -> np.ndarray:
    """
        Get the color map for the dataset
        Args:
            dataset_name: name of the dataset
        Returns:
            color_map: color map for the dataset
    """
    if dataset_name == "pascal":
        color_map = get_pascal_labels(bgr=bgr)
    elif dataset_name == "nyu2":
        color_map = get_nyu2_40_labels(bgr=bgr)
    elif dataset_name == "airsim":
        color_map = get_airsim_labels(bgr=bgr)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return color_map

def get_pascal_labels(bgr=False):
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [0, 64, 0],  # 1=aeroplane # TEMPORAL CHANGE
            [0, 128, 0],  # 2=bicycle
            [128, 128, 0],  # 3=bird
            [0, 0, 128],  # 4=boat
            [128, 0, 128],  # 5=bottle
            [0, 128, 128],  # 6=bus
            [128, 128, 128],  # 7=car
            [64, 0, 0],  # 8=cat
            [192, 0, 0],  # 9=chair
            [64, 128, 0],  # 10=cow
            [192, 128, 0],  # 11=diningtable
            [64, 0, 128],  # 12=dog
            [192, 0, 128],  # 13=horse
            [64, 128, 128],  # 14=motorbike
            [192, 128, 128],  # 15=person
            [0, 64, 0],  # 16=potted plant
            [128, 64, 0],  # 17=sheep
            [0, 192, 0],  # 18=sofa
            [128, 192, 0],  # 19=train
            [0, 64, 128],  # 20=tv/monitor
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

# Get pascal labels without background
def get_pascal_labels_wo_background(bgr=False):
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (20, 3)
    """
    color_map = np.array(
        [
            [0, 64, 0],  # 1=aeroplane # TEMPORAL CHANGE
            [0, 128, 0],  # 2=bicycle
            [128, 128, 0],  # 3=bird
            [0, 0, 128],  # 4=boat
            [128, 0, 128],  # 5=bottle
            [0, 128, 128],  # 6=bus
            [128, 128, 128],  # 7=car
            [64, 0, 0],  # 8=cat
            [192, 0, 0],  # 9=chair
            [64, 128, 0],  # 10=cow
            [192, 128, 0],  # 11=diningtable
            [64, 0, 128],  # 12=dog
            [192, 0, 128],  # 13=horse
            [64, 128, 128],  # 14=motorbike
            [192, 128, 128],  # 15=person
            [0, 64, 0],  # 16=potted plant
            [128, 64, 0],  # 17=sheep
            [0, 192, 0],  # 18=sofa
            [128, 192, 0],  # 19=train
            [0, 64, 128],  # 20=tv/monitor
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_pascal_8_labels(bgr=False):
    """Load the mapping that associates 7 pascal classes with label colors
    Returns:
        np.ndarray with dimensions (7, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [128, 0, 128],  # 5=bottle, purple
            [192, 0, 0],  # 9=chair, light red
            [192, 128, 0],  # 11=diningtable, light blue
            [192, 128, 128],  # 15=person, pink
            [0, 64, 0],  # 16=potted plant, dark green
            [0, 192, 0],  # 18=sofa, light green
            [0, 64, 128],  # 20=tv/monitor, dark blue
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map


def get_pascal_7_labels_wo_background(bgr=False):
    """Load the mapping that associates 7 pascal classes with label colors
    Returns:
        np.ndarray with dimensions (7, 3)
    """
    color_map = np.array(
        [
            [128, 0, 128],  # 5=bottle, purple
            [192, 0, 0],  # 9=chair, light red
            [192, 128, 0],  # 11=diningtable, light blue
            [192, 128, 128],  # 15=person, pink
            [0, 64, 0],  # 16=potted plant, dark green
            [0, 192, 0],  # 18=sofa, light green
            [0, 64, 128],  # 20=tv/monitor, dark blue
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_nyu2_40_labels(bgr=False):
    """Load the mapping that associates NYU2 classes with label colors
    Returns:
        np.ndarray with dimensions (41, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [174, 199, 232],  # 1=wall
            [152, 223, 138],  # 2=floor
            [31, 119, 180],  # 3=cabinet
            [255, 187, 120],  # 4=bed
            [188, 189, 34],  # 5=chair
            [140, 86, 75],  # 6=sofa
            [255, 152, 150],  # 7=table
            [214, 39, 40],  # 8=door
            [197, 176, 213],  # 9=window
            [148, 103, 189],  # 10=bookshelf
            [196, 156, 148],  # 11=picture
            [23, 190, 207],  # 12=counter
            [178, 76, 76],  # 13=blinds
            [247, 182, 210],  # 14=desk
            [66, 188, 102],  # 15=shelves
            [219, 219, 141],  # 16=curtain
            [140, 57, 197],  # 17=dresser
            [202, 185, 52],  # 18=pillow
            [51, 176, 203],  # 19=mirror
            [200, 54, 131],  # 20=floormat
            [92, 193, 61],  # 21=clothes
            [78, 71, 183],  # 22=ceiling
            [172, 114, 82],  # 23=books
            [255, 127, 14],  # 24=refrigerator
            [91, 163, 138],  # 25=television
            [153, 98, 156],  # 26=paper
            [140, 153, 101],  # 27=towel
            [158, 218, 229],  # 28=showercurtain
            [100, 125, 154],  # 29=box
            [178, 127, 135],  # 30=whiteboard
            [120, 185, 128],  # 31=person
            [146, 111, 194],  # 32=nightstand
            [44, 160, 44],  # 33=toilet
            [112, 128, 144],  # 34=sink
            [96, 207, 209],  # 35=lamp
            [227, 119, 194],  # 36=bathtub
            [213, 92, 176],  # 37=bag
            [94, 106, 211],  # 38=otherstructure
            [82, 84, 163],  # 39=otherfurniture
            [100, 85, 144],  # 40=otherprop
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_nyu2_39_labels_wo_background(bgr=False):
    """Load the mapping that associates NYU2 classes with label colors
    Returns:
        np.ndarray with dimensions (41, 3)
    """
    color_map = np.array(
        [
            [174, 199, 232],  # 1=wall
            [152, 223, 138],  # 2=floor
            [31, 119, 180],  # 3=cabinet
            [255, 187, 120],  # 4=bed
            [188, 189, 34],  # 5=chair
            [140, 86, 75],  # 6=sofa
            [255, 152, 150],  # 7=table
            [214, 39, 40],  # 8=door
            [197, 176, 213],  # 9=window
            [148, 103, 189],  # 10=bookshelf
            [196, 156, 148],  # 11=picture
            [23, 190, 207],  # 12=counter
            [178, 76, 76],  # 13=blinds
            [247, 182, 210],  # 14=desk
            [66, 188, 102],  # 15=shelves
            [219, 219, 141],  # 16=curtain
            [140, 57, 197],  # 17=dresser
            [202, 185, 52],  # 18=pillow
            [51, 176, 203],  # 19=mirror
            [200, 54, 131],  # 20=floormat
            [92, 193, 61],  # 21=clothes
            [78, 71, 183],  # 22=ceiling
            [172, 114, 82],  # 23=books
            [255, 127, 14],  # 24=refrigerator
            [91, 163, 138],  # 25=television
            [153, 98, 156],  # 26=paper
            [140, 153, 101],  # 27=towel
            [158, 218, 229],  # 28=showercurtain
            [100, 125, 154],  # 29=box
            [178, 127, 135],  # 30=whiteboard
            [120, 185, 128],  # 31=person
            [146, 111, 194],  # 32=nightstand
            [44, 160, 44],  # 33=toilet
            [112, 128, 144],  # 34=sink
            [96, 207, 209],  # 35=lamp
            [227, 119, 194],  # 36=bathtub
            [213, 92, 176],  # 37=bag
            [94, 106, 211],  # 38=otherstructure
            [82, 84, 163],  # 39=otherfurniture
            [100, 85, 144],  # 40=otherprop
        ]
    )

    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_nyu2_14_classes(bgr=False):
    """Load the mapping that associates NYU2 classes with label colors
    Returns:
        np.ndarray with dimensions (13, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [255, 187, 120],  # 1=bed
            [172, 114, 82],  # 2=books
            [78, 71, 183],  # 3=ceiling
            [188, 189, 34],  # 4=chair
            [152, 223, 138],  # 5=floor
            [140, 153, 101],  # 6=furniture
            [255, 127, 14],  # 7=objects
            [161, 171, 27],  # 8=picture
            [190, 225, 64],  # 9=sofa
            [206, 190, 59],  # 10=table
            [115, 176, 195],  # 11=tv
            [153, 108, 6],  # 12=wall
            [247, 182, 210],  # 13=window
        ]
    )

    if bgr:
        color_map = color_map[:, ::-1]
    return color_map


def get_airsim_labels(bgr=False):

    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [153, 108, 6],  # 1=aeroplane
            [112, 105, 191],  # 2=bicycle
            [89, 121, 72],  # 3=bird
            [190, 225, 64],  # 4=boat
            [206, 190, 59],  # 5=bottle
            [81, 13, 36],  # 6=bus
            [115, 176, 195],  # 7=car
            [161, 171, 27],  # 8=cat
            [135, 169, 180],  # 9=chair
            [29, 26, 199],  # 10=cow
            [102, 16, 239],  # 11=diningtable
            [242, 107, 146],  # 12=dog
            [156, 198, 23],  # 13=horse
            [49, 89, 160],  # 14=motorbike
            [68, 218, 116],  # 15=person
            [11, 236, 9],  # 16=potted plant
            [196, 30, 8],  # 17=sheep
            [121, 67, 28],  # 18=sofa
            [0, 53, 65],  # 19=train
            [146, 52, 70],  # 20=tv/monitor
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_airsim_labels2(bgr=False):

    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [153, 108, 6],  # 5=bottle
            [112, 105, 191],  # 9=chair
            [89, 121, 72],  # 11=diningtable
            [116, 218, 68],  # 15=person
            [206, 190, 59],  # 16=potted plant
            [81, 13, 36],  # 18=sofa
            [115, 176, 195],  # 20=tv/monitor
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

# Transform rgb colors to class labels
def rgb_to_class(rgb_image, class_colors):
    """A function that converts each color of an rgb image in a matrix with class labels (class_colors can be getPascalLabels())
    Returns:
        class_image: a matrix with the same shape as the input image, where each pixel is a class label
    """
    rgb_np = np.array(rgb_image, dtype=np.uint8)[:, :, :3]
    class_image = np.zeros(rgb_np.shape[:2], dtype=np.uint8)

    for class_label, class_color in enumerate(class_colors):
        mask = np.all(rgb_np == class_color, axis=-1)
        class_image[mask] = class_label
    return class_image

# Transform class labels to rgb colors
def class_to_rgb(class_image, class_colors):
    """A function that converts each class label of a class image in a matrix with rgb colors (class_colors can be getPascalLabels())
    Returns:
        rgb_image: a matrix with the same shape as the input image, where each pixel is a rgb color
    """
    rgb_image = np.zeros(
        (class_image.shape[0], class_image.shape[1], 3), dtype=np.uint8)
    for class_label, class_color in enumerate(class_colors):
        mask = class_image == class_label
        rgb_image[mask] = class_color
    return rgb_image

def label2rgb(label_map, label_colors): # Not related to sent info, only display
    num_classes = label_colors.shape[0]
    label_colors = label_colors[:num_classes, :]
    rgb_label_map = utils.class_to_rgb(label_map, label_colors)

    return rgb_label_map

def rgb2label(self, rgb_label_map, label_colors):
    num_classes = label_colors.shape[0]
    label_colors = label_colors[:num_classes, :]
    # Save img
    label_map = utils.rgb_to_class(rgb_label_map, label_colors)
    return label_map