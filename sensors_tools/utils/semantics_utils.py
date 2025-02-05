import numpy as np


# A function that obtains the different colors of an image
def compute_class_colors(rgb_image, prev_class_colors=[]):
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
    if dataset_name == "coco_voc":
        color_map = get_pascal_labels(bgr=bgr)
    elif dataset_name == "pascal_8":
        color_map = get_pascal_8_labels(bgr=bgr)
    elif dataset_name == "nyu2":
        color_map = get_nyu2_40_labels(bgr=bgr)
    elif dataset_name == "nyu2_14":
        color_map = get_nyu2_14_classes(bgr=bgr)
    elif dataset_name == "airsim":
        color_map = get_airsim_labels(bgr=bgr)
    elif dataset_name == "binary":
        color_map = get_binary_labels(bgr=bgr)
    elif dataset_name == "ade20k":
        color_map = get_ade20k_labels(bgr=bgr)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return color_map


def get_label_mapper(mapper_name: str) -> dict:
    """
    Get the label mapper for the dataset
    Args:
        mapper_name: name of the mapper
    Returns:
        label_mapper: label mapper for the dataset
    """
    if mapper_name == "coco_voc_2_pascal_8":
        label_mapper = get_coco_voc_2_pascal_8_label_mapper()
    else:
        raise ValueError(f"Mapper {mapper_name} not supported")
    return label_mapper


def get_coco_voc_2_pascal_8_label_mapper() -> dict:
    """
    Get the label mapper for the dataset
    Args:
        mapper_name: name of the mapper
    Returns:
        label_mapper: label mapper for the dataset
    """
    label_mapper = {0: 0, 5: 1, 9: 2, 11: 3, 15: 4, 16: 5, 18: 6, 20: 7}
    return label_mapper


def apply_label_map(label_image: np.ndarray, label_map: dict) -> np.ndarray:
    """
    Apply the label map to the labels
    Args:
        label_map: label mapper
        labels: labels to be mapped
    Returns:
        mapped_labels: mapped labels
    """
    mapped_labels = np.zeros_like(label_image)
    for key, value in label_map.items():
        mapped_labels[label_image == key] = value
    return mapped_labels


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


def get_ade20k_labels(bgr=False):
    color_map = np.array(
        [
            [0, 0, 0],
            [120, 120, 120],
            [180, 120, 120],
            [6, 230, 230],
            [80, 50, 50],
            [4, 200, 3],
            [120, 120, 80],
            [140, 140, 140],
            [204, 5, 255],
            [230, 230, 230],
            [4, 250, 7],
            [224, 5, 255],
            [235, 255, 7],
            [150, 5, 61],
            [120, 120, 70],
            [8, 255, 51],
            [255, 6, 82],
            [143, 255, 140],
            [204, 255, 4],
            [255, 51, 7],
            [204, 70, 3],
            [0, 102, 200],
            [61, 230, 250],
            [255, 6, 51],
            [11, 102, 255],
            [255, 7, 71],
            [255, 9, 224],
            [9, 7, 230],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [7, 255, 224],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [255, 122, 8],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
            [31, 255, 0],
            [255, 31, 0],
            [255, 224, 0],
            [153, 255, 0],
            [0, 0, 255],
            [255, 71, 0],
            [0, 235, 255],
            [0, 173, 255],
            [31, 0, 255],
            [11, 200, 200],
            [255, 82, 0],
            [0, 255, 245],
            [0, 61, 255],
            [0, 255, 112],
            [0, 255, 133],
            [255, 0, 0],
            [255, 163, 0],
            [255, 102, 0],
            [194, 255, 0],
            [0, 143, 255],
            [51, 255, 0],
            [0, 82, 255],
            [0, 255, 41],
            [0, 255, 173],
            [10, 0, 255],
            [173, 255, 0],
            [0, 255, 153],
            [255, 92, 0],
            [255, 0, 255],
            [255, 0, 245],
            [255, 0, 102],
            [255, 173, 0],
            [255, 0, 20],
            [255, 184, 184],
            [0, 31, 255],
            [0, 255, 61],
            [0, 71, 255],
            [255, 0, 204],
            [0, 255, 194],
            [0, 255, 82],
            [0, 10, 255],
            [0, 112, 255],
            [51, 0, 255],
            [0, 194, 255],
            [0, 122, 255],
            [0, 255, 163],
            [255, 153, 0],
            [0, 255, 10],
            [255, 112, 0],
            [143, 255, 0],
            [82, 0, 255],
            [163, 255, 0],
            [255, 235, 0],
            [8, 184, 170],
            [133, 0, 255],
            [0, 255, 92],
            [184, 0, 255],
            [255, 0, 31],
            [0, 184, 255],
            [0, 214, 255],
            [255, 0, 112],
            [92, 255, 0],
            [0, 224, 255],
            [112, 224, 255],
            [70, 184, 160],
            [163, 0, 255],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [255, 0, 143],
            [0, 255, 235],
            [133, 255, 0],
            [255, 0, 235],
            [245, 0, 255],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 41, 255],
            [0, 255, 204],
            [41, 0, 255],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [122, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [0, 133, 255],
            [255, 214, 0],
            [25, 194, 194],
            [102, 255, 0],
            [92, 0, 255],
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map


# Source: https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation/utils/colormaps.py
def get_pascal_labels_names():
    return [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]


def get_pascal_8_labels_names():
    return [
        "background",
        "bottle",
        "chair",
        "diningtable",
        "person",
        "potted plant",
        "sofa",
        "tv/monitor",
    ]


def get_ade20k_labels_names():
    return [
        "",
        "wall",
        "building;edifice",
        "sky",
        "floor;flooring",
        "tree",
        "ceiling",
        "road;route",
        "bed",
        "windowpane;window",
        "grass",
        "cabinet",
        "sidewalk;pavement",
        "person;individual;someone;somebody;mortal;soul",
        "earth;ground",
        "door;double;door",
        "table",
        "mountain;mount",
        "plant;flora;plant;life",
        "curtain;drape;drapery;mantle;pall",
        "chair",
        "car;auto;automobile;machine;motorcar",
        "water",
        "painting;picture",
        "sofa;couch;lounge",
        "shelf",
        "house",
        "sea",
        "mirror",
        "rug;carpet;carpeting",
        "field",
        "armchair",
        "seat",
        "fence;fencing",
        "desk",
        "rock;stone",
        "wardrobe;closet;press",
        "lamp",
        "bathtub;bathing;tub;bath;tub",
        "railing;rail",
        "cushion",
        "base;pedestal;stand",
        "box",
        "column;pillar",
        "signboard;sign",
        "chest;of;drawers;chest;bureau;dresser",
        "counter",
        "sand",
        "sink",
        "skyscraper",
        "fireplace;hearth;open;fireplace",
        "refrigerator;icebox",
        "grandstand;covered;stand",
        "path",
        "stairs;steps",
        "runway",
        "case;display;case;showcase;vitrine",
        "pool;table;billiard;table;snooker;table",
        "pillow",
        "screen;door;screen",
        "stairway;staircase",
        "river",
        "bridge;span",
        "bookcase",
        "blind;screen",
        "coffee;table;cocktail;table",
        "toilet;can;commode;crapper;pot;potty;stool;throne",
        "flower",
        "book",
        "hill",
        "bench",
        "countertop",
        "stove;kitchen;stove;range;kitchen;range;cooking;stove",
        "palm;palm;tree",
        "kitchen;island",
        "computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system",
        "swivel;chair",
        "boat",
        "bar",
        "arcade;machine",
        "hovel;hut;hutch;shack;shanty",
        "bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle",
        "towel",
        "light;light;source",
        "truck;motortruck",
        "tower",
        "chandelier;pendant;pendent",
        "awning;sunshade;sunblind",
        "streetlight;street;lamp",
        "booth;cubicle;stall;kiosk",
        "television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box",
        "airplane;aeroplane;plane",
        "dirt;track",
        "apparel;wearing;apparel;dress;clothes",
        "pole",
        "land;ground;soil",
        "bannister;banister;balustrade;balusters;handrail",
        "escalator;moving;staircase;moving;stairway",
        "ottoman;pouf;pouffe;puff;hassock",
        "bottle",
        "buffet;counter;sideboard",
        "poster;posting;placard;notice;bill;card",
        "stage",
        "van",
        "ship",
        "fountain",
        "conveyer;belt;conveyor;belt;conveyer;conveyor;transporter",
        "canopy",
        "washer;automatic;washer;washing;machine",
        "plaything;toy",
        "swimming;pool;swimming;bath;natatorium",
        "stool",
        "barrel;cask",
        "basket;handbasket",
        "waterfall;falls",
        "tent;collapsible;shelter",
        "bag",
        "minibike;motorbike",
        "cradle",
        "oven",
        "ball",
        "food;solid;food",
        "step;stair",
        "tank;storage;tank",
        "trade;name;brand;name;brand;marque",
        "microwave;microwave;oven",
        "pot;flowerpot",
        "animal;animate;being;beast;brute;creature;fauna",
        "bicycle;bike;wheel;cycle",
        "lake",
        "dishwasher;dish;washer;dishwashing;machine",
        "screen;silver;screen;projection;screen",
        "blanket;cover",
        "sculpture",
        "hood;exhaust;hood",
        "sconce",
        "vase",
        "traffic;light;traffic;signal;stoplight",
        "tray",
        "ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin",
        "fan",
        "pier;wharf;wharfage;dock",
        "crt;screen",
        "plate",
        "monitor;monitoring;device",
        "bulletin;board;notice;board",
        "shower",
        "radiator",
        "glass;drinking;glass",
        "clock",
        "flag",
    ]


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


def get_binary_labels(bgr=False):

    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [255, 20, 20],  # 1=foreground
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
    rgb_image = np.zeros((class_image.shape[0], class_image.shape[1], 3), dtype=np.uint8)
    for class_label, class_color in enumerate(class_colors):
        mask = class_image == class_label
        rgb_image[mask] = class_color
    return rgb_image


def label2rgb(label_map, label_colors):  # Not related to sent info, only display
    num_classes = label_colors.shape[0]
    label_colors = label_colors[:num_classes, :]
    rgb_label_map = class_to_rgb(label_map, label_colors)

    return rgb_label_map


def rgb2label(self, rgb_label_map, label_colors):
    num_classes = label_colors.shape[0]
    label_colors = label_colors[:num_classes, :]
    # Save img
    label_map = rgb_to_class(rgb_label_map, label_colors)
    return label_map


# Color map used in ADE20K dataset
ADE20K_COLOR_MAP = np.array(
    [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]
)

# Generic color map from COCO dataset
COCO_COLOR_MAP = np.array(
    [
        [0, 0, 0],
        [220, 20, 60],
        [119, 11, 32],
        [0, 0, 142],
        [0, 0, 230],
        [106, 0, 228],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 70],
        [0, 0, 192],
        [250, 170, 30],
        [100, 170, 30],
        [220, 220, 0],
        [175, 116, 175],
        [250, 0, 30],
        [165, 42, 42],
        [255, 77, 255],
        [0, 226, 252],
        [182, 182, 255],
        [0, 82, 0],
        [120, 166, 157],
        [110, 76, 0],
        [174, 57, 255],
        [199, 100, 0],
        [72, 0, 118],
        [255, 179, 240],
        [0, 125, 92],
        [209, 0, 151],
        [188, 208, 182],
        [0, 220, 176],
        [255, 99, 164],
        [92, 0, 73],
        [133, 129, 255],
        [78, 180, 255],
        [0, 228, 0],
        [174, 255, 243],
        [45, 89, 255],
        [134, 134, 103],
        [145, 148, 174],
        [255, 208, 186],
        [197, 226, 255],
        [171, 134, 1],
        [109, 63, 54],
        [207, 138, 255],
        [151, 0, 95],
        [9, 80, 61],
        [84, 105, 51],
        [74, 65, 105],
        [166, 196, 102],
        [208, 195, 210],
        [255, 109, 65],
        [0, 143, 149],
        [179, 0, 194],
        [209, 99, 106],
        [5, 121, 0],
        [227, 255, 205],
        [147, 186, 208],
        [153, 69, 1],
        [3, 95, 161],
        [163, 255, 0],
        [119, 0, 170],
        [0, 182, 199],
        [0, 165, 120],
        [183, 130, 88],
        [95, 32, 0],
        [130, 114, 135],
        [110, 129, 133],
        [166, 74, 118],
        [219, 142, 185],
        [79, 210, 114],
        [178, 90, 62],
        [65, 70, 15],
        [127, 167, 115],
        [59, 105, 106],
        [142, 108, 45],
        [196, 172, 0],
        [95, 54, 80],
        [128, 76, 255],
        [201, 57, 1],
        [246, 0, 122],
        [191, 162, 208],
        [255, 255, 128],
        [147, 211, 203],
        [150, 100, 100],
        [168, 171, 172],
        [146, 112, 198],
        [210, 170, 100],
        [92, 136, 89],
        [218, 88, 184],
        [241, 129, 0],
        [217, 17, 255],
        [124, 74, 181],
        [70, 70, 70],
        [255, 228, 255],
        [154, 208, 0],
        [193, 0, 92],
        [76, 91, 113],
        [255, 180, 195],
        [106, 154, 176],
        [230, 150, 140],
        [60, 143, 255],
        [128, 64, 128],
        [92, 82, 55],
        [254, 212, 124],
        [73, 77, 174],
        [255, 160, 98],
        [255, 255, 255],
        [104, 84, 109],
        [169, 164, 131],
        [225, 199, 255],
        [137, 54, 74],
        [135, 158, 223],
        [7, 246, 231],
        [107, 255, 200],
        [58, 41, 149],
        [183, 121, 142],
        [255, 73, 97],
        [107, 142, 35],
        [190, 153, 153],
        [146, 139, 141],
        [70, 130, 180],
        [134, 199, 156],
        [209, 226, 140],
        [96, 36, 108],
        [96, 96, 96],
        [64, 170, 64],
        [152, 251, 152],
        [208, 229, 228],
        [206, 186, 171],
        [152, 161, 64],
        [116, 112, 0],
        [0, 114, 143],
        [102, 102, 156],
        [250, 141, 255],
    ]
)
