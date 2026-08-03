from enum import Enum

WSI_EXTs = [".tiff", ".tif", ".mrxs"]
ANNOTATION_EXTs = [".xml"]

class AnnotationDataType(Enum):
    PIXELS_LVL_0 = 1
    UM_LVL_0 = 2

