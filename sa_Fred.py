import torch
import cv2
import json
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from cvStuff import showImg

import supervision as sv

# define a few things
CHECKPOINT_PATH = "models\\sam_vit_h_4b8939.pth"
# CHECKPOINT_PATH = "models\\sam_vit_l_0b3195.pth"
# CHECKPOINT_PATH = "models\\sam_vit_b_01ec64.pth"
IMAGE_PATH = "images\\test\\mito.TIF"

# loading the SAM model into memory
# (ViT-B 91M, ViT-L 308M, ViT-H 636M)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

# segment complete image
mask_generator = SamAutomaticMaskGenerator(sam)
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = mask_generator.generate(image_rgb)

# annotate image
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(result)
annotated_image = mask_annotator.annotate(image_bgr, detections)
showImg("annotated image", annotated_image)
cv2.imwrite("images\\test\\mito_annoH.TIF", annotated_image)
