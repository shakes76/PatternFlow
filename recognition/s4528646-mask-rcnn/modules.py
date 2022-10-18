from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision

def get_model(num_classes=3):
    """
    Instantiate a PyTorch model based on Mask-RCNN with ResNet backbone,
    pretrained on COCO data set. 
    """
    model = maskrcnn_resnet50_fpn(
        weights="DEFAULT",
        min_size=1000,
        max_size=4288,
        # box_detections_per_img=1,
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes,
                                                       )
    return model
