import torch
import yaml
from torch import nn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
import numpy as np
from torchvision.ops.boxes import nms as nms_torch

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

class EfficientDetWithPostProcessing(nn.Module):
    def __init__(self, backbone_model, threshold=0.5, nms_threshold=0.5):
        super(EfficientDetWithPostProcessing, self).__init__()
        self.backbone = backbone_model
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        
    def forward(self, x):
        # Get raw model outputs
        features, regression, classification, anchors = self.backbone(x)
        
        # Transform regression deltas to actual bounding boxes
        boxes = self.regressBoxes(anchors, regression)
        
        # Clip boxes to image boundaries
        boxes = self.clipBoxes(boxes, x)
        
        # Apply sigmoid to classification scores (if not already applied)
        # Note: The classifier in the backbone already applies sigmoid, so this might be redundant
        scores = classification
        
        batch_size = x.shape[0]
        
        # Process each image in the batch
        batch_boxes = []
        batch_scores = []
        batch_classes = []
        
        for batch_idx in range(batch_size):
            # Get boxes and scores for current image
            img_boxes = boxes[batch_idx]  # Shape: [num_anchors, 4]
            img_scores = scores[batch_idx]  # Shape: [num_anchors, num_classes]
            
            # Get max scores and corresponding class indices
            max_scores, class_indices = torch.max(img_scores, dim=1)
            
            # Filter by confidence threshold
            valid_indices = max_scores > self.threshold
            
            if valid_indices.sum() == 0:
                # No valid detections
                batch_boxes.append(torch.zeros((0, 4), dtype=x.dtype, device=x.device))
                batch_scores.append(torch.zeros((0,), dtype=x.dtype, device=x.device))
                batch_classes.append(torch.zeros((0,), dtype=torch.long, device=x.device))
                continue
                
            filtered_boxes = img_boxes[valid_indices]
            filtered_scores = max_scores[valid_indices]
            filtered_classes = class_indices[valid_indices]
            
            # Apply NMS
            keep_indices = nms_torch(filtered_boxes, filtered_scores, self.nms_threshold)
            
            final_boxes = filtered_boxes[keep_indices]
            final_scores = filtered_scores[keep_indices]
            final_classes = filtered_classes[keep_indices]
            
            batch_boxes.append(final_boxes)
            batch_scores.append(final_scores)
            batch_classes.append(final_classes)
        
        # For ONNX export, we need fixed-size outputs
        # Let's return the results for the first image only (typical for inference)
        if len(batch_boxes[0]) > 0:
            return batch_boxes[0], batch_scores[0], batch_classes[0]
        else:
            # Return empty tensors with proper shape
            return (torch.zeros((0, 4), dtype=x.dtype, device=x.device),
                   torch.zeros((0,), dtype=x.dtype, device=x.device),
                   torch.zeros((0,), dtype=torch.long, device=x.device))

class EfficientDetWithFixedOutputs(nn.Module):
    """
    Version with fixed-size outputs for better ONNX compatibility
    """
    def __init__(self, backbone_model, threshold=0.5, nms_threshold=0.5, max_detections=100):
        super(EfficientDetWithFixedOutputs, self).__init__()
        self.backbone = backbone_model
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
    def forward(self, x):
        # Get raw model outputs
        features, regression, classification, anchors = self.backbone(x)
        
        # Transform regression deltas to actual bounding boxes
        boxes = self.regressBoxes(anchors, regression)
        
        # Clip boxes to image boundaries
        boxes = self.clipBoxes(boxes, x)
        
        # Process first image in batch (typical for inference)
        img_boxes = boxes[0]  # Shape: [num_anchors, 4]
        img_scores = classification[0]  # Shape: [num_anchors, num_classes]
        
        # Get max scores and corresponding class indices
        max_scores, class_indices = torch.max(img_scores, dim=1)
        
        # Filter by confidence threshold
        valid_indices = max_scores > self.threshold
        
        if valid_indices.sum() == 0:
            # No valid detections - return zeros
            output_boxes = torch.zeros((self.max_detections, 4), dtype=x.dtype, device=x.device)
            output_scores = torch.zeros((self.max_detections,), dtype=x.dtype, device=x.device)
            output_classes = torch.zeros((self.max_detections,), dtype=torch.long, device=x.device)
            num_detections = torch.tensor(0, dtype=torch.long, device=x.device)
            return output_boxes, output_scores, output_classes, num_detections
            
        filtered_boxes = img_boxes[valid_indices]
        filtered_scores = max_scores[valid_indices]
        filtered_classes = class_indices[valid_indices]
        
        # Apply NMS
        keep_indices = nms_torch(filtered_boxes, filtered_scores, self.nms_threshold)
        
        final_boxes = filtered_boxes[keep_indices]
        final_scores = filtered_scores[keep_indices]
        final_classes = filtered_classes[keep_indices]
        
        # Limit to max_detections
        num_detections = min(len(final_boxes), self.max_detections)
        
        # Prepare fixed-size outputs
        output_boxes = torch.zeros((self.max_detections, 4), dtype=x.dtype, device=x.device)
        output_scores = torch.zeros((self.max_detections,), dtype=x.dtype, device=x.device)
        output_classes = torch.zeros((self.max_detections,), dtype=torch.long, device=x.device)
        
        if num_detections > 0:
            output_boxes[:num_detections] = final_boxes[:num_detections]
            output_scores[:num_detections] = final_scores[:num_detections]
            output_classes[:num_detections] = final_classes[:num_detections]
        
        num_detections_tensor = torch.tensor(num_detections, dtype=torch.long, device=x.device)
        
        return output_boxes, output_scores, output_classes, num_detections_tensor

def export_model():
    device = torch.device('cpu')
    params = Params('projects/abhil.yml')
    
    # Load the backbone model
    backbone_model = EfficientDetBackbone(
        num_classes=len(params.obj_list), 
        compound_coef=2, 
        onnx_export=True,
        ratios=eval(params.anchors_ratios), 
        scales=eval(params.anchors_scales)
    ).to(device)

    backbone_model.backbone_net.model.set_swish(memory_efficient=False)
    
    # Load trained weights
    backbone_model.load_state_dict(torch.load(
        'F:/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhil/efficientdet-d2_49_5784.pth'
    ))
    
    backbone_model.eval()
    
    # Create wrapper with postprocessing
    # Option 1: Variable output size (might be less compatible with some ONNX runtimes)
    model_with_postprocessing = EfficientDetWithPostProcessing(
        backbone_model, 
        threshold=0.5, 
        nms_threshold=0.5
    )
    
    # Option 2: Fixed output size (better ONNX compatibility)
    model_with_fixed_outputs = EfficientDetWithFixedOutputs(
        backbone_model, 
        threshold=0.5, 
        nms_threshold=0.5, 
        max_detections=100
    )
    
    dummy_input = torch.randn((1, 3, 768, 768), dtype=torch.float32).to(device)
    
    # Export Option 1: Variable outputs
    print("Exporting model with variable outputs...")
    torch.onnx.export(
        model_with_postprocessing, 
        dummy_input, 
        "efficientdet-d2-with-postprocessing.onnx", 
        verbose=False, 
        input_names=['input'], 
        output_names=['boxes', 'scores', 'classes'],
        opset_version=11,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'num_detections'},
            'scores': {0: 'num_detections'},
            'classes': {0: 'num_detections'}
        }
    )
    
    # Export Option 2: Fixed outputs (recommended)
    print("Exporting model with fixed outputs...")
    torch.onnx.export(
        model_with_fixed_outputs, 
        dummy_input, 
        "efficientdet-d2-with-postprocessing-fixed.onnx", 
        verbose=False, 
        input_names=['input'], 
        output_names=['boxes', 'scores', 'classes', 'num_detections'],
        opset_version=11,
        dynamic_axes={
            'input': {0: 'batch_size'}
        }
    )
    
    print("Export completed!")
    print("Files created:")
    print("1. efficientdet-d2-with-postprocessing.onnx (variable output size)")
    print("2. efficientdet-d2-with-postprocessing-fixed.onnx (fixed output size)")
    print("\nModel outputs:")
    print("- boxes: [max_detections, 4] - bounding boxes in format [x1, y1, x2, y2]")
    print("- scores: [max_detections] - confidence scores")
    print("- classes: [max_detections] - class indices")
    print("- num_detections: scalar - number of valid detections")

if __name__ == "__main__":
    export_model()
