"""
Fixed COCO-Style Evaluation for ONNX EfficientDet models with robust error handling

Usage:
python eval_onnx_coco_fixed.py -p abhil -w efficientdet-d2-with-postprocessing-fixed.onnx -c 2
"""

import json
import os
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import cv2

import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def preprocess_image(image_path, target_size, mean, std):
    """
    Preprocess image for ONNX model inference with error handling
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image: {image_path}")
            return None, None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Resize image while maintaining aspect ratio
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Pad image to target size
        padded_image = np.ones((target_size, target_size, 3), dtype=np.uint8) * 114
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        
        # Normalize
        padded_image = padded_image.astype(np.float32) / 255.0
        padded_image = (padded_image - np.array(mean)) / np.array(std)
        
        # Change from HWC to CHW
        padded_image = np.transpose(padded_image, (2, 0, 1))
        
        # Add batch dimension
        padded_image = np.expand_dims(padded_image, axis=0)
        
        # Return preprocessing metadata for inverse transformation
        meta = {
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'original_width': original_width,
            'original_height': original_height
        }
        
        return padded_image, meta
    
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None, None

def safe_postprocess_detections(boxes, scores, classes, num_detections, meta):
    """
    Transform detections back to original image coordinates with safety checks
    """
    try:
        # Handle different input types
        if isinstance(num_detections, np.ndarray):
            if num_detections.size == 1:
                num_detections = int(num_detections.item())
            else:
                num_detections = int(num_detections[0]) if len(num_detections) > 0 else 0
        elif isinstance(num_detections, (int, float)):
            num_detections = int(num_detections)
        else:
            num_detections = 0
        
        if num_detections <= 0:
            return [], [], []
        
        # Ensure we don't access out of bounds
        max_detections = min(num_detections, len(boxes), len(scores), len(classes))
        
        if max_detections <= 0:
            return [], [], []
        
        # Get valid detections
        valid_boxes = boxes[:max_detections]
        valid_scores = scores[:max_detections]
        valid_classes = classes[:max_detections]
        
        # Ensure boxes have the right shape
        if len(valid_boxes.shape) == 1:
            valid_boxes = valid_boxes.reshape(-1, 4)
        
        # Transform coordinates back to original image space
        scale = meta['scale']
        x_offset = meta['x_offset']
        y_offset = meta['y_offset']
        
        # Remove padding offset
        valid_boxes[:, 0] -= x_offset  # x1
        valid_boxes[:, 1] -= y_offset  # y1
        valid_boxes[:, 2] -= x_offset  # x2
        valid_boxes[:, 3] -= y_offset  # y2
        
        # Scale back to original size
        valid_boxes /= scale
        
        # Clip to image boundaries
        valid_boxes[:, 0] = np.clip(valid_boxes[:, 0], 0, meta['original_width'])
        valid_boxes[:, 1] = np.clip(valid_boxes[:, 1], 0, meta['original_height'])
        valid_boxes[:, 2] = np.clip(valid_boxes[:, 2], 0, meta['original_width'])
        valid_boxes[:, 3] = np.clip(valid_boxes[:, 3], 0, meta['original_height'])
        
        return valid_boxes, valid_scores, valid_classes
    
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        return [], [], []

def safe_onnx_inference(ort_session, input_image):
    """
    Run ONNX inference with error handling
    """
    try:
        # Run inference
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: input_image}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Parse outputs based on number of outputs
        if len(ort_outputs) == 4:
            # Fixed output format: boxes, scores, classes, num_detections
            boxes, scores, classes, num_detections = ort_outputs
            
            # Ensure proper shapes
            if len(boxes.shape) > 2:
                boxes = boxes.squeeze()
            if len(scores.shape) > 1:
                scores = scores.squeeze()
            if len(classes.shape) > 1:
                classes = classes.squeeze()
                
            return boxes, scores, classes, num_detections
            
        elif len(ort_outputs) == 3:
            # Variable output format: boxes, scores, classes
            boxes, scores, classes = ort_outputs
            num_detections = len(boxes) if len(boxes.shape) > 1 else 0
            
            # Ensure proper shapes
            if len(boxes.shape) > 2:
                boxes = boxes.squeeze()
            if len(scores.shape) > 1:
                scores = scores.squeeze()
            if len(classes.shape) > 1:
                classes = classes.squeeze()
                
            return boxes, scores, classes, num_detections
        else:
            print(f"Unexpected number of outputs: {len(ort_outputs)}")
            return None, None, None, 0
            
    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        return None, None, None, 0

def evaluate_onnx_coco(onnx_model_path, img_path, set_name, image_ids, coco, params, compound_coef):
    """
    Evaluate ONNX model on COCO dataset with robust error handling
    """
    results = []
    
    # Load ONNX model
    print(f"Loading ONNX model: {onnx_model_path}")
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None
    
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    target_size = input_sizes[compound_coef]
    
    print(f"Target size: {target_size}")
    print(f"Processing {len(image_ids)} images...")
    
    # Counters for tracking
    processed_count = 0
    error_count = 0
    detection_count = 0
    
    for image_id in tqdm(image_ids, desc="Processing images"):
        try:
            image_info = coco.loadImgs(image_id)[0]
            image_path = img_path + image_info['file_name']
            
            # Check if image file exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                error_count += 1
                continue
            
            # Preprocess image
            input_image, meta = preprocess_image(
                image_path, 
                target_size, 
                params.mean, 
                params.std
            )
            
            if input_image is None:
                error_count += 1
                continue
            
            # Run inference
            boxes, scores, classes, num_detections = safe_onnx_inference(ort_session, input_image)
            
            if boxes is None:
                error_count += 1
                continue
            
            # Transform back to original image coordinates
            valid_boxes, valid_scores, valid_classes = safe_postprocess_detections(
                boxes, scores, classes, num_detections, meta
            )
            
            if len(valid_boxes) == 0:
                processed_count += 1
                continue
            
            # Convert to COCO format
            for i in range(len(valid_boxes)):
                try:
                    box = valid_boxes[i]
                    score = float(valid_scores[i])
                    class_id = int(valid_classes[i])
                    
                    # Validate class_id
                    if class_id < 0 or class_id >= len(params.obj_list):
                        print(f"Invalid class_id: {class_id}, skipping detection")
                        continue
                    
                    # Convert from [x1, y1, x2, y2] to [x, y, width, height]
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Validate box dimensions
                    if width <= 0 or height <= 0:
                        continue
                    
                    result = {
                        'image_id': image_id,
                        'category_id': class_id + 1,  # COCO uses 1-based indexing
                        'bbox': [float(x1), float(y1), float(width), float(height)],
                        'score': score
                    }
                    results.append(result)
                    detection_count += 1
                    
                except Exception as e:
                    print(f"Error processing detection {i} in image {image_path}: {e}")
                    continue
            
            processed_count += 1
                    
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            error_count += 1
            continue
    
    # Print processing summary
    print(f"\nProcessing Summary:")
    print(f"Total images: {len(image_ids)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Total detections: {detection_count}")
    
    if len(results) == 0:
        print("⚠️  No valid detections found. Possible issues:")
        print("   - Model confidence threshold too high")
        print("   - Model not detecting any objects")
        print("   - Preprocessing issues")
        print("   - Model format mismatch")
        return None
    
    # Write output
    filepath = f'{set_name}_onnx_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)
    
    print(f"Results saved to: {filepath}")
    return filepath

def _eval(coco_gt, image_ids, pred_json_path):
    """
    Run COCO evaluation
    """
    try:
        # Load results in COCO evaluation tool
        coco_pred = coco_gt.loadRes(pred_json_path)

        # Run COCO evaluation
        print('\nBBox Evaluation Results:')
        coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return coco_eval
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

def main():
    parser = argparse.ArgumentParser('ONNX EfficientDet COCO Evaluation (Fixed)')
    parser.add_argument('-p', '--project', type=str, default='abhil', help='Project name')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='Model compound coefficient')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--max_images', type=int, default=100000, help='Maximum number of images to evaluate')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for first few images')
    
    args = parser.parse_args()
    
    # Load parameters
    project_file = f'projects/{args.project}.yml'
    if not os.path.exists(project_file):
        print(f"Error: Project file not found: {project_file}")
        return
    
    params = Params(project_file)
    
    # Dataset paths
    SET_NAME = params.val_set
    VAL_GT = f'datasets/{params.project_name}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params.project_name}/{SET_NAME}/'
    
    print(f"Project: {args.project}")
    print(f"Model: {args.weights}")
    print(f"Dataset: {VAL_GT}")
    print(f"Images: {VAL_IMGS}")
    
    # Check if files exist
    if not os.path.exists(args.weights):
        print(f"Error: ONNX model not found: {args.weights}")
        return
    
    if not os.path.exists(VAL_GT):
        print(f"Error: Dataset annotations not found: {VAL_GT}")
        return
    
    if not os.path.exists(VAL_IMGS):
        print(f"Error: Dataset images not found: {VAL_IMGS}")
        return
    
    # Load COCO dataset
    try:
        coco_gt = COCO(VAL_GT)
        image_ids = coco_gt.getImgIds()[:args.max_images]
    except Exception as e:
        print(f"Error loading COCO dataset: {e}")
        return
    
    print(f"Number of images: {len(image_ids)}")
    print(f"Classes: {params.obj_list}")
    
    # Debug mode - process only first few images
    if args.debug:
        print("\n🔍 DEBUG MODE: Processing first 5 images only")
        image_ids = image_ids[:5]
    
    # Run evaluation
    pred_json_path = evaluate_onnx_coco(
        args.weights, VAL_IMGS, SET_NAME, image_ids, coco_gt, params, args.compound_coef
    )
    
    if pred_json_path is None:
        print("❌ Evaluation failed")
        return
    
    # Compute metrics
    coco_eval = _eval(coco_gt, image_ids, pred_json_path)
    
    if coco_eval is not None:
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {args.weights}")
        print(f"Dataset: {SET_NAME}")
        print(f"Images processed: {len(image_ids)}")
        print(f"mAP @ IoU=0.50:0.95: {coco_eval.stats[0]:.4f}")
        print(f"mAP @ IoU=0.50: {coco_eval.stats[1]:.4f}")
        print(f"mAP @ IoU=0.75: {coco_eval.stats[2]:.4f}")
    else:
        print("❌ Failed to compute evaluation metrics")

if __name__ == '__main__':
    main()
