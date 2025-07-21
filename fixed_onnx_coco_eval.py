"""
Final Fixed COCO-Style Evaluation for ONNX EfficientDet models
Specifically handles the fixed-size output format: [100, 4], [100], [100], []

Usage:
python eval_onnx_coco_final.py -p abhil -w efficientdet-d2-with-postprocessing-fixed.onnx -c 2
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
    Preprocess image for ONNX model inference
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
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
        print(f"Error preprocessing {image_path}: {e}")
        return None, None

def postprocess_detections(boxes, scores, classes, num_detections, meta):
    """
    Transform detections back to original image coordinates
    Handles fixed-size arrays [100, 4], [100], [100] with scalar num_detections
    """
    try:
        # Handle num_detections - it should be a scalar
        if isinstance(num_detections, np.ndarray):
            if num_detections.shape == ():
                # Scalar
                num_det = int(num_detections.item())
            elif num_detections.shape == (1,):
                # Single element array
                num_det = int(num_detections[0])
            else:
                print(f"Unexpected num_detections shape: {num_detections.shape}")
                num_det = 0
        else:
            num_det = int(num_detections)
        
        if num_det <= 0:
            return [], [], []
        
        # Extract only the valid detections
        valid_boxes = boxes[:num_det].copy()
        valid_scores = scores[:num_det].copy()
        valid_classes = classes[:num_det].copy()
        
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

def evaluate_onnx_coco(onnx_model_path, img_path, set_name, image_ids, coco, params, compound_coef):
    """
    Evaluate ONNX model on COCO dataset
    """
    results = []
    
    # Load ONNX model
    print(f"Loading ONNX model: {onnx_model_path}")
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    target_size = input_sizes[compound_coef]
    
    print(f"Target size: {target_size}")
    print(f"Classes: {params.obj_list}")
    print(f"Processing {len(image_ids)} images...")
    
    # Counters
    processed_count = 0
    error_count = 0
    detection_count = 0
    
    for image_id in tqdm(image_ids, desc="Processing images"):
        try:
            image_info = coco.loadImgs(image_id)[0]
            image_path = img_path + image_info['file_name']
            
            # Check if image exists
            if not os.path.exists(image_path):
                error_count += 1
                continue
            
            # Preprocess image
            input_image, meta = preprocess_image(
                image_path, target_size, params.mean, params.std
            )
            
            if input_image is None:
                error_count += 1
                continue
            
            # Run inference
            input_name = ort_session.get_inputs()[0].name
            ort_inputs = {input_name: input_image}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Parse outputs - expecting exactly 4 outputs
            if len(ort_outputs) != 4:
                print(f"⚠️  Expected 4 outputs, got {len(ort_outputs)}")
                error_count += 1
                continue
            
            boxes, scores, classes, num_detections = ort_outputs
            
            # Transform back to original coordinates
            valid_boxes, valid_scores, valid_classes = postprocess_detections(
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
                    print(f"Error processing detection {i}: {e}")
                    continue
            
            processed_count += 1
                    
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            error_count += 1
            continue
    
    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Total images: {len(image_ids)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Total detections: {detection_count}")
    
    if len(results) == 0:
        print("❌ No valid detections found!")
        print("Possible solutions:")
        print("  1. Lower confidence threshold in model export")
        print("  2. Check if model was trained properly")
        print("  3. Verify image preprocessing matches training")
        return None
    
    # Save results
    filepath = f'{set_name}_onnx_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ Results saved to: {filepath}")
    return filepath

def run_coco_eval(coco_gt, image_ids, pred_json_path):
    """
    Run COCO evaluation
    """
    try:
        # Load results in COCO evaluation tool
        coco_pred = coco_gt.loadRes(pred_json_path)

        # Run COCO evaluation
        print('\n' + '='*50)
        print('COCO EVALUATION RESULTS')
        print('='*50)
        coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return coco_eval
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None

def main():
    parser = argparse.ArgumentParser('ONNX EfficientDet COCO Evaluation (Final)')
    parser.add_argument('-p', '--project', type=str, default='abhil', help='Project name')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='Model compound coefficient')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--max_images', type=int, default=100000, help='Maximum number of images to evaluate')
    parser.add_argument('--debug', action='store_true', help='Debug mode - process only first 10 images')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.weights):
        print(f"❌ ONNX model not found: {args.weights}")
        return
    
    project_file = f'projects/{args.project}.yml'
    if not os.path.exists(project_file):
        print(f"❌ Project file not found: {project_file}")
        return
    
    # Load parameters
    try:
        params = Params(project_file)
    except Exception as e:
        print(f"❌ Error loading project parameters: {e}")
        return
    
    # Dataset paths
    SET_NAME = params.val_set
    VAL_GT = f'datasets/{params.project_name}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params.project_name}/{SET_NAME}/'
    
    # Validate dataset
    if not os.path.exists(VAL_GT):
        print(f"❌ Dataset annotations not found: {VAL_GT}")
        return
    
    if not os.path.exists(VAL_IMGS):
        print(f"❌ Dataset images not found: {VAL_IMGS}")
        return
    
    print(f"✓ Project: {args.project}")
    print(f"✓ Model: {args.weights}")
    print(f"✓ Dataset: {VAL_GT}")
    print(f"✓ Images: {VAL_IMGS}")
    
    # Load COCO dataset
    try:
        coco_gt = COCO(VAL_GT)
        image_ids = coco_gt.getImgIds()[:args.max_images]
        print(f"✓ Loaded {len(image_ids)} images")
    except Exception as e:
        print(f"❌ Error loading COCO dataset: {e}")
        return
    
    # Debug mode
    if args.debug:
        print("\n🔍 DEBUG MODE: Processing first 10 images only")
        image_ids = image_ids[:10]
    
    # Run evaluation
    pred_json_path = evaluate_onnx_coco(
        args.weights, VAL_IMGS, SET_NAME, image_ids, coco_gt, params, args.compound_coef
    )
    
    if pred_json_path is None:
        print("❌ Evaluation failed - no results generated")
        return
    
    # Compute metrics
    coco_eval = run_coco_eval(coco_gt, image_ids, pred_json_path)
    
    if coco_eval is not None:
        # Print final summary
        print("\n" + "="*60)
        print("FINAL EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {os.path.basename(args.weights)}")
        print(f"Dataset: {SET_NAME}")
        print(f"Images: {len(image_ids)}")
        print(f"mAP @ IoU=0.50:0.95: {coco_eval.stats[0]:.4f}")
        print(f"mAP @ IoU=0.50: {coco_eval.stats[1]:.4f}")
        print(f"mAP @ IoU=0.75: {coco_eval.stats[2]:.4f}")
        print(f"mAP (small): {coco_eval.stats[3]:.4f}")
        print(f"mAP (medium): {coco_eval.stats[4]:.4f}")
        print(f"mAP (large): {coco_eval.stats[5]:.4f}")
        print("="*60)
    else:
        print("❌ Failed to compute evaluation metrics")

if __name__ == '__main__':
    main()
