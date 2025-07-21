"""
COCO-Style Evaluation for ONNX EfficientDet models with postprocessing

Usage:
python eval_onnx_coco.py -p abhil -w efficientdet-d2-with-postprocessing-fixed.onnx -c 2
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
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
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

def postprocess_detections(boxes, scores, classes, num_detections, meta):
    """
    Transform detections back to original image coordinates
    """
    if num_detections == 0:
        return [], [], []
    
    # Get valid detections
    valid_boxes = boxes[:num_detections]
    valid_scores = scores[:num_detections]
    valid_classes = classes[:num_detections]
    
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

def evaluate_onnx_coco(onnx_model_path, img_path, set_name, image_ids, coco, params, compound_coef):
    """
    Evaluate ONNX model on COCO dataset
    """
    results = []
    
    # Load ONNX model
    print(f"Loading ONNX model: {onnx_model_path}")
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        raise
    
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    target_size = input_sizes[compound_coef]
    
    print(f"Target size: {target_size}")
    print(f"Processing {len(image_ids)} images...")
    
    # Track statistics
    processed_count = 0
    error_count = 0
    detection_count = 0
    
    for i, image_id in enumerate(tqdm(image_ids, desc="Processing images")):
        try:
            # Load image info from COCO
            image_info_list = coco.loadImgs(image_id)
            if not image_info_list:
                error_count += 1
                continue
            
            image_info = image_info_list[0]
            image_path = img_path + image_info['file_name']
            
            # Check if image file exists
            if not os.path.exists(image_path):
                error_count += 1
                continue
            
            # Preprocess image
            input_image, meta = preprocess_image(
                image_path, 
                target_size, 
                params.mean, 
                params.std
            )
            
            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: input_image}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Parse outputs
            if len(ort_outputs) == 4:
                # Fixed output format: boxes, scores, classes, num_detections
                boxes, scores, classes, num_detections = ort_outputs
                num_detections = int(num_detections)
            else:
                # Variable output format: boxes, scores, classes
                boxes, scores, classes = ort_outputs
                num_detections = len(boxes)
            
            processed_count += 1
            
            if num_detections == 0:
                continue
            
            # Transform back to original image coordinates
            valid_boxes, valid_scores, valid_classes = postprocess_detections(
                boxes, scores, classes, num_detections, meta
            )
            
            # Convert to COCO format
            for j in range(len(valid_boxes)):
                box = valid_boxes[j]
                score = float(valid_scores[j])
                class_id = int(valid_classes[j])
                
                # Validate class ID
                if class_id < 0 or class_id >= len(params.obj_list):
                    continue
                
                # Convert from [x1, y1, x2, y2] to [x, y, width, height]
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                if width > 0 and height > 0:
                    result = {
                        'image_id': image_id,
                        'category_id': class_id + 1,  # COCO uses 1-based indexing
                        'bbox': [float(x1), float(y1), float(width), float(height)],
                        'score': score
                    }
                    results.append(result)
                    detection_count += 1
                    
        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Only print first 10 errors to avoid spam
                print(f"Error processing image {image_path if 'image_path' in locals() else image_id}: {e}")
            continue
    
    print(f"\nProcessing Summary:")
    print(f"Total images: {len(image_ids)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Total detections: {detection_count}")
    
    if not len(results):
        raise Exception("The model does not provide any valid output. Check your model and dataset.")
    
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
    # Load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # Run COCO evaluation
    print('BBox Evaluation Results:')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval

def main():
    parser = argparse.ArgumentParser('ONNX EfficientDet COCO Evaluation')
    parser.add_argument('-p', '--project', type=str, default='abhil', help='Project name')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='Model compound coefficient')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--max_images', type=int, default=100000, help='Maximum number of images to evaluate')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.weights):
        print(f"Error: ONNX model not found: {args.weights}")
        return
    
    project_file = f'projects/{args.project}.yml'
    if not os.path.exists(project_file):
        print(f"Error: Project file not found: {project_file}")
        return
    
    # Load parameters
    params = Params(project_file)
    
    # Dataset paths
    SET_NAME = params.val_set
    VAL_GT = f'datasets/{params.project_name}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params.project_name}/{SET_NAME}/'
    
    print(f"Project: {args.project}")
    print(f"Model: {args.weights}")
    print(f"Dataset: {VAL_GT}")
    print(f"Images: {VAL_IMGS}")
    
    # Verify dataset files exist
    if not os.path.exists(VAL_GT):
        print(f"Error: Annotations file not found: {VAL_GT}")
        return
    
    if not os.path.exists(VAL_IMGS):
        print(f"Error: Images directory not found: {VAL_IMGS}")
        return
    
    # Load COCO dataset
    try:
        coco_gt = COCO(VAL_GT)
        image_ids = coco_gt.getImgIds()
        
        # Filter out image IDs that don't have corresponding files
        valid_image_ids = []
        for img_id in image_ids[:args.max_images]:
            try:
                img_info_list = coco_gt.loadImgs(img_id)
                if img_info_list:
                    img_info = img_info_list[0]
                    img_path = VAL_IMGS + img_info['file_name']
                    if os.path.exists(img_path):
                        valid_image_ids.append(img_id)
                    elif args.debug:
                        print(f"Debug: Image file missing: {img_path}")
            except Exception as e:
                if args.debug:
                    print(f"Debug: Error loading image ID {img_id}: {e}")
                continue
        
        image_ids = valid_image_ids
        
    except Exception as e:
        print(f"Error loading COCO dataset: {e}")
        return
    
    print(f"Total images in dataset: {len(coco_gt.getImgIds())}")
    print(f"Valid images found: {len(image_ids)}")
    print(f"Classes: {params.obj_list}")
    
    if len(image_ids) == 0:
        print("Error: No valid images found for evaluation!")
        return
    
    # Run evaluation
    try:
        pred_json_path = evaluate_onnx_coco(
            args.weights, VAL_IMGS, SET_NAME, image_ids, coco_gt, params, args.compound_coef
        )
        
        # Compute metrics
        coco_eval = _eval(coco_gt, image_ids, pred_json_path)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
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

if __name__ == '__main__':
    main()
