"""
Debug ONNX model outputs to identify the issue

Usage:
python debug_onnx_outputs.py -w efficientdet-d2-with-postprocessing-fixed.onnx -i test_image.jpg -p abhil
"""

import argparse
import numpy as np
import cv2
import yaml
import os

import onnxruntime as ort

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
    
    return padded_image

def debug_onnx_model(onnx_model_path, image_path, params, compound_coef):
    """
    Debug ONNX model outputs
    """
    # Load ONNX model
    print(f"Loading ONNX model: {onnx_model_path}")
    ort_session = ort.InferenceSession(onnx_model_path)
    
    # Get model info
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    
    print("Inputs:")
    for i, input_info in enumerate(ort_session.get_inputs()):
        print(f"  {i}: {input_info.name} - {input_info.shape} - {input_info.type}")
    
    print("\nOutputs:")
    for i, output_info in enumerate(ort_session.get_outputs()):
        print(f"  {i}: {output_info.name} - {output_info.shape} - {output_info.type}")
    
    # Get target size
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    target_size = input_sizes[compound_coef]
    print(f"\nTarget image size: {target_size}")
    
    # Preprocess image
    print(f"\nProcessing image: {image_path}")
    input_image = preprocess_image(image_path, target_size, params.mean, params.std)
    print(f"Input image shape: {input_image.shape}")
    
    # Run inference
    print(f"\nRunning inference...")
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_image}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print("\n" + "="*50)
    print("MODEL OUTPUTS")
    print("="*50)
    
    print(f"Number of outputs: {len(ort_outputs)}")
    
    for i, output in enumerate(ort_outputs):
        print(f"\nOutput {i}:")
        print(f"  Shape: {output.shape}")
        print(f"  Dtype: {output.dtype}")
        print(f"  Min value: {np.min(output):.6f}")
        print(f"  Max value: {np.max(output):.6f}")
        print(f"  Mean value: {np.mean(output):.6f}")
        
        # Print some sample values
        if output.size <= 20:
            print(f"  All values: {output.flatten()}")
        else:
            print(f"  First 10 values: {output.flatten()[:10]}")
            if len(output.shape) >= 2 and output.shape[0] > 0:
                print(f"  Sample from first row: {output[0].flatten()[:min(10, output[0].size)]}")
    
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    # Analyze the outputs based on expected format
    if len(ort_outputs) == 4:
        boxes, scores, classes, num_detections = ort_outputs
        print("✓ Model has 4 outputs (expected format: boxes, scores, classes, num_detections)")
        
        print(f"\nBoxes: {boxes.shape}")
        print(f"Scores: {scores.shape}")
        print(f"Classes: {classes.shape}")
        print(f"Num detections: {num_detections} (type: {type(num_detections)})")
        
        if isinstance(num_detections, np.ndarray):
            if num_detections.size == 1:
                num_det_value = int(num_detections.item())
                print(f"Number of detections: {num_det_value}")
                
                if num_det_value > 0:
                    print(f"\nFirst few detections:")
                    for i in range(min(5, num_det_value)):
                        if i < len(boxes) and i < len(scores) and i < len(classes):
                            box = boxes[i]
                            score = scores[i]
                            class_id = classes[i]
                            class_name = params.obj_list[class_id] if class_id < len(params.obj_list) else f"Unknown({class_id})"
                            print(f"  {i+1}: {class_name} - {score:.3f} - [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                else:
                    print("No detections found")
            else:
                print(f"⚠️  Unexpected num_detections shape: {num_detections.shape}")
        else:
            print(f"⚠️  num_detections is not a numpy array: {type(num_detections)}")
            
    elif len(ort_outputs) == 3:
        boxes, scores, classes = ort_outputs
        print("✓ Model has 3 outputs (variable format: boxes, scores, classes)")
        
        print(f"\nBoxes: {boxes.shape}")
        print(f"Scores: {scores.shape}")
        print(f"Classes: {classes.shape}")
        
        if len(boxes) > 0:
            print(f"\nFirst few detections:")
            for i in range(min(5, len(boxes))):
                box = boxes[i]
                score = scores[i]
                class_id = classes[i]
                class_name = params.obj_list[class_id] if class_id < len(params.obj_list) else f"Unknown({class_id})"
                print(f"  {i+1}: {class_name} - {score:.3f} - [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        else:
            print("No detections found")
    else:
        print(f"⚠️  Unexpected number of outputs: {len(ort_outputs)}")
        print("Expected either 3 (boxes, scores, classes) or 4 (boxes, scores, classes, num_detections)")
    
    # Check for potential issues
    print(f"\n" + "="*50)
    print("POTENTIAL ISSUES")
    print("="*50)
    
    issues_found = False
    
    # Check if any outputs are empty
    for i, output in enumerate(ort_outputs):
        if output.size == 0:
            print(f"⚠️  Output {i} is empty")
            issues_found = True
    
    # Check if shapes make sense
    if len(ort_outputs) >= 3:
        boxes_shape = ort_outputs[0].shape
        scores_shape = ort_outputs[1].shape
        classes_shape = ort_outputs[2].shape
        
        if len(boxes_shape) >= 2 and boxes_shape[-1] != 4:
            print(f"⚠️  Boxes should have 4 coordinates, got shape: {boxes_shape}")
            issues_found = True
        
        if boxes_shape[:-1] != scores_shape or boxes_shape[:-1] != classes_shape:
            print(f"⚠️  Shape mismatch between outputs:")
            print(f"     Boxes: {boxes_shape}")
            print(f"     Scores: {scores_shape}")
            print(f"     Classes: {classes_shape}")
            issues_found = True
    
    if not issues_found:
        print("✓ No obvious issues found")
    
    print(f"\nClasses in project: {params.obj_list}")
    print(f"Number of classes: {len(params.obj_list)}")

def main():
    parser = argparse.ArgumentParser('Debug ONNX Model Outputs')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('-i', '--image', type=str, required=True, help='Path to input image')
    parser.add_argument('-p', '--project', type=str, default='abhil', help='Project name')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='Model compound coefficient')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.weights):
        print(f"Error: ONNX model not found: {args.weights}")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    project_file = f'projects/{args.project}.yml'
    if not os.path.exists(project_file):
        print(f"Error: Project file not found: {project_file}")
        return
    
    # Load parameters
    params = Params(project_file)
    
    print(f"Project: {args.project}")
    print(f"Model: {args.weights}")
    print(f"Image: {args.image}")
    
    try:
        debug_onnx_model(args.weights, args.image, params, args.compound_coef)
    except Exception as e:
        print(f"\n❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
