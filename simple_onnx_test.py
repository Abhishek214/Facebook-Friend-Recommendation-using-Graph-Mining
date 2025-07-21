"""
Simple test script for ONNX model to verify it works before running full evaluation

Usage:
python simple_onnx_test.py -w efficientdet-d2-with-postprocessing-fixed.onnx -i test_image.jpg -p abhil
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
    """Simple preprocessing"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Simple resize without aspect ratio preservation for quick test
    resized = cv2.resize(image, (target_size, target_size))
    
    # Normalize
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - np.array(mean)) / np.array(std)
    
    # HWC to CHW and add batch dimension
    input_tensor = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor

def test_onnx_model(onnx_model_path, image_path, params, compound_coef):
    """Simple test of ONNX model"""
    
    print(f"Testing ONNX model: {onnx_model_path}")
    print(f"Input image: {image_path}")
    
    # Load model
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Get model info
    input_info = ort_session.get_inputs()[0]
    print(f"✓ Input: {input_info.name} - {input_info.shape}")
    
    for i, output_info in enumerate(ort_session.get_outputs()):
        print(f"✓ Output {i}: {output_info.name} - {output_info.shape}")
    
    # Preprocess image
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    target_size = input_sizes[compound_coef]
    
    try:
        input_tensor = preprocess_image(image_path, target_size, params.mean, params.std)
        print(f"✓ Image preprocessed: {input_tensor.shape}")
    except Exception as e:
        print(f"❌ Failed to preprocess image: {e}")
        return False
    
    # Run inference
    try:
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: input_tensor}
        ort_outputs = ort_session.run(None, ort_inputs)
        print(f"✓ Inference successful: {len(ort_outputs)} outputs")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False
    
    # Analyze outputs
    try:
        if len(ort_outputs) == 4:
            boxes, scores, classes, num_detections = ort_outputs
            
            print(f"✓ Boxes shape: {boxes.shape}")
            print(f"✓ Scores shape: {scores.shape}")
            print(f"✓ Classes shape: {classes.shape}")
            print(f"✓ Num detections shape: {num_detections.shape}")
            
            # Get number of detections
            if num_detections.shape == ():
                num_det = int(num_detections.item())
            elif num_detections.shape == (1,):
                num_det = int(num_detections[0])
            else:
                print(f"⚠️  Unusual num_detections shape: {num_detections.shape}")
                num_det = 0
            
            print(f"✓ Number of detections: {num_det}")
            
            if num_det > 0:
                print(f"✓ First detection:")
                print(f"   Box: {boxes[0]}")
                print(f"   Score: {scores[0]:.4f}")
                print(f"   Class: {classes[0]} ({params.obj_list[classes[0]] if classes[0] < len(params.obj_list) else 'Unknown'})")
                
                # Check score range
                valid_scores = scores[:num_det]
                print(f"✓ Score range: {np.min(valid_scores):.4f} - {np.max(valid_scores):.4f}")
                
                # Check class range
                valid_classes = classes[:num_det]
                print(f"✓ Class range: {np.min(valid_classes)} - {np.max(valid_classes)}")
                
                # Count detections per class
                unique_classes, counts = np.unique(valid_classes, return_counts=True)
                print(f"✓ Detections per class:")
                for cls, count in zip(unique_classes, counts):
                    cls_name = params.obj_list[cls] if cls < len(params.obj_list) else f"Unknown({cls})"
                    print(f"   {cls_name}: {count}")
                
            else:
                print("⚠️  No detections found")
                print("   This might be normal if the image doesn't contain target objects")
                print("   or the confidence threshold is too high")
            
            return True
            
        else:
            print(f"⚠️  Expected 4 outputs, got {len(ort_outputs)}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to analyze outputs: {e}")
        return False

def main():
    parser = argparse.ArgumentParser('Simple ONNX Model Test')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('-i', '--image', type=str, required=True, help='Path to test image')
    parser.add_argument('-p', '--project', type=str, default='abhil', help='Project name')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='Model compound coefficient')
    
    args = parser.parse_args()
    
    # Check files
    if not os.path.exists(args.weights):
        print(f"❌ ONNX model not found: {args.weights}")
        return
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    project_file = f'projects/{args.project}.yml'
    if not os.path.exists(project_file):
        print(f"❌ Project file not found: {project_file}")
        return
    
    # Load parameters
    try:
        params = Params(project_file)
        print(f"✓ Project loaded: {args.project}")
        print(f"✓ Classes: {params.obj_list}")
    except Exception as e:
        print(f"❌ Failed to load project: {e}")
        return
    
    # Test model
    success = test_onnx_model(args.weights, args.image, params, args.compound_coef)
    
    if success:
        print(f"\n🎉 Model test PASSED! You can now run the full evaluation.")
        print(f"Next step: python eval_onnx_coco_final.py -w {args.weights} -p {args.project} -c {args.compound_coef}")
    else:
        print(f"\n❌ Model test FAILED! Check the issues above before running evaluation.")

if __name__ == '__main__':
    main()
