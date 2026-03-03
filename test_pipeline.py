import numpy as np
import sys
import os

# Ensure the current directory is in the path
sys.path.append(os.getcwd())

try:
    from ares_pipeline import AresPipeline
    print("Successfully imported AresPipeline.")
except ImportError as e:
    print(f"Error importing AresPipeline: {e}")
    sys.exit(1)

def test_pipeline():
    print("Initializing AresPipeline...")
    # Initialize without tokens for testing
    pipeline = AresPipeline()
    
    print("Creating dummy input data...")
    # Create a dummy image (256x256 RGB)
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    print("Running pipeline...")
    result = pipeline.run(dummy_image)
    
    print("Pipeline run complete.")
    print(f"Flood detected: {result['flood_detected']}")
    print(f"Mask shape: {result['validated_mask'].shape}")
    
    if result['validated_mask'].shape == (256, 256):
        print("TEST PASSED: Output mask shape matches expected dimensions.")
    else:
        print("TEST FAILED: Output mask shape mismatch.")

if __name__ == "__main__":
    test_pipeline()
