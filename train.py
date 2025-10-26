from ultralytics import YOLO

def main():
    """
    This script trains a YOLOv8 model on a custom dataset.
    
    Instructions:
    1. Make sure you have a 'data.yaml' file in your project directory.
    2. Ensure the paths in 'data.yaml' are correct and absolute.
    3. Run this script from your activated virtual environment:
       python train.py
    """
    
    print("Loading YOLOv8 'nano' model (yolov8n.pt)...")
    # Load a pretrained YOLOv8 nano model. 
    # 'yolov8n.pt' is small and fast, good for starting.
    # You can also use 'yolov8s.pt' (small) or 'yolov8m.pt' (medium) for better accuracy.
    model = YOLO('yolov8n.pt')

    print("Starting training...")
    # Train the model using the 'data.yaml' file.
    # - data: Path to your dataset configuration file.
    # - epochs: Number of times to go through the entire dataset. 50 is a good start.
    # - imgsz: Image size to use for training. 640 is standard for YOLOv8.
    # - device: 'cpu' or 0 (for the first GPU). Training on GPU is MUCH faster.
    #           Set to 'cpu' if you don't have an NVIDIA GPU.
    try:
        results = model.train(
            data='data.yaml', 
            epochs=50, 
            imgsz=640,
            device='cpu'  # <-- CHANGE THIS to 0 if you have a compatible NVIDIA GPU
        )
        
        print("Training complete!")
        print("Model results:", results)
        print("\nYour trained model is saved in the 'runs/detect/train/weights/' directory.")
        print("Look for the file 'best.pt'. This is your custom model!")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Please check the following:")
        print("1. Is 'data.yaml' in the same directory?")
        print("2. Are the paths in 'data.yaml' correct and absolute?")
        print("3. Do you have enough disk space?")
        print("4. If using GPU (device=0), are your NVIDIA drivers and CUDA installed correctly?")

if __name__ == '__main__':
    main()
