import cv2
import torch
from torchvision.transforms import ToTensor
from zoedepth.utils.misc import colorize
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import time

# Use GPU with NVIDIA CUDA cores if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: Running on CPU. This will be slow. Check your CUDA installation.")

# Initialize ZoeDepth
print("*" * 20 + " Initializing zoedepth " + "*" * 20)
conf = get_config("zoedepth_nk", "infer") # Choose specific ZoeDepth model
model = build_model(conf).to(DEVICE) # Build the Monocular Depth Estimation model
model.eval() # Set the model to evaluation mode

# Open the laptop webcam feed to use as input device
cap = cv2.VideoCapture(0)

while cap.isOpened():

    start_time = time.time() # Grab time for FPS calculation

    ret, frame = cap.read() # Grab a frame the webcam feed
    if not ret:
        break

    orig_size = frame.shape[:2][::-1] # Grab original size of frame
    X = ToTensor()(frame).unsqueeze(0).to(DEVICE) # Perform tensor device conversion

    with torch.no_grad():
        out = model.infer(X).cpu() # Make Metric Depth prediction
    pred = colorize(out[0]) # Colorize the output

    # Resize prediction to match the input frame size
    pred = cv2.resize(pred, orig_size)

    # Grab end time & calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time

    # Draw FPS on the output frame
    cv2.putText(pred, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output frame in a window
    cv2.imshow('Depth Prediction', pred)

    # Press 'q' to stop the script and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()