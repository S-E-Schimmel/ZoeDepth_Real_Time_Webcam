import time
import cv2
import torch

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cuda')
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Initialize variables for FPS calculation
cap = cv2.VideoCapture(0)
start_time = cv2.getTickCount()
frame_count = 0

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame,dsize=(480,240))

    # Calculate FPS
    frame_count += 1
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    fps = frame_count / elapsed_time

    # Convert the frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cuda')

    with torch.inference_mode():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

    output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    output_norm = cv2.flip(output_norm, 1)
    frame = cv2.flip(frame, 1)

    # Draw FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Output', output_norm)
    cv2.imshow('Input', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()