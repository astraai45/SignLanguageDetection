import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize the camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Set up constants for image processing
offset = 20
imgSize = 300
counter = 0

# Get the label name from user input and create folder if it doesn't exist
label = input("Enter the label name for the new class: ")
folder_path = f"Data/{label}"
os.makedirs(folder_path, exist_ok=True)
print(f"Images will be saved in {folder_path}")

# Start capturing images with specific controls
capturing = False
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a blank white image to place the resized hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the image around the hand with checks to ensure it’s within bounds
        if 0 <= y - offset and y + h + offset <= img.shape[0] and \
           0 <= x - offset and x + w + offset <= img.shape[1]:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Only proceed if imgCrop has content
            if imgCrop.size > 0:
                # Check the aspect ratio and resize accordingly
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                # Display the cropped and white background images
                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)
        else:
            print("Warning: Detected hand's bounding box is out of image bounds.")

    # Display the main image feed
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    # Start capturing images on pressing 's'
    if key == ord("s"):
        capturing = True
        print("Capturing started... Press 'q' to stop.")

    # Stop capturing images on pressing 'q'
    elif key == ord("q"):
        capturing = False
        print("Capturing stopped.")

    # Exit the application on pressing 'x'
    elif key == ord("x"):
        print("Exiting...")
        break

    # Capture and save image if capturing mode is enabled and hand is detected
    if capturing and hands and imgCrop.size > 0:
        counter += 1
        cv2.imwrite(f'{folder_path}/Image_{time.time()}.jpg', imgWhite)
        print(f"Captured image {counter}")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
