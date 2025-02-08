import pyautogui
import time

# Path to the image you want to click
image_path = "E:/TechnyxProject/ClickEuclideanDistance/image.png"

# Time interval between clicks (in seconds)
delay = 0.3

try:
    while True:
        location = pyautogui.locateCenterOnScreen(image_path, confidence=0.8)
        if location:
            pyautogui.click(location)
            print(f"Clicked at {location}")
        else:
            print("Image not found")
        time.sleep(delay)
except KeyboardInterrupt:
    print("Script stopped by user.")
