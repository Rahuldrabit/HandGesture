import cv2
import time

def capture_images():
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            
            cv2.imshow("Live Webcam", frame)  # Display the live webcam feed
            
            filename = f"captured_image_{count}.jpg"
            cv2.imwrite(filename, frame)  # Save the frame as an image
            print(f"Captured {filename}")
            
            count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.9)  # Wait for 0.3 seconds before capturing next image
    
    except KeyboardInterrupt:
        print("Capture stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()
