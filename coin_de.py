import numpy as np
from typing import List
import imutils
from skimage import io
import streamlit as st
from ultralytics import YOLO
import math
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Custom video transformer class
class FrameCaptureTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to BGR format
        bgr_frame = frame.to_ndarray(format="bgr24")

        # Process the frame (e.g., object detection, annotation)
        processed_frame = detect_objects_and_annotate(bgr_frame)

        # Return the processed frame
        return processed_frame

def detect_objects_and_annotate(image, num_cavities, tons_per_inch_sq):
    # Load YOLOv5 model
    model = YOLO("yolov8m-seg-custom.pt")

    # Initialize pixel_per_cm outside the conditional block
    pixel_per_cm = None

    # Detect objects using YOLOv5
    results = model.predict(source=image, show=False)

    # Check if results is not empty
    if results:
        # Initialize pixel_per_cm here to avoid undefined variable error
        pixel_per_cm = None
        
        for result in results:
            # Get bounding box coordinates for each image
            bounding_boxes = result.boxes.xyxy  # Access bounding box coordinates in [x1, y1, x2, y2] format

            # Draw circles around the detected objects
            for box in bounding_boxes:
                x1, y1, x2, y2 = box[:4].int().tolist()  # Convert tensor to list
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                radius = max(abs(x2 - x1), abs(y2 - y1)) // 2
                cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)  # Draw circle
                
                # Calculate dimensions of the reference object (coin)
                ref_w, ref_h = abs(x2 - x1), abs(y2 - y1)
                dist_in_pixel = max(ref_w, ref_h)  # Assuming the longer side of the bounding box as the reference size
                
                # Diameter of the coin in cm
                ref_coin_diameter_cm = 2.426
                
                # Calculate pixel-to-cm conversion factor
                pixel_per_cm = dist_in_pixel / ref_coin_diameter_cm

                # Draw reference object size message above the detected object
                ref_text = " Size=0.955"
                cv2.putText(image, ref_text, (center_x - 150, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
        if pixel_per_cm is None:
            st.error("No Reference object detected in the image. Please recapture.")
            return None  # Skip further processing for this image if no objects are detected

    # Find contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edged = cv2.Canny(blur, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Filter out contours detected by YOLO
        filtered_contours = []
        for cnt in cnts:
            if cv2.contourArea(cnt) > 50:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Check if the contour falls within any bounding box of objects detected by YOLO
                contour_in_yolo_object = False
                for yolo_box in bounding_boxes:
                    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_box[:4].int().tolist()
                    if yolo_x1 < rect[0][0] < yolo_x2 and yolo_y1 < rect[0][1] < yolo_y2:
                        contour_in_yolo_object = True
                        break

                if not contour_in_yolo_object:
                    filtered_contours.append(cnt)

        # Find the contour with the largest area
        largest_contour = max(filtered_contours, key=cv2.contourArea)

        # Draw contour of the object with the largest area
        if largest_contour is not None:
            # Draw contour of the object
            cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)  # Draw contour line instead of bounding box
            
            # Calculate dimensions and area of the object
            area_cm2 = cv2.contourArea(largest_contour) / (pixel_per_cm ** 2)
            rect = cv2.minAreaRect(largest_contour)
            (x, y), (width_px, height_px), angle = rect
            width_cm = width_px / pixel_per_cm
            height_cm = height_px / pixel_per_cm

            # If area is less than 1, check shape of contour line and calculate area accordingly
            if area_cm2 < 1:
                # Calculate aspect ratio of the bounding rectangle
                aspect_ratio = width_px / height_px
                
                # If aspect ratio is less than 1, consider it as a long and narrow shape (e.g., rectangle)
                if aspect_ratio < 1:
                    # Calculate the perimeter of the contour
                    perimeter = cv2.arcLength(largest_contour, True)
    
                    # Estimate the diameter of a circle with the same perimeter
                    diameter = perimeter / math.pi
    
                    # Calculate the area of the circle as an approximation of the irregular shape's area
                    area_cm2 = (diameter / 2) ** 2 * math.pi

            # Calculate text positions
            text_x = int(x - 100)
            text_y = int(y - 20)

            # Calculate dimensions and area of the object in inches
            width_in = width_cm / 2.54
            height_in = height_cm / 2.54
            area_in2 = area_cm2 / 2.54

            # Calculate tonnage
            tonnage = calculate_tonnage(area_in2, num_cavities, tons_per_inch_sq)

            # Draw text annotations with dimensions in inches and tonnage
            cv2.putText(image, "Length: {:.1f}in".format(width_in), (text_x, text_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, "Breadth: {:.1f}in".format(height_in), (text_x, text_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, "Area: {:.1f}in^2".format(area_in2), (text_x, text_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, "Tonnage: {:.2f}".format(tonnage), (text_x, text_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the size above the image
    cv2.putText(image, "Coin is the reference Object", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 1), 2)
    st.success(f" Length: {height_in:.1f}in, Breadth: {width_in:.1f}in, Projected Area: {area_in2:.1f}in^2, Tonnage: {tonnage:.2f}")

    return image

# Function to calculate tonnage based on area, number of cavities, and tons per inch square
def calculate_tonnage(area_in2, num_cavities, tons_per_inch_sq):
    # Implement your tonnage calculation logic here
    tonnage = area_in2 * num_cavities * tons_per_inch_sq
    return tonnage

# Streamlit app
def main():
    st.title("Object Detection and Size Estimation")
    choice = st.sidebar.selectbox("Choose an option:", ("Upload Image", "Capture from Camera"))

    if choice == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = io.imread(uploaded_file)
            image_name = uploaded_file.name  # Get the name of the uploaded file
            num_cavities = st.number_input("Number of Cavities", value=1)
            tons_per_inch_sq = st.number_input("Tons per Inch Square", value=1.0)
            if st.button("Check Dimensions"):
                # Display the captured image
                st.image(image, caption="Uploaded Image", use_column_width=True)
                annotated_image = detect_objects_and_annotate(image, num_cavities, tons_per_inch_sq)
                if annotated_image is not None:
                    st.image(annotated_image, caption="Annotated Image", use_column_width=True)
                else:
                    st.error("Cannot Calculate Dimension and Tonnage Without Coin")
    else:
        st.write("Press the button below to capture an image:")
            st.write("Capturing image...")
            # Use WebRTC to capture video from the camera
            webrtc_streamer(key="example", video_transformer_factory=FrameCaptureTransformer)

if __name__ == "__main__":
    main()
