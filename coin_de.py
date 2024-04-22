import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Define a custom video transformer to capture frames
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        self.frame = frame
        return frame

# Main Streamlit app function
def main():
    st.title("WebRTC Camera Image Capture")

    # Create a webrtc streamer to capture video from the camera
    webrtc_ctx = webrtc_streamer(
        key="camera",
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )

    # Display the captured image
    if webrtc_ctx.video_transformer:
        st.image(webrtc_ctx.video_transformer.frame, channels="BGR")

if __name__ == "__main__":
    main()
