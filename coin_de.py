import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Display the camera stream
        return frame

def main():
    st.title("Live Image Capture")
    st.write("This app displays a live camera stream. Click the button below to capture an image.")

    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    if webrtc_ctx.video_processor:
        if st.button("Capture Image"):
            # Capture the current frame from the camera stream
            captured_frame = webrtc_ctx.video_processor.frame
            if captured_frame is not None:
                st.write("Image captured successfully!")
                st.image(captured_frame.to_ndarray(format="rgb24"), channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()
