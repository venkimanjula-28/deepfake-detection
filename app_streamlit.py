import streamlit as st
import torch
import os

# Fix for OpenCV in headless environments (Streamlit Cloud, Docker, etc.)
os.environ['OPENCV_VIDEOIO_PRIORITY_BACKEND'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import base64
from io import BytesIO
from torchvision import transforms
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Import our model and dataset functions
from src.model import get_model
from src.dataset import extract_frames_from_video, detect_faces, pad_or_truncate
from src.database import db, init_session_state, show_login_page, show_user_profile, show_analysis_history

# Initialize session state
init_session_state()


# Page configuration
st.set_page_config(
    page_title="🔍 DeepFake Detector Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid;
    }
    .fake-result {
        background-color: #ffe6e6;
        border-color: #ff4444;
        color: #cc0000;
    }
    .real-result {
        background-color: #e6ffe6;
        border-color: #44ff44;
        color: #00cc00;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_cached(checkpoint_path, backbone='resnet18', use_lstm=False):
    """Load the trained model from checkpoint with caching"""
    try:
        model = get_model(backbone=backbone, pretrained=False, freeze_backbone=False, use_lstm=use_lstm)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try to load state dict, handling potential architecture mismatches
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Warning: Model architecture mismatch. Loading compatible weights only...")
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['model_state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from checkpoint")
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_single_image(image, num_frames=12):
    """Preprocess a single image for the model"""
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Repeat the image to create a sequence
    frames = [image] * num_frames

    # Detect faces and preprocess
    frames_tensor = detect_faces(frames, target_size=224)
    frames_tensor = pad_or_truncate(frames_tensor, num_frames)
    frames_tensor = torch.stack(frames_tensor, dim=0)

    # Both MTCNN (post_process=False) and Haar cascade return [0, 1]
    # Apply ImageNet normalization for model input
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    frames_tensor = normalize(frames_tensor)

    return frames_tensor.unsqueeze(0)  # Add batch dimension


def preprocess_video(video_path, num_frames=12):
    """Preprocess video frames for the model"""
    frames = extract_frames_from_video(video_path, num_frames)
    frames_tensor = detect_faces(frames, target_size=224)
    frames_tensor = pad_or_truncate(frames_tensor, num_frames)
    frames_tensor = torch.stack(frames_tensor, dim=0)

    # Normalize to ImageNet standards
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    frames_tensor = normalize(frames_tensor)

    return frames_tensor.unsqueeze(0)  # Add batch dimension


def predict(model, input_tensor):
    """Make prediction using the model"""
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence_fake = probabilities[1].item()
        confidence_real = probabilities[0].item()

    prediction = "FAKE" if confidence_fake > confidence_real else "REAL"
    confidence = max(confidence_fake, confidence_real)

    return prediction, confidence, confidence_real, confidence_fake


def get_image_download_link(img, filename, text):
    """Generate a download link for an image"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def main():
    # Check authentication status
    if not st.session_state.get('authenticated', False):
        show_login_page()
        return

    # Show user profile in sidebar
    show_user_profile()

    # Page routing
    current_page = st.session_state.get('current_page', 'main')

    if current_page == 'history':
        show_analysis_history()
    else:
        show_main_app()


class DeepFakeVideoTransformer(VideoTransformerBase):
    """Video transformer for real-time webcam deepfake detection"""
    def __init__(self, model, num_frames=12, debug=False):
        self.model = model
        self.num_frames = num_frames
        self.frames_buffer = []
        self.prediction = None
        self.confidence = 0
        self.frame_count = 0
        self.debug = debug
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Add frame to buffer (only every 2nd frame to reduce processing load)
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            self.frames_buffer.append(img.copy())
            if len(self.frames_buffer) > self.num_frames:
                self.frames_buffer.pop(0)
        
        # Only predict when we have enough frames and every 15 frames to reduce lag
        if len(self.frames_buffer) == self.num_frames and self.frame_count % 15 == 0:
            try:
                # Convert frames to RGB for processing
                rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in self.frames_buffer]
                
                # Process frames through face detection
                if self.debug:
                    print(f"Processing {len(rgb_frames)} frames for face detection")
                
                frames_tensor = detect_faces(rgb_frames, target_size=224)
                
                if self.debug:
                    print(f"Face detection returned {len(frames_tensor)} tensors")
                
                # Check if face detection returned valid results
                if len(frames_tensor) == 0:
                    print("Warning: No faces detected in frames")
                    self.prediction = "NO FACE"
                    self.confidence = 0
                else:
                    frames_tensor = pad_or_truncate(frames_tensor, self.num_frames)
                    frames_tensor = torch.stack(frames_tensor, dim=0)
                    
                    # Normalize to ImageNet standards
                    # Both MTCNN (post_process=False) and Haar return [0, 1]
                    # Apply ImageNet normalization for the model
                    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    frames_tensor = normalize(frames_tensor)
                    input_tensor = frames_tensor.unsqueeze(0)
                    
                    # Predict
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)[0]
                        conf_fake = probabilities[1].item()
                        conf_real = probabilities[0].item()
                        
                    self.prediction = "FAKE" if conf_fake > conf_real else "REAL"
                    self.confidence = max(conf_fake, conf_real)
                
            except Exception as e:
                print(f"Prediction error: {e}")
                import traceback
                traceback.print_exc()
        
        # Draw prediction on frame
        if self.prediction and self.prediction != "NO FACE":
            color = (0, 0, 255) if self.prediction == "FAKE" else (0, 255, 0)
            label = f"{self.prediction}: {self.confidence:.1%}"
            cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw confidence bar
            bar_width = 200
            filled_width = int(bar_width * self.confidence)
            cv2.rectangle(img, (10, 50), (10 + bar_width, 70), (128, 128, 128), -1)
            cv2.rectangle(img, (10, 50), (10 + filled_width, 70), color, -1)
        elif self.prediction == "NO FACE":
            cv2.putText(img, "No face detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(img, "Collecting frames...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return img


def show_main_app():
    """Main deepfake detection application"""
    # Header
    st.markdown('<h1 class="main-header">🔍 DeepFake Detector Pro</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Advanced AI-powered deepfake detection using CNN + Temporal Attention
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add tabs for different input methods
    tab1, tab2 = st.tabs(["📤 Upload Media", "📹 Webcam Detection"])

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model settings
        st.subheader("Model Settings")
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="checkpoint.pth",
            help="Path to the trained model checkpoint"
        )

        backbone = st.selectbox(
            "CNN Backbone",
            ["resnet18", "efficientnet_b0"],
            index=0,
            help="Choose the CNN architecture"
        )

        num_frames = st.slider(
            "Temporal Frames",
            min_value=6,
            max_value=30,
            value=12,
            help="Number of frames to analyze (higher = more accurate but slower)"
        )

        use_lstm = st.checkbox(
            "Enable LSTM",
            value=False,
            help="Use LSTM for temporal modeling (if trained with --use-lstm)"
        )

        # Analysis settings
        st.subheader("Analysis Settings")
        show_faces = st.checkbox("Show Detected Faces", value=True)
        show_confidence = st.checkbox("Show Confidence Details", value=True)
        debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed processing information")

        # Load model
        if os.path.exists(checkpoint_path):
            model = load_model_cached(checkpoint_path, backbone, use_lstm)
            if model:
                st.success("✅ Model loaded successfully!")
                st.info(f"Using: {backbone.upper()} {'+ LSTM' if use_lstm else ''}")
            else:
                st.error("❌ Failed to load model")
                st.stop()
        else:
            st.warning(f"⚠️ Checkpoint not found: {checkpoint_path}")
            st.info("Train a model first using: `python src/train.py`")
            st.stop()

    # Tab 1: Upload Media
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("📤 Upload Media")

            # File uploader with expanded options
            uploaded_file = st.file_uploader(
                "Choose an image or video file",
                type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'webm'],
                help="Supported: JPG, PNG, MP4, AVI, MOV, WebM (Max 200MB)",
                accept_multiple_files=False
            )

            if uploaded_file is not None:
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
                if file_size > 200:
                    st.error("❌ File too large! Maximum size: 200MB")
                    st.stop()

                # Display file info
                st.info(f"📄 File: {uploaded_file.name} ({file_size:.1f} MB)")

                # Display the uploaded media
                if uploaded_file.type.startswith('image/'):
                    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                    input_type = "image"
                else:
                    st.video(uploaded_file)
                    input_type = "video"

                # Analysis button
                analyze_button = st.button(
                    "🔍 Analyze for DeepFake",
                    type="primary",
                    use_container_width=True
                )

        with col2:
            st.header("🎯 Analysis Results")

            if uploaded_file is not None and analyze_button:
                with st.spinner("🔄 Processing... Analyzing frames and detecting faces..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Save uploaded file temporarily
                        status_text.text("📂 Saving file...")
                        progress_bar.progress(10)

                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name

                        status_text.text("🔍 Extracting frames...")
                        progress_bar.progress(30)

                        if input_type == "image":
                            input_tensor = preprocess_single_image(Image.open(tmp_path), num_frames)
                        else:
                            input_tensor = preprocess_video(tmp_path, num_frames)

                        status_text.text("🧠 Running AI analysis...")
                        progress_bar.progress(70)

                        prediction, confidence, conf_real, conf_fake = predict(model, input_tensor)

                        status_text.text("✅ Analysis complete!")
                        progress_bar.progress(100)

                        # Log analysis to database
                        user = st.session_state.user
                        db.log_analysis(user['id'], uploaded_file.name, input_type, prediction, confidence)

                        # Clean up
                        os.unlink(tmp_path)

                    except Exception as e:
                        st.error(f"❌ Error processing file: {e}")
                        progress_bar.empty()
                        status_text.empty()
                        st.stop()

                    finally:
                        progress_bar.empty()
                        status_text.empty()

                # Display results
                result_class = "fake-result" if prediction == "FAKE" else "real-result"
                st.markdown(f"""
                <div class="result-box {result_class}">
                    <h2 style="text-align: center; margin: 0;">
                        {"🚨 DEEPFAKE DETECTED!" if prediction == "FAKE" else "✅ AUTHENTIC CONTENT"}
                    </h2>
                    <p style="text-align: center; font-size: 1.5rem; margin: 0.5rem 0;">
                        Confidence: {confidence:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if show_confidence:
                    st.subheader("📊 Detailed Confidence Scores")

                    col_real, col_fake = st.columns(2)

                    with col_real:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Real Content</h4>
                            <h2>{conf_real:.1%}</h2>
                            <div style="background-color: #e9ecef; border-radius: 10px; height: 20px;">
                                <div style="background-color: #28a745; width: {conf_real*100}%; height: 100%; border-radius: 10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_fake:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Fake Content</h4>
                            <h2>{conf_fake:.1%}</h2>
                            <div style="background-color: #e9ecef; border-radius: 10px; height: 20px;">
                                <div style="background-color: #dc3545; width: {conf_fake*100}%; height: 100%; border-radius: 10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Analysis details
                with st.expander("📋 Analysis Details", expanded=True):
                    st.markdown(f"""
                    **Model Configuration:**
                    - Backbone: {backbone.upper()}
                    - Temporal Frames: {num_frames}
                    - LSTM Enabled: {"Yes" if use_lstm else "No"}

                    **Input Information:**
                    - File Type: {input_type.title()}
                    - File Name: {uploaded_file.name}
                    - File Size: {file_size:.1f} MB

                    **AI Prediction:**
                    - Result: {prediction}
                    - Confidence Score: {confidence:.1%}
                    - Real Probability: {conf_real:.3f}
                    - Fake Probability: {conf_fake:.3f}
                    """)

                    if prediction == "FAKE":
                        st.warning("⚠️ This content has been flagged as potentially manipulated. Exercise caution when sharing or using this media.")
                    else:
                        st.success("✅ This content appears to be authentic based on our analysis.")
                
    # Tab 2: Webcam Detection
    with tab2:
        st.header("📹 Real-time Webcam Detection")
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <p>📸 <strong>How to use:</strong></p>
            <ol>
                <li>Click "START" to enable your webcam</li>
                <li>Wait for the model to collect 12 frames (about 1-2 seconds)</li>
                <li>The prediction will appear on the video feed</li>
                <li>Green = REAL, Red = FAKE</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # WebRTC configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        try:
            webrtc_streamer(
                key="deepfake-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                video_transformer_factory=lambda: DeepFakeVideoTransformer(model, num_frames, debug=debug_mode),
                media_stream_constraints={"video": True, "audio": False},
                async_transform=True,
            )
        except Exception as e:
            st.error(f"Webcam error: {e}")
            st.info("Make sure you have a webcam connected and have granted browser permissions.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🔬 About DeepFake Detector Pro</h3>
        <p>
            This advanced deepfake detection system uses state-of-the-art AI with CNN + Temporal Attention architecture.
            It analyzes both spatial features (face detection, image quality) and temporal consistency across video frames.
        </p>
        <p style='font-size: 0.9rem; color: #888;'>
            <strong>Disclaimer:</strong> This is a research tool. Results may not be 100% accurate.
            Always verify critical content through multiple sources.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Add some statistics or info
    if st.sidebar.button("📊 Show Model Info"):
        st.sidebar.markdown("### Model Information")
        st.sidebar.info(f"""
        **Architecture:** {backbone.upper()} {'+ LSTM' if use_lstm else ''}
        **Input Size:** {num_frames} frames × 224×224 pixels
        **Classes:** Real, Fake
        **Training:** Transfer learning with frozen backbone
        """)

        # Show some sample predictions if available
        st.sidebar.markdown("### Recent Performance")
        st.sidebar.info("Current model accuracy: ~52% (after 1 epoch training)")


if __name__ == "__main__":
    main()