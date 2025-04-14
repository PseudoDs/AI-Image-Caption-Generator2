import sys
import threading
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QTextEdit, QStatusBar, QFileDialog, QWidget, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage

from PIL import Image
from transformers import pipeline
import onnxruntime as ort
from transformers import AutoTokenizer, AutoFeatureExtractor


class SignalEmitter(QObject):
    update_status = pyqtSignal(str)
    update_caption = pyqtSignal(str)
    update_image = pyqtSignal(QPixmap)
    enable_copy = pyqtSignal(bool)
    show_error = pyqtSignal(str)
    update_progress = pyqtSignal(int)


class CaptionGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Caption Generator")
        self.setGeometry(100, 100, 800, 800)
        
        # Initialize model variables
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self.ort_session = None
        self.use_onnx = False
        
        # Create UI
        self.init_ui()
        
        # Load model in background
        self.load_model_threaded()

    def init_ui(self):
        """Initialize the user interface"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        self.main_layout.addWidget(self.image_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)
        
        # Button controls
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setSpacing(10)
        
        self.browse_btn = QPushButton("Browse Image")
        self.browse_btn.clicked.connect(self.load_image)
        self.browse_btn.setStyleSheet("padding: 8px;")
        self.controls_layout.addWidget(self.browse_btn)
        
        self.copy_btn = QPushButton("Copy Caption")
        self.copy_btn.clicked.connect(self.copy_caption)
        self.copy_btn.setEnabled(False)
        self.copy_btn.setStyleSheet("padding: 8px;")
        self.controls_layout.addWidget(self.copy_btn)
        
        self.main_layout.addLayout(self.controls_layout)
        
        # Caption display
        self.caption_text = QTextEdit()
        self.caption_text.setReadOnly(True)
        self.caption_text.setStyleSheet("font-size: 12pt; padding: 10px;")
        self.caption_text.setPlaceholderText("Caption will appear here...")
        self.main_layout.addWidget(self.caption_text)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to load an image")
        
        # Signal emitter for cross-thread communication
        self.signal = SignalEmitter()
        self.signal.update_status.connect(self.status_bar.showMessage)
        self.signal.update_caption.connect(self.update_caption_text)
        self.signal.update_image.connect(self.update_image_display)
        self.signal.enable_copy.connect(self.copy_btn.setEnabled)
        self.signal.show_error.connect(self.show_error_message)
        self.signal.update_progress.connect(self.update_progress_bar)

    def load_model_threaded(self):
        """Load the model in a background thread"""
        self.signal.update_status.emit("Loading AI model (first time may take a few minutes)...")
        self.progress_bar.setVisible(True)
        
        thread = threading.Thread(target=self.initialize_model, daemon=True)
        thread.start()

    def initialize_model(self):
        """Initialize either ONNX or standard model"""
        try:
            # First try to load ONNX model
            self.signal.update_progress.emit(10)
            onnx_path = Path("vit-gpt2-image-captioning.onnx")
            
            if onnx_path.exists():
                self.signal.update_status.emit("Loading ONNX model...")
                self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                self.feature_extractor = AutoFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                
                # Configure ONNX runtime
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=options,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.use_onnx = True
                self.signal.update_progress.emit(80)
            else:
                # Fall back to standard model
                self.signal.update_status.emit("Loading standard model...")
                self.model = pipeline(
                    "image-to-text",
                    model="nlpconnect/vit-gpt2-image-captioning",
                    device="cuda" if ort.get_device() == "GPU" else "cpu"
                )
                self.signal.update_progress.emit(80)
            
            self.signal.update_status.emit("Model loaded successfully! Ready to use.")
            self.signal.update_progress.emit(100)
        
        except Exception as e:
            self.signal.show_error.emit(f"Failed to load model:\n{str(e)}")
            self.signal.update_status.emit("Model failed to load")
        finally:
            self.signal.update_progress.emit(0)
            self.progress_bar.setVisible(False)

    def load_image(self):
        """Open file dialog to select an image"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select an image",
            "",
            "Images (*.jpg *.jpeg *.png);;All Files (*)"
        )
        
        if filepath:
            try:
                image = Image.open(filepath)
                self.display_image(image)
                
                self.signal.update_status.emit("Generating caption...")
                self.signal.enable_copy.emit(False)
                
                # Start caption generation in a separate thread
                thread = threading.Thread(
                    target=self.generate_caption,
                    args=(image,),
                    daemon=True
                )
                thread.start()
                
            except Exception as e:
                self.signal.show_error.emit(f"Failed to load image:\n{str(e)}")
                self.signal.update_status.emit("Error loading image")

    def display_image(self, image: Image.Image):
        """Display the selected image in the UI"""
        image = image.convert("RGB")
        image.thumbnail((600, 400))
        
        # Convert PIL Image to QPixmap
        qimage = QImage(
            image.tobytes(),
            image.size[0],
            image.size[1],
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        
        self.signal.update_image.emit(pixmap)

    def generate_caption(self, image: Image.Image):
        """Generate caption for the given image"""
        try:
            if not (self.model or (self.use_onnx and self.ort_session)):
                self.signal.show_error.emit("Model is still loading, please wait")
                return
            
            self.signal.update_progress.emit(10)
            
            if self.use_onnx:
                # ONNX inference
                inputs = self.feature_extractor(images=image, return_tensors="np")
                pixel_values = inputs.pixel_values.astype("float32")
                
                # Generate initial input_ids
                input_ids = self.tokenizer(
                    "",
                    return_tensors="np"
                ).input_ids.astype("int32")
                
                # Beam search parameters
                num_beams = 5
                max_length = 64
                length_penalty = 1.0
                early_stopping = True
                
                # Prepare inputs for ONNX
                ort_inputs = {
                    "pixel_values": pixel_values,
                    "input_ids": input_ids,
                    "max_length": np.array([max_length], dtype=np.int32),
                    "num_beams": np.array([num_beams], dtype=np.int32),
                    "length_penalty": np.array([length_penalty], dtype=np.float32),
                    "early_stopping": np.array([early_stopping], dtype=bool)
                }
                
                self.signal.update_progress.emit(30)
                
                # Run inference
                outputs = self.ort_session.run(None, ort_inputs)
                caption = self.tokenizer.decode(outputs[0][0], skip_special_tokens=True)
                
            else:
                # Standard transformer pipeline
                result = self.model(
                    image,
                    max_new_tokens=64,
                    num_beams=5,
                    length_penalty=1.0,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    do_sample=False
                )
                caption = result[0]['generated_text']
            
            self.signal.update_progress.emit(90)
            self.signal.update_caption.emit(caption)
            self.signal.enable_copy.emit(True)
            self.signal.update_status.emit("Caption generated successfully!")
            self.signal.update_progress.emit(100)
            
        except Exception as e:
            self.signal.show_error.emit(f"Caption generation failed:\n{str(e)}")
            self.signal.update_status.emit("Caption generation failed")
        finally:
            self.signal.update_progress.emit(0)

    def update_caption_text(self, text: str):
        """Update the caption text display"""
        self.caption_text.setPlainText(text)

    def update_image_display(self, pixmap: QPixmap):
        """Update the image display"""
        self.image_label.setPixmap(pixmap)

    def copy_caption(self):
        """Copy caption to clipboard"""
        caption = self.caption_text.toPlainText()
        if caption:
            clipboard = QApplication.clipboard()
            clipboard.setText(caption)
            self.status_bar.showMessage("Caption copied to clipboard!")
            
            # Reset status after 2 seconds
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(2000, lambda: self.status_bar.showMessage("Ready to load another image"))

    def show_error_message(self, message: str):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

    def update_progress_bar(self, value: int):
        """Update progress bar visibility and value"""
        if value == 0:
            self.progress_bar.setVisible(False)
        else:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style for modern look
    app.setStyle("Fusion")
    
    # Create and show main window
    window = CaptionGeneratorApp()
    window.show()
    
    sys.exit(app.exec_())