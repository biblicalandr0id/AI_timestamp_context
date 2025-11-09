"""
Vision Interface Module
Provides image understanding capabilities using CLIP and other vision models
Allows chatbot to "see" and understand images
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from PIL import Image
import base64
import io

try:
    from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("transformers not available. Install with: pip install transformers")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("opencv not available. Install with: pip install opencv-python")


class ImageEncoder:
    """Encode images to embeddings using CLIP"""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize image encoder

        Args:
            model_name: CLIP model to use
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install with: pip install transformers")

        print(f"Loading CLIP model: {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        print(f"✅ CLIP model loaded on {self.device}")

    def encode_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Encode image to embedding vector

        Args:
            image: Image file path, PIL Image, or numpy array

        Returns:
            Image embedding as numpy array
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        # Get embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector (for image-text matching)

        Args:
            text: Text to encode

        Returns:
            Text embedding as numpy array
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()[0]

    def compute_similarity(self, image: Union[str, Path, Image.Image], text: str) -> float:
        """
        Compute similarity between image and text

        Args:
            image: Image to compare
            text: Text to compare

        Returns:
            Similarity score (0-1)
        """
        image_emb = self.encode_image(image)
        text_emb = self.encode_text(text)

        # Cosine similarity
        similarity = np.dot(image_emb, text_emb)

        return float(similarity)

    def classify_image(self, image: Union[str, Path, Image.Image], candidates: List[str]) -> Dict[str, float]:
        """
        Classify image into one of the candidate categories

        Args:
            image: Image to classify
            candidates: List of possible categories

        Returns:
            Dictionary mapping categories to probabilities
        """
        # Encode image
        image_emb = self.encode_image(image)

        # Encode all candidates
        results = {}
        for candidate in candidates:
            text_emb = self.encode_text(candidate)
            similarity = np.dot(image_emb, text_emb)
            results[candidate] = float(similarity)

        # Softmax to get probabilities
        exp_scores = np.exp(list(results.values()))
        probs = exp_scores / exp_scores.sum()

        return {cat: prob for cat, prob in zip(results.keys(), probs)}


class ImageCaptioner:
    """Generate captions for images using BLIP"""

    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """
        Initialize image captioner

        Args:
            model_name: BLIP model to use
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install with: pip install transformers")

        print(f"Loading BLIP model: {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.processor = BlipProcessor.from_pretrained(model_name)

        print(f"✅ BLIP model loaded on {self.device}")

    def caption_image(self, image: Union[str, Path, Image.Image], max_length=50) -> str:
        """
        Generate caption for image

        Args:
            image: Image to caption
            max_length: Maximum caption length

        Returns:
            Image caption
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        # Generate caption
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length)

        caption = self.processor.decode(output[0], skip_special_tokens=True)

        return caption

    def conditional_caption(self, image: Union[str, Path, Image.Image], prompt: str, max_length=50) -> str:
        """
        Generate conditional caption based on prompt

        Args:
            image: Image to caption
            prompt: Conditioning prompt
            max_length: Maximum caption length

        Returns:
            Conditional caption
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        # Process with prompt
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)

        # Generate caption
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length)

        caption = self.processor.decode(output[0], skip_special_tokens=True)

        return caption


class ObjectDetector:
    """Detect objects in images using simple methods"""

    def __init__(self):
        """Initialize object detector"""
        if not CV2_AVAILABLE:
            raise ImportError("opencv required. Install with: pip install opencv-python")

    def detect_faces(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image

        Args:
            image: Image to analyze

        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        return [tuple(face) for face in faces]

    def detect_edges(self, image: Union[str, Path, np.ndarray], low_threshold=50, high_threshold=150) -> np.ndarray:
        """
        Detect edges using Canny edge detection

        Args:
            image: Image to analyze
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold

        Returns:
            Edge image
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        return edges


class VisionInterface:
    """Complete vision interface combining all vision capabilities"""

    def __init__(
        self,
        enable_clip=True,
        enable_blip=True,
        enable_detection=True,
        clip_model="openai/clip-vit-base-patch32",
        blip_model="Salesforce/blip-image-captioning-base"
    ):
        """
        Initialize vision interface

        Args:
            enable_clip: Enable CLIP encoder
            enable_blip: Enable BLIP captioner
            enable_detection: Enable object detection
            clip_model: CLIP model name
            blip_model: BLIP model name
        """
        self.encoder = None
        self.captioner = None
        self.detector = None

        if enable_clip:
            try:
                self.encoder = ImageEncoder(clip_model)
            except Exception as e:
                print(f"⚠️ Could not load CLIP: {e}")

        if enable_blip:
            try:
                self.captioner = ImageCaptioner(blip_model)
            except Exception as e:
                print(f"⚠️ Could not load BLIP: {e}")

        if enable_detection and CV2_AVAILABLE:
            try:
                self.detector = ObjectDetector()
            except Exception as e:
                print(f"⚠️ Could not load detector: {e}")

    def analyze_image(self, image: Union[str, Path, Image.Image]) -> Dict[str, any]:
        """
        Comprehensive image analysis

        Args:
            image: Image to analyze

        Returns:
            Dictionary with analysis results
        """
        results = {
            'caption': None,
            'embedding': None,
            'faces': [],
            'has_people': False,
            'description': None
        }

        # Generate caption
        if self.captioner:
            try:
                results['caption'] = self.captioner.caption_image(image)
                results['description'] = results['caption']
            except Exception as e:
                print(f"⚠️ Caption error: {e}")

        # Encode image
        if self.encoder:
            try:
                results['embedding'] = self.encoder.encode_image(image)
            except Exception as e:
                print(f"⚠️ Encoding error: {e}")

        # Detect faces
        if self.detector and isinstance(image, (str, Path)):
            try:
                results['faces'] = self.detector.detect_faces(image)
                results['has_people'] = len(results['faces']) > 0
            except Exception as e:
                print(f"⚠️ Detection error: {e}")

        # Check for people using CLIP
        if self.encoder and not results['has_people']:
            try:
                person_sim = self.encoder.compute_similarity(image, "a photo of people")
                results['has_people'] = person_sim > 0.25
            except Exception as e:
                print(f"⚠️ CLIP person check error: {e}")

        return results

    def question_answering(self, image: Union[str, Path, Image.Image], question: str) -> str:
        """
        Answer questions about an image

        Args:
            image: Image to analyze
            question: Question to answer

        Returns:
            Answer to the question
        """
        if not self.captioner:
            return "Visual question answering not available (BLIP not loaded)"

        try:
            # Use conditional captioning as VQA
            answer = self.captioner.conditional_caption(image, question)
            return answer
        except Exception as e:
            return f"Error answering question: {e}"

    def search_images_by_text(self, image_paths: List[Union[str, Path]], query: str, top_k=5) -> List[Tuple[str, float]]:
        """
        Search for images matching text query

        Args:
            image_paths: List of image paths to search
            query: Text query
            top_k: Number of top results to return

        Returns:
            List of (image_path, similarity_score) tuples
        """
        if not self.encoder:
            return []

        results = []

        for img_path in image_paths:
            try:
                similarity = self.encoder.compute_similarity(img_path, query)
                results.append((str(img_path), similarity))
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


def demo_vision_interface():
    """Demo the vision interface"""
    print("Vision Interface Demo")
    print("=" * 60)

    # Create interface
    vision = VisionInterface(
        enable_clip=True,
        enable_blip=True,
        enable_detection=True
    )

    # Test with a sample image (you would provide your own)
    print("\n1. Image Analysis Demo:")
    print("   (Requires sample image)")
    print("   Example: vision.analyze_image('path/to/image.jpg')")

    print("\n2. Image Classification Demo:")
    if vision.encoder:
        print("   Example:")
        print("   categories = ['cat', 'dog', 'bird', 'car']")
        print("   results = vision.encoder.classify_image('image.jpg', categories)")
        print("   print(results)")

    print("\n3. Question Answering Demo:")
    if vision.captioner:
        print("   Example:")
        print("   answer = vision.question_answering('image.jpg', 'What is in this image?')")
        print("   print(answer)")

    print("\n4. Image Search Demo:")
    if vision.encoder:
        print("   Example:")
        print("   images = ['img1.jpg', 'img2.jpg', 'img3.jpg']")
        print("   results = vision.search_images_by_text(images, 'a cat')")
        print("   print(results)")

    print("\n✅ Vision interface ready!")
    print("   - CLIP for image-text matching")
    print("   - BLIP for image captioning")
    print("   - OpenCV for face detection")


if __name__ == '__main__':
    demo_vision_interface()
