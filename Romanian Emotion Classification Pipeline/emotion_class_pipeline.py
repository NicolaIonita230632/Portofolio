import os
import argparse
import torch
import whisper
import pandas as pd
import tempfile
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from transformers import MarianMTModel, MarianTokenizer
from datetime import timedelta

class VideoTranscriber:
    """
    Handles video transcription using OpenAI's Whisper model.
    """
    def __init__(self, model_size="large"):
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded successfully.")

    def transcribe(self, video_path):
        """
        Transcribe a video file and return segments with timestamps.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            list: List of dictionaries containing transcription segments.
        """
        print(f"Transcribing video: {video_path}")
        result = self.model.transcribe(video_path, verbose=False)
        print(f"Transcription complete: {len(result['segments'])} segments found.")
        return result['segments']

class TextTranslator:
    """
    Handles translation of non-English text to English.
    """
    def __init__(self, model_name="BlackKakapo/opus-mt-ro-en"):
        print(f"Loading translation model: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        print("Translation model loaded successfully.")

    def translate(self, text):
        """
        Translate text to English.
        
        Args:
            text (str): Text to translate.
            
        Returns:
            str: Translated text.
        """
        if not text.strip():
            return ""
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            translated = self.model.generate(**inputs)
        translation = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return translation

class EmotionClassifier:
    """
    Classifies emotion in text using a pre-trained XLMRoberta model.
    """
    def __init__(self, model_path="roberta_model_romanian_unbalanced", num_labels=7):
        print(f"Loading emotion classification model from: {model_path}")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            print("No GPU available, using CPU.")
            
        # Load model configuration
        self.config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
        
        # Load model and tokenizer
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_path, config=self.config)
        self.model.to(self.device)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        
        # Setup pipeline
        self.classifier = pipeline("text-classification", 
                                   model=self.model, 
                                   tokenizer=self.tokenizer,
                                   device=0 if torch.cuda.is_available() else -1)
        
        # Define emotion labels
        self.emotion_labels = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        
        print("Emotion classification model loaded successfully.")

    def classify(self, text):
        """
        Classify the emotion in a text.
        
        Args:
            text (str): Text to classify.
            
        Returns:
            str: Detected emotion.
        """
        if not text.strip():
            return "neutral"
            
        result = self.classifier(text)
        
        # Map the numerical label to emotion name
        # The model returns a label as an index ('LABEL_0', 'LABEL_1', ..., 'LABEL_6')
        label_idx = int(result[0]['label'].split('_')[1])
        emotion = self.emotion_labels[label_idx] if label_idx < len(self.emotion_labels) else "unknown"
        
        return emotion

class EmotionPipeline:
    """
    End-to-end pipeline for processing videos, transcribing speech, 
    translating text, and classifying emotions.
    """
    def __init__(self, transcriber, translator, classifier):
        self.transcriber = transcriber
        self.translator = translator
        self.classifier = classifier

    def process_video(self, video_path, output_path):
        """
        Process a video through the entire pipeline.
        
        Args:
            video_path (str): Path to the video file.
            output_path (str): Path to save the results.
        """
        print(f"Starting processing pipeline for: {video_path}")
        
        # Transcribe video
        segments = self.transcriber.transcribe(video_path)
        
        # Process each segment
        results = []
        for segment in segments:
            start_time = self._format_timestamp(segment['start'])
            end_time = self._format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            # Translate text
            translation = self.translator.translate(text)
            
            # Classify emotion on original text instead of translation
            emotion = self.classifier.classify(text)
            
            # Store results
            results.append({
                'Start Time': start_time,
                'End Time': end_time,
                'Sentence': text,
                'Translation': translation,
                'Emotion': emotion
            })
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        return df

    def _format_timestamp(self, seconds):
        """
        Format a timestamp in seconds to the format: HH:MM:SS,mmm
        
        Args:
            seconds (float): Time in seconds.
            
        Returns:
            str: Formatted timestamp.
        """
        td = timedelta(seconds=seconds)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{td}".split('.')[0] + f",{milliseconds:03d}"



def main():
    parser = argparse.ArgumentParser(description="Emotion Classification Pipeline")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--output", default="output.csv", help="Path to save the results")
    parser.add_argument("--whisper_model", default="large", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--translation_model", default="BlackKakapo/opus-mt-ro-en", help="Translation model name")
    parser.add_argument("--emotion_model", default="Task_5/Transformers/Transformers_model_2_RoBerta/roberta_model_romanian_unbalanced_2", help="Path to the emotion classification model")
    parser.add_argument("--num_labels", type=int, default=7, help="Number of emotion labels in the model")
    
    args = parser.parse_args()
    
    # Validate that the video file exists
    if not os.path.exists(args.video):
        parser.error(f"Video file not found: {args.video}")
    
    # Initialize components
    transcriber = VideoTranscriber(model_size=args.whisper_model)
    translator = TextTranslator(model_name=args.translation_model)
    classifier = EmotionClassifier(model_path=args.emotion_model, num_labels=args.num_labels)
    
    # Create and run pipeline
    pipeline = EmotionPipeline(transcriber, translator, classifier)
    pipeline.process_video(args.video, args.output)
    
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()