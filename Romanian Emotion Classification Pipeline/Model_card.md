# Model Card: Romanian Emotion Classifier (RobBERT)

## Project Context: Emotion Classification in Video Content

This model is designed as part of a larger pipeline that processes video content by transcribing audio, optionally translating text, and classifying the expressed emotions per sentence. The client, the **Content Intelligence Agency**, uses artificial intelligence to analyze TV shows at scale, requiring accurate emotion detection in Romanian dialogue. This model aims to adress this.

---

## Model Overview


### Architecture

- **Base Model**: `xlm-roberta-base` (a multilingual transformer pretrained on 100+ languages including Romanian)
- **Fine-tuned Model**: A custom model trained on Romanian TV show transcripts (`roberta_model_romanian_unbalanced`)
- **Classification Head**: A linear layer on top of the XLM-RoBERTa base for predicting one of 7 emotion classes.
- **Target Emotions**:
  1. Neutral  
  2. Happiness  
  3. Sadness  
  4. Surprise  
  5. Fear  
  6. Anger  
  7. Disgust  

**Technical details**:
- Model: `XLMRobertaForSequenceClassification`
- Tokenizer: `XLMRobertaTokenizer`
- Maximum sequence length: 256 tokens
- Optimizer: `AdamW` (learning rate = 2e-5)
- Scheduler: Linear with no warmup steps
- Batch size: 16
- Epochs: 10
- Device: CPU or GPU (depending on availability)


### Purpose

The primary goal is to **automate emotion recognition** in **Romanian-language media** content. This addresses use cases such as:
- Analysis of transcripts in film or TV,
- Emotion-aware content recommendations,
- Content tagging and searchability.

### Development Context

- Model development addressed **class imbalance** by using synonym replacement for underrepresented emotions (fear, disgust, anger, sadness, surprise).
- Pretrained Romanian language model (BERT) was fine-tuned to leverage state-of-the-art transformer-based learning.

---

## Intended Use

### Suitable Applications

- **Media Analysis**: Classify emotions in Romanian subtitles or transcripted dialogue.  
- **Content Recommendation**: Tag content by emotional profile for improved user recommendations.

### Limitations

- **Romanian-only Scope**: The emotion classifier is primarily trained on Romanian text and is not suitable for other languages unless retrained.
- **Imperfect Performance**: Despite reaching around 85% on the weigthed F1 score in trinaing on standard Romanian text, the model is not flawless and may misinterpret subtle expressions, resulting in reduced performance in specific scenarios.
- **Separate Translation Model**: A separate model for translation into English exists but is **not currently used** in this project. It is generally accurate, though it may still produce errors in certain domains or specialized contexts.



### Client Relevance

- **Content Intelligence Agency** can integrate this model into an automated pipeline, supporting large-scale analysis of TV shows, movies, and online videos.  
- The model’s outputs can highlight emotional patterns.

---

## Dataset Details

### Source and Preprocessing

- **Dataset**: The training dataset (`translated_dataset.csv`) consists of transcriptions from Romanian TV shows. These transcripts were manually labeled with the seven target emotions
- **Data Cleaning**:
  - Removal of entries missing labels.
  - Consolidation to the specified emotion classes
- **Tokenization**: The Romanian BERT tokenizer was applied to all text entries to prepare them for model input.

### Language and Representation

- Focuses on **standard Romanian**; may not generalize to dialects or heavily colloquial text.
- Cultural and demographic representation may be limited to the data at hand.
- **Future training** might expand coverage to domain-specific or slang-heavy corpora.

---

## Performance Metrics and Evaluation


### Error Analysis Report

[Error analysis](Task_8\Error_Analysis.md)

The model achieved an overall **accuracy of 0.41** and a **weighted F1 score of 0.43**. Performance varies significantly across emotion categories due to **severe class imbalance**. 'Neutral' and 'Happiness' are the best-represented and best-performing classes, with F1 scores of 0.56 and 0.51 respectively. In contrast, minority classes such as 'Disgust', 'Fear', and 'Surprise' are often poorly predicted or missed altogether, with F1 scores near zero in some cases.

Key findings include:
- **Frequent misclassification** between semantically close classes (e.g., anger ↔ neutral, sadness ↔ surprise).
- **'Disgust' was never correctly identified**, likely due to only 7 instances being available.
- The **confusion matrix** highlights strong bidirectional confusion between 'neutral' and 'happiness', and misclassification of 'sadness' as 'surprise' in many cases.

Model performance also suffers on short or emotionally ambiguous sentences, rhetorical expressions, and sarcasm. To address these issues, recommended actions include:
- Oversampling of minority emotion classes.
- Implementation of a **weighted loss function**.
- Curating a larger, more diverse training set.

(Refer to `Error_Analysis.md` for detailed label-by-label breakdown and illustrative examples.)


---

## Explainability and Transparency

[XAI_report](Task_9\XAI_notebook.ipynb)

The model was evaluated using explainability tools like SHAP and/or LIME to understand which tokens most influence predictions. Salient features include:
- Strong correlation between emotionally charged words and class predictions.
- Overreliance on **keywords**, rather than contextual nuance, especially for 'fear' and 'disgust'.
- Visualization examples showed that the model struggles with **irony**, **metaphors**, and **cultural idioms**.

Future development may benefit from additional **attention visualization** and token-level explanation alignment with human interpretation.


---

## Recommendations for Use

### Deployment Notes

- **Input Length**: Keep input text below 256 tokens to prevent truncation and potential loss of key context.  
- **Translation Considerations**: If dealing with non-Romanian content, a separate translation model may be required. Ensure translation quality meets the needs of emotion analysis.

### Operational Risks

- **Linguistic Variability**: The model may struggle with sarcasm, slang, code-switching, or highly figurative language.  
- **Bias Amplification**: Any biases in the training data can manifest in the model’s predictions. 
- **Model Drift**: Over time, changes in language usage or shifts in application domains can degrade performance. Periodic retraining and evaluation are recommended.

### Guidance for Media Companies

- **Pipeline Integration**: Implement the classifier as part of a broader analytics pipeline that can include transcription, translation (if needed), and emotion tagging on a per-sentence basis.  
- **Editorial Insights**: Use the model’s output to spot emotion trends across a large catalog of films, series, or live broadcasts. This can guide content creation, scheduling, or targeted viewer recommendations.  
- **Human-in-the-Loop**: Treat the classifier’s output as a support tool rather than a final decision-maker. Employ review to validate critical decisions and refine model performance over time.