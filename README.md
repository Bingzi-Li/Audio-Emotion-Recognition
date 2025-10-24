# Speech Emotion Recognition with Deep Learning

This project builds deep learning models for classifying emotions from audio data.

## Methodology
- Convert `.wav` audio files to **Mel-spectrogram** features
- Train multiple architectures:
  - 2D CNN baseline
  - CNN + LSTM
  - CNN + Attention-based LSTM
  - CNN + Autoencoder representation

## Results
| Model | Best Accuracy |
|--------|----------------|
| CNN | 92.3% |
| CNN + LSTM | 95.8% |
| CNN + Attention LSTM | 96.3% |

## Key Takeaways
- Attention mechanisms and unsupervised pretraining significantly improved performance.
- Demonstrated potential of hybrid CNN-LSTM structures for emotion recognition.

## Tech Stack
- Python, Keras, Librosa, NumPy, Matplotlib
