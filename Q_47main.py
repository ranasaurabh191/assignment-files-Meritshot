# Machine Translation: English to Spanish using Seq2Seq with Attention
# This script implements a Seq2Seq model with LSTM and attention mechanism
# for translating English sentences to Spanish. It includes preprocessing,
# model training, evaluation with BLEU score, and a Flask API for deployment.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import uuid
from flask import Flask, request, jsonify
import logging

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ------------------- Data Preprocessing -------------------
# Description: Tokenize sentences, create vocabularies, and pad sequences.
# Special tokens: <START>, <END>, <PAD>, <UNK> are added to handle sequence generation.

def preprocess_data(english_sentences, spanish_sentences, max_len=50):
    # Tokenize English sentences
    eng_tokenizer = Tokenizer(filters='', oov_token='<UNK>')
    eng_tokenizer.fit_on_texts(english_sentences)
    eng_sequences = eng_tokenizer.texts_to_sequences(english_sentences)
    
    # Tokenize Spanish sentences
    spa_tokenizer = Tokenizer(filters='', oov_token='<UNK>')
    spa_tokenizer.fit_on_texts(['<START> ' + s + ' <END>' for s in spanish_sentences])
    spa_sequences = spa_tokenizer.texts_to_sequences(['<START> ' + s + ' <END>' for s in spanish_sentences])
    
    # Pad sequences (post-padding for both encoder and decoder)
    eng_padded = pad_sequences(eng_sequences, maxlen=max_len, padding='post', truncating='post')
    spa_padded = pad_sequences(spa_sequences, maxlen=max_len, padding='post', truncating='post')
    
    # Vocabulary sizes (including special tokens)
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    spa_vocab_size = len(spa_tokenizer.word_index) + 1
    
    return eng_padded, spa_padded, eng_tokenizer, spa_tokenizer, eng_vocab_size, spa_vocab_size

# ------------------- Seq2Seq Model with Attention -------------------
# Description: Build an encoder-decoder model with LSTM and attention.
# Encoder: Processes English input into a context vector.
# Decoder: Generates Spanish output using attention to focus on input tokens.
# Teacher forcing is used during training.

def build_seq2seq_model(eng_vocab_size, spa_vocab_size, embedding_dim=256, lstm_units=512):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(eng_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(spa_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Attention Layer
    attention = Attention()
    attention_output = attention([decoder_outputs, encoder_lstm])
    
    # Dense layer for output
    decoder_dense = Dense(spa_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(attention_output)
    
    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model, encoder_inputs, encoder_lstm, encoder_states, decoder_inputs, decoder_lstm, decoder_embedding, decoder_dense, attention

# ------------------- Inference Setup -------------------
# Description: Setup encoder and decoder models for inference with beam search.
# Beam search considers multiple possible outputs for better translation quality.

def setup_inference_models(encoder_inputs, encoder_lstm, encoder_states, decoder_inputs, decoder_lstm, decoder_embedding, decoder_dense, attention, lstm_units=512):
    # Encoder model
    encoder_model = Model(encoder_inputs, [encoder_lstm] + encoder_states)
    
    # Decoder model
    decoder_state_input_h = Input(shape=(lstm_units,))
    decoder_state_input_c = Input(shape=(lstm_units,))
    encoder_outputs_input = Input(shape=(None, lstm_units))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    attention_output = attention([decoder_outputs, encoder_outputs_input])
    decoder_outputs = decoder_dense(attention_output)
    
    decoder_model = Model([decoder_inputs, encoder_outputs_input] + decoder_states_inputs,
                         [decoder_outputs, state_h, state_c])
    
    return encoder_model, decoder_model

# Beam Search Decoding
def beam_search_decode(encoder_model, decoder_model, input_seq, spa_tokenizer, beam_width=3, max_len=50):
    # Encode the input
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq)
    states = [state_h, state_c]
    
    # Initialize beam
    start_token = spa_tokenizer.word_index['<START>']
    end_token = spa_tokenizer.word_index['<END>']
    beams = [([start_token], states, 0.0)]
    completed = []
    
    for _ in range(max_len):
        all_candidates = []
        for seq, states, score in beams:
            target_seq = np.array([[seq[-1]]])
            output, h, c = decoder_model.predict([target_seq, encoder_outputs] + states, verbose=0)
            probs = output[0, -1, :]
            top_indices = np.argsort(probs)[-beam_width:]
            
            for idx in top_indices:
                new_seq = seq + [idx]
                new_score = score - np.log(probs[idx] + 1e-10)
                all_candidates.append((new_seq, [h, c], new_score))
        
        # Select top beam_width candidates
        beams = sorted(all_candidates, key=lambda x: x[2])[:beam_width]
        
        # Check for completed sequences
        for seq, states, score in beams:
            if seq[-1] == end_token:
                completed.append((seq, score))
        beams = [b for b in beams if b[0][-1] != end_token]
        
        if not beams and completed:
            break
    
    # Return the best sequence
    if completed:
        best_seq = min(completed, key=lambda x: x[1])[0]
        return [spa_tokenizer.index_word.get(idx, '<UNK>') for idx in best_seq[1:-1]]
    return []

# ------------------- Evaluation with BLEU -------------------
# Description: Evaluate translations using BLEU score and perform manual inspection.

def evaluate_bleu(references, hypotheses):
    smoothie = SmoothingFunction().method4
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = [ref.split()]
        hyp_tokens = hyp.split()
        score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
        scores.append(score)
    return np.mean(scores)

# ------------------- Main Training and Evaluation -------------------
# Description: Load data, preprocess, train the model, and evaluate performance.
# For simplicity, we use a small synthetic dataset. Replace with a real corpus (e.g., Tatoeba).

# Sample data (replace with real dataset)
english_sentences = ["I love to eat", "The sun is shining", "She is reading a book"]
spanish_sentences = ["Me encanta comer", "El sol está brillando", "Ella está leyendo un libro"]

# Preprocess data
max_len = 50
eng_padded, spa_padded, eng_tokenizer, spa_tokenizer, eng_vocab_size, spa_vocab_size = preprocess_data(english_sentences, spanish_sentences, max_len)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(eng_padded, spa_padded, test_size=0.2, random_state=42)

# Build and train the model
model, encoder_inputs, encoder_lstm, encoder_states, decoder_inputs, decoder_lstm, decoder_embedding, decoder_dense, attention = build_seq2seq_model(eng_vocab_size, spa_vocab_size)
model.fit([X_train, y_train[:, :-1]], y_train[:, 1:, np.newaxis], 
          batch_size=64, epochs=10, validation_data=([X_val, y_val[:, :-1]], y_val[:, 1:, np.newaxis]))

# Setup inference models
encoder_model, decoder_model = setup_inference_models(encoder_inputs, encoder_lstm, encoder_states, 
                                                     decoder_inputs, decoder_lstm, decoder_embedding, 
                                                     decoder_dense, attention)

# Evaluate translations
translated = []
for seq in X_val:
    translated.append(' '.join(beam_search_decode(encoder_model, decoder_model, seq[np.newaxis, :], spa_tokenizer)))
bleu_score = evaluate_bleu(spanish_sentences[:len(translated)], translated)
print(f"BLEU Score: {bleu_score}")

# Manual evaluation (print sample translations)
for i in range(min(3, len(X_val))):
    print(f"English: {english_sentences[i]}")
    print(f"Predicted Spanish: {translated[i]}")
    print(f"Actual Spanish: {spanish_sentences[i]}\n")

# ------------------- Error Analysis -------------------
# Description: Common errors include misaligned sentence structure and incorrect translations.
# Potential improvements: Use larger corpus, better tokenization, or pre-trained embeddings.

# Example error analysis
errors = {
    "Misaligned structure": "Model may swap subject-verb order.",
    "Context misinterpretation": "Ambiguous words translated incorrectly.",
    "Complex phrases": "Idioms or long phrases mistranslated."
}
print("Error Analysis:")
for error, desc in errors.items():
    print(f"{error}: {desc}")

# ------------------- Flask API for Deployment -------------------
# Description: Deploy the model as a Flask web service for real-time translation.

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess input
        input_text = data['text']
        input_seq = eng_tokenizer.texts_to_sequences([input_text])
        input_padded = pad_sequences(input_seq, maxlen=max_len, padding='post')
        
        # Translate
        translation = beam_search_decode(encoder_model, decoder_model, input_padded, spa_tokenizer)
        return jsonify({'translation': ' '.join(translation)})
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# ------------------- Notes on Extensions -------------------
# 1. Pre-trained Embeddings: Use GloVe/FastText by loading pre-trained vectors and initializing the Embedding layer.
# 2. Transformers: Replace Seq2Seq with pre-trained models like T5 or mBART for better performance.
# 3. Hyperparameter Tuning: Experiment with embedding_dim (128-512), lstm_units (256-1024), batch_size (32-128), learning_rate (0.001-0.0001).
# 4. Multilingual: Extend to other languages using multilingual datasets and models like mBART.
