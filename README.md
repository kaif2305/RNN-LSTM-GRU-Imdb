# RNN, LSTM, and GRU Comparison for Sentiment Analysis in TensorFlow

This project provides a comparative basic example of using a simple Recurrent Neural Network (RNN), a Long Short-Term Memory (LSTM) network, and a Gated Recurrent Unit (GRU) network for sentiment analysis on the IMDB movie reviews dataset using TensorFlow/Keras. The goal is to demonstrate the fundamental differences in their architecture and performance in handling sequential data, particularly focusing on their ability to capture long-term dependencies.

## Project Overview

The Python script performs the following steps for all three recurrent neural network types (RNN, LSTM, and GRU):

1.  **Dataset Loading and Preprocessing**: Loads the IMDB movie reviews dataset and pads sequences to a uniform length.
2.  **Model Building**: Defines three separate `Sequential` models: one using `SimpleRNN`, one using `LSTM`, and one using `GRU`.
3.  **Model Compilation**: Configures each model with an Adam optimizer, binary cross-entropy loss, and accuracy as a metric.
4.  **Model Training**: Trains all three models on the preprocessed training data with validation.
5.  **Model Evaluation**: Assesses the performance of all trained models on the unseen test dataset.
6.  **Comparison**: Outputs the test loss and accuracy for all three models for direct comparison.

## Dataset

The **IMDB movie reviews dataset** is a standard benchmark for binary sentiment classification. It contains 50,000 highly polarized movie reviews (25,000 for training, 25,000 for testing), labeled as either positive (1) or negative (0). The dataset is pre-processed, with reviews already converted into sequences of integers, where each integer represents a specific word.

### Data Preprocessing

* **Vocabulary Size (`vocab_size`)**: Set to 10,000, considering only the most frequent words.
* **Maximum Sequence Length (`max_len`)**: Set to 200. All movie review sequences are padded with zeros (`padding='post'`) or truncated to this fixed length.

## Understanding RNN, LSTM, and GRU

All three are types of recurrent neural networks designed to process sequential data. They differ in their internal architecture and, consequently, their ability to capture long-term dependencies and mitigate issues like the vanishing gradient problem.

### 1. Simple RNN (`tf.keras.layers.SimpleRNN`)

* **Core Idea**: A basic recurrent unit where the hidden state at the current time step is a function of the current input and the hidden state from the previous time step.
* **Strengths**: Conceptual simplicity, computationally less intensive per step.
* **Weaknesses**:
    * **Vanishing/Exploding Gradient Problem**: Gradients can shrink or grow exponentially over many time steps, making it difficult for the network to learn dependencies that span long sequences.
    * **Short-term Memory**: Due to vanishing gradients, simple RNNs struggle to retain information over long sequences, leading to a "short-term memory" issue.

### 2. Long Short-Term Memory (LSTM) (`tf.keras.layers.LSTM`)

* **Core Idea**: LSTMs are an extension of RNNs that introduce a more complex recurrent unit with a "cell state" (`C_t`) alongside the hidden state (`h_t`). The cell state acts as a memory unit that can retain information over very long periods. Information flow into and out of the cell state is controlled by three special "gates":
    * **Forget Gate ($f_t$)**: Decides what information from the previous cell state should be discarded.
    * **Input Gate ($i_t$)**: Decides what new information should be stored in the cell state.
    * **Output Gate ($o_t$)**: Decides what part of the cell state should be output as the hidden state.
* **Strengths**:
    * **Long-term Dependency Learning**: Effectively addresses the vanishing gradient problem, allowing them to learn and remember information over many time steps.
    * Widely used and highly effective in various sequential tasks.
* **Weaknesses**: More computationally intensive and have more parameters than simple RNNs or GRUs due to their complex gating mechanisms.

### 3. Gated Recurrent Unit (GRU) (`tf.keras.layers.GRU`)

* **Core Idea**: GRUs are a slightly simplified version of LSTMs, also designed to overcome the vanishing gradient problem and capture long-term dependencies. They combine the forget and input gates into a single "update gate" and merge the cell state and hidden state. They typically have two gates:
    * **Update Gate ($z_t$)**: Controls how much of the previous hidden state should be carried over to the current hidden state, and how much new information should be added.
    * **Reset Gate ($r_t$)**: Controls how much of the previous hidden state to forget.
* **Strengths**:
    * Similar performance to LSTMs on many tasks, especially with smaller datasets.
    * Fewer parameters than LSTMs, leading to faster training and less memory usage.
    * Addresses vanishing gradients effectively.
* **Weaknesses**: May sometimes perform slightly worse than LSTMs on very long or complex sequences where the separate cell state of LSTMs proves beneficial.

## Model Architectures (Common Structure)

All three models in this project share a similar overall structure:

* **`Embedding(input_dim=vocab_size, output_dim=128)`**: Converts word indices into dense 128-dimensional vectors. This layer learns numerical representations for each word.
* **Recurrent Layer (SimpleRNN, LSTM, or GRU)**: The core sequential processing layer.
    * `128`: The number of units (neurons/hidden dimensions) in the recurrent layer.
    * `activation='tanh'`: The activation function used within the recurrent unit (for internal transformations, not the gates themselves in LSTM/GRU).
    * `return_sequences=False`: Ensures that only the output from the last time step of the sequence is passed to the next layer. This is suitable for sequence classification tasks where a single prediction is made per sequence.
* **`Dense(1, activation='sigmoid')`**: The output layer for binary classification, yielding a probability between 0 and 1.

## Training and Evaluation

All models are trained for `5 epochs` with a `batch_size` of 32. A `validation_split` of 0.2 is used to monitor performance during training. They are compiled with the `'adam'` optimizer, `'binary_crossentropy'` loss, and `'accuracy'` as a metric.

The key comparison comes from their performance on the test set: