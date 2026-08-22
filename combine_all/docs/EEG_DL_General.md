# General Functionality & Deep Learning (EEG_DL)

This module handles everything related to deep learning, EEG signal processing, network communication, and overall system integration.

## EEG & Deep Learning (`EEG_DL/EEG/`)

- **`model_interface.py`**: The central interface bridging the DL pipelines, data loading, predicting, and the game loop. It manages training (`train_or_calib`), predictions, and keeps track of active models.
- **`MI_train.py`**: Contains the deep learning architectures. It features `EEGNet_our_ver` (a variant of EEGNet) and `BraindecodeTrainer` for training models using the Braindecode framework.
- **`EEG_Read.py`**: Handles asynchronous reading of EEG streams via LSL (Lab Streaming Layer) and saves the raw data to CSV files.
- **`EEG_Predict.py`**: Runs a background thread to continuously fetch EEG data, preprocess it, and make predictions using the active deep learning model, sending results to the game via TCP.
- **`EEG_Train.py`**: Implements the training pipeline to train or fine-tune models on collected data.
- **`data_process_np.py`**: Manages the preprocessing of raw EEG data, creating sliding windows, and packaging data into PyTorch datasets for training. It integrates with the RL module to selectively include/weight data.

## Utilities & Core Logic (`EEG_DL/Utils/` & Root)

- **`config.py`**: Contains global configuration variables (e.g., paths, hyperparameters, IP addresses).
- **`global_value.py`**: Manages global state variables shared across different threads (e.g., EEG buffers, prediction buffers).
- **`LSL.py`**: Utility functions for establishing and managing Lab Streaming Layer connections.
- **`TCP_use.py`**: Provides `TCPServer` and `TCPClient` for inter-process or inter-thread communication, primarily used to send predictions from the predictor thread to the game thread.
- **`main.py`**: The main entry point. It initializes the UI, handles user accounts, coordinates the Pygame loop, starts the EEG recording/predicting threads, and manages the RL closed-loop.

## Workflow

1. **Initialization**: The system sets up LSL streams, TCP sockets, and loads base DL models and RL models.
2. **Data Streaming**: As the user plays, `EEG_Read` collects raw signals, while `EEG_Predict` simultaneously fetches segments, predicts motor imagery intents, and feeds them to the game.
3. **Calibration / Training**: After a song or during specific intervals, `model_interface` triggers data processing. It can train a new model from scratch or fine-tune an existing one based on the latest performance.
4. **RL Closed-Loop**: The RL bandit observes the data quality, chooses actions, and updates itself based on the change in the DL model's loss, creating a continuous improvement cycle.
