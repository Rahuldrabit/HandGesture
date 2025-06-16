# HandGesture

A machine learning project for recognizing and processing hand gestures. This repository contains code for managing gesture data, training and running models, and storing results using various utilities and scripts.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Files](#model-files)
- [Data and Storage](#data-and-storage)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

- `FingerSqSTDJson.py` — Utilities for working with finger sequence data in JSON format.
- `PredictSq.py`, `PredictSqCopy.py` — Scripts for predicting hand gestures using trained models.
- `SequenceAddDBG.py` — Script for debugging or adding new gesture sequences.
- `SequenceGestureLoad.py` — Loads and processes gesture sequences.
- `cromadbCheck.py`, `cromadbTest.py` — Scripts for integrating or testing with ChromaDB for vector storage/retrieval.
- `finger_sequences.json`, `image_count.json` — Data files for gesture sequences and image counts.
- `label_encoder_model.pkl` — Pickle file containing the label encoder for gesture classes.
- `lstm_finger_model.h5`, `lstm_finger_Sq_model.h5`, `lstm_finger_Sq32_model.h5` — Pretrained LSTM model weights for gesture recognition.

### Folders

- `FInalCompleted/` — (Details not shown; likely contains final scripts, models, or notebooks.)
- `PastRun/` — (Details not shown; likely contains logs, previous runs, or saved results.)
- `chroma_db/`, `chroma_storage/` — Storage for ChromaDB database and related files.
- `ipnybFile/` — Jupyter notebook files for experimentation or documentation.

> To see all files and their contents, visit the [repository on GitHub](https://github.com/Rahuldrabit/HandGesture/tree/main/).

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rahuldrabit/HandGesture.git
   cd HandGesture
   ```

2. **Install dependencies:**
   - Make sure you have Python 3.x installed.
   - Install required Python packages (details may be in a requirements.txt or within the scripts themselves).

3. **Download model/data files:**
   - Ensure `.h5`, `.pkl`, and `.json` files are present in the repository root.

## Usage

- **Run gesture prediction:**
  ```bash
  python PredictSq.py
  ```
  or
  ```bash
  python PredictSqCopy.py
  ```

- **Update or debug gesture sequences:**
  ```bash
  python SequenceAddDBG.py
  ```

- **Load and process gesture sequences:**
  ```bash
  python SequenceGestureLoad.py
  ```

- **ChromaDB Integration:**
  - Use `cromadbCheck.py` and `cromadbTest.py` for database operations.

## Model Files

- `lstm_finger_model.h5`, `lstm_finger_Sq_model.h5`, `lstm_finger_Sq32_model.h5`: LSTM-based models for gesture recognition.
- `label_encoder_model.pkl`: Label encoder for gesture classes.

## Data and Storage

- `finger_sequences.json`: Contains gesture sequence data.
- `image_count.json`: Stores image count information.
- `chroma_db/`, `chroma_storage/`: Used for vector database storage.

## Notebooks

- Jupyter notebooks may be found in the `ipnybFile/` directory for further exploration and experimentation.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
