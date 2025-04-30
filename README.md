# SMILES Generation with Transformer Decoder

This project implements a **SMILES generation model** using a **Transformer Decoder** architecture. The model is trained to generate **SMILES (Simplified Molecular Input Line Entry System)** strings, which are text representations of molecular structures, useful in cheminformatics and drug discovery.

The code combines both **training** and **generation** functionality in a single script (`main.py`), allowing you to train the model and generate SMILES strings in one flow.

## 🚀 Features

- **Transformer Decoder Model**: The core model leverages a Transformer architecture, using the decoder-only setup to generate SMILES sequences.
- **SMILES Generation**: Generates SMILES strings from a trained model, given a starting token.
- **Tokenizer**: Built-in dictionary for encoding SMILES strings into numerical tokens.
- **Training and Generation**: Both training the model and generating SMILES are handled in the `main.py` script.

## 📋 Requirements

To run this project, you will need to install the following Python libraries:

- Python 3.x
- **PyTorch** (for building and training the model)
- **NumPy** (for handling numerical operations)
- **TQDM** (for progress bar during training)
- **Torchvision** (optional, but sometimes needed in conjunction with PyTorch)

To install these dependencies, use the following command:

```bash
pip install torch numpy tqdm torchvision
```

## 🔧 Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/smiles-generation-transformer.git
cd smiles-generation-transformer
```

Then, install the necessary packages:

```bash
pip install torch numpy tqdm torchvision
```

## 🧑‍💻 Usage

### 1. Prepare Your Dataset:
The dataset should contain SMILES strings. Ensure that the SMILES data file (`smiles_train.txt`) is formatted such that each line contains one SMILES string.

### 2. Train and Generate SMILES:
To train the model and generate SMILES strings, simply run:

```bash
python main.py
```

### 3. Model Output:
During training, the model will output generated SMILES sequences after every epoch for evaluation.

## 🧑‍🔬 Model Architecture

The model is based on a **Transformer Decoder** architecture and includes:
- **Token Embeddings**: The SMILES characters are represented by learned embeddings.
- **Positional Encoding**: The model incorporates positional encoding to maintain the sequence order of the SMILES string.
- **Transformer Decoder**: The decoder processes the embedded input sequence using self-attention and generates the next tokens.
- **Output Layer**: A linear layer maps the decoder output to the vocabulary size, producing logits for each token.

### Parameters:
- `vocab_size`: Number of tokens in the SMILES dictionary.
- `dim`: Dimensionality of the model (default: 256).
- `nhead`: Number of attention heads (default: 8).
- `num_layers`: Number of Transformer decoder layers (default: 6).
- `dim_feedforward`: Size of the feedforward layer in the Transformer (default: 1024).
- `max_seq_length`: Maximum sequence length of SMILES strings (default: 101).

## 🔧 Model Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 (adjustable)
- **Loss Function**: Cross-entropy loss with padding token masking
- **Batch Size**: 64 (adjustable)

## 🤝 Contributing

Feel free to fork this repository, create issues, and submit pull requests. Contributions and improvements are welcome!

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Additional Notes:
- **Dataset Format**: Each line in the dataset should contain a single SMILES string.
- **Generated SMILES**: The model generates SMILES sequences, starting from a special "SOS" token and ending at the "EOS" token.
