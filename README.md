Thank you for sharing the code! Based on it, here is a **custom README** for your **SMILES Generation Transformer Decoder** project:

---

# SMILES Generation with Transformer Decoder

This project implements a **SMILES generation model** using a **Transformer Decoder** architecture. The model is trained to generate **SMILES (Simplified Molecular Input Line Entry System)** strings, which are text representations of molecular structures, useful in cheminformatics and drug discovery.

## 🚀 Features

- **Transformer Decoder Model**: The core model leverages a Transformer architecture, using the decoder-only setup to generate SMILES sequences.
- **SMILES Generation**: Generates SMILES strings from a trained model, given a starting token.
- **Training & Generation**: Includes training scripts and a generation function for SMILES sequences.
- **Tokenizer**: Built-in dictionary for encoding SMILES strings into numerical tokens.

## 📋 Requirements

- Python 3.x
- PyTorch
- `torchvision`
- Other dependencies in `requirements.txt`

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## 🔧 Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/smiles-generation-transformer.git
cd smiles-generation-transformer
```

Install the necessary packages:

```bash
pip install -r requirements.txt
```

## 🧑‍💻 Usage

1. **Prepare Your Dataset**:  
   The dataset should contain SMILES strings. Make sure the SMILES data file (`smiles_train.txt`) is formatted such that each line contains one SMILES string.

2. **Train the Model**:  
   Use the provided script to train the model. For example, to train the model on your dataset, run:

   ```bash
   python train.py
   ```

   You can adjust hyperparameters like the number of epochs, learning rate, and batch size in the code.

3. **Generate SMILES**:  
   After training, you can generate SMILES sequences using the `generate_smiles()` function. Here’s an example usage within the `train.py` script:
   
   ```python
   generated_smiles = generate_smiles(model, dataset, start_token=dataset.dictionary['SOS'], device=device, k=5)
   print("Generated SMILES:", generated_smiles[3:])
   ```

   This will generate SMILES strings starting with the 'SOS' token and continuing until the 'EOS' token is reached.

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

---

Feel free to adjust any specific details like hyperparameters, file paths, or custom instructions for your setup. Let me know if you'd like more details or modifications!