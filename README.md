# GeoLocST: Efficient Visual Geolocation via Supervised Fine-Tuning

[![Paper](https://img.shields.io/badge/Paper-arXiv:25XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2506.01277)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-PIG_R1-blue)](https://huggingface.co/datasets/paidaixing/PIG_R1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official implementation of the paper **"GeoLocSFT: Efficient Visual Geolocation via Supervised Fine-Tuning of Multimodal Foundation Models"** (NeurIPS 2025). This repository provides the complete, end-to-end pipeline for data generation, model fine-tuning, and performance evaluation.

Our framework demonstrates that high-quality, structured "geo-caption" data can efficiently adapt large multimodal models for planet-scale visual geolocation, achieving competitive performance with a remarkably small dataset (~2,700 samples).

![GeoLocSFT Approach Comparison](https-placeholder-for-figure1) 
*<p align="center">Figure 1: Comparison between traditional methods (Top) and our reasoning-based GeoLocSFT approach (Bottom).</p>*

---

## 🚀 Workflow Overview

This project is structured as a clear, four-step pipeline:

1.  **Data Acquisition**: Download the required image and metadata assets from our [**PIG_R1 Dataset Hub**](https://huggingface.co/datasets/paidaixing/PIG_R1).
2.  **Generate Geo-Captions**: Use the Gemini API to perform expert-level analysis on the images, generating detailed, structured JSON annotations (`geo-captions`).
3.  **Format for Training**: Convert the generated JSON files into a single dataset file compatible with the [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) framework.
4.  **Fine-Tune & Evaluate**: Fine-tune a foundation model (e.g., Gemma, Qwen2.5-VL) and subsequently evaluate its geolocation accuracy.

## 🛠️ Setup and Installation

### 1. Clone the Code Repository
```bash
git clone https://github.com/[Your-GitHub-Username]/GeoLocSFT.git
cd GeoLocSFT
```

### 2. Create Local Data Directories
This project follows the best practice of separating code from data. The following directories are required locally but will be ignored by Git.
```bash
# Create directories for raw data source and processed data
mkdir dataset_source
mkdir data
```

### 3. Download the Dataset from Hugging Face
Our complete dataset, **PIG_R1**, is hosted on Hugging Face Hub. 

*   **Go to the dataset repository: [https://huggingface.co/datasets/paidaixing/PIG_R1](https://huggingface.co/datasets/paidaixing/PIG_R1)**

*   Download the necessary components. For example, to replicate the SFT experiments, you'll need `HFRL.zip` and its corresponding metadata.
*   Place the downloaded and unzipped files into the `dataset_source/` directory you created. Your local structure should look something like this:
    ```
    GeoLocSFT/
    ├── dataset_source/
    │   ├── images/
    │   │   ├── 001.jpg
    │   │   └── ...
    │   └── metadata.csv
    └── ... (other code folders)
    ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Usage Pipeline

Follow these steps to replicate our results from scratch.

### Step 1: Generate Geo-Captions (`scripts/1_generate_geocaptions.py`)

This script uses the Google Gemini API to analyze each image based on its GPS coordinates and generate a detailed JSON file containing a rich "geo-caption".

**➡️ How to Run:**

First, open `scripts/1_generate_geocaptions.py` and configure the constants at the top of the file to point to your local data directories (`IMAGE_DIR`, `CSV_PATH`) and provide your `API_KEY`.

Then, execute the script:
```bash
python scripts/1_generate_geocaptions.py
```
*   **Input:** Images and metadata from your local `dataset_source/` directory.
*   **Output:** A collection of individual `.json` files in the specified `OUTPUT_DIR`.

### Step 2: Format Data for LLaMA Factory (`scripts/2_format_for_llama_factory.py`)

This script gathers all the individual JSON files and converts them into a single dataset file in the `sharegpt` format required by LLaMA Factory.

**➡️ How to Run:**```bash
python scripts/2_format_for_llama_factory.py \
    --input_dir /path/to/your/generated_jsons \
    --output_dir ./data \
    --dataset_name hfrl_data
```
*   **Input:** The directory containing the JSON files from Step 1.
*   **Output:** Creates `data/hfrl_data.json` and `data/dataset_info.json`.

### Step 3: Train the Model (`training/`)

We provide example training scripts in the `training/` directory for both Gemma and Qwen models, configured for multi-GPU training with DeepSpeed.

**➡️ How to Run:**

First, edit the script (e.g., `training/train_gemma.sh`) to ensure all paths (`LLAMA_FACTORY_ROOT`, `MODEL_PATH`, `MEDIA_DIR_ABS` for images) are correct for your system. Then, launch the training:
```bash
bash training/train_ggemma.sh
```

### Step 4: Evaluate the Results (`scripts/evaluate_accuracy.py`)

After training is complete and you have generated a prediction CSV file, this script calculates the final accuracy. The prediction file must contain the columns: `pred_lat`, `pred_lon`, `true_lat`, `true_lon`.

**➡️ How to Run:**

First, open `scripts/evaluate_accuracy.py` and update the path to your prediction CSV file. Then, run the script:
```bash
python scripts/evaluate_accuracy.py
```
*   **Output:** A printed table showing the Acc@R percentages at different distance thresholds (1km, 25km, ..., 2500km).

## Citation

If you find our work and this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{YourLastName2025GeoLocSFT,
  title={GeoLocSFT: Efficient Visual Geolocation via Supervised Fine-Tuning of Multimodal Foundation Models},
  author={Your Name and Co-author Name},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgements
This work is made possible by the foundational contributions of many projects. We extend our gratitude to:
- The creators of the **LLaMA Factory** framework for their excellent open-source tools.
- **Google** and **Alibaba** for the powerful Gemma and Qwen foundation models.
- The teams behind public datasets like **OSV-5M** and **Mapillary Vistas**, which formed a basis for our data curation.
