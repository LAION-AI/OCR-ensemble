# OCR-ensemble

## Some results

https://colab.research.google.com/drive/1hKu8q2SH80baCj-0IRBb9rLDSgBaU1w7#scrollTo=C9v0iNYVJO6Y

## Installation

Follow these steps to set up the environment and install the required dependencies using conda.

### Prerequisites

- Python 3.9
- PyTorch (GPU version)
- PaddleOCR 

### Installing Dependencies

1. Clone the repository:

```bash
git clone git@github.com:LAION-AI/OCR-ensemble.git
cd OCR-ensemble
```

2. Create a conda virtual environment (optional, but recommended):

```bash
conda create -n your-env-name python=3.9
conda activate your-env-name
```

3. Install PyTorch (GPU version) by following the instructions on the [official website](https://pytorch.org/get-started/locally/). Make sure to choose the conda-based installation for your system.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

4. Install paddlepaddle by following the instructions on the [official GitHub repository](https://github.com/PaddlePaddle/PaddleOCR#installation). 
In order to install the GPU version, this might be helpful:

#### Linux
```bash
python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Windows
```bash
python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

5. Install the remaining required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```


### Overview
1. Classify document for type of text 
2. Use expert from ensemble of existing OCR + layout parsing models  to get text+bboxes of text, —> concant that to caption 
3. If there is no original caption like for screenshots of websites and books, just make a caption, concat that with OCR results 
4. Use this data set to train clip with character level tokenization

**Now we are working on Step 2.**

### Pipeline: 2 Passes
1. Classify images to determine text types
2. Expert models process filtered images


### Candidate Expert Models
* **[Printed Document]** Machine-printed text
https://huggingface.co/naver-clova-ocr/bros-large-uncased
https://huggingface.co/microsoft/layoutlmv3-large
[multilingual]
https://github.com/PaddlePaddle/PaddleOCR

* **[Handwritten]** Handwritten text [implemented]
https://huggingface.co/microsoft/trocr-large-handwritten

* **[Handwritten]** Handwritten math [implemented]
https://huggingface.co/Azu/trocr-handwritten-math

* **[Printed Document, Latex formula]** Latex expert [implemented]
https://colab.research.google.com/drive/1TO10E5fa9KeVyHQBhQQP3VESeigRTcsG?usp=sharing

* CLIP language detector, limited functionality [implemented]
https://colab.research.google.com/drive/16XU0v8JEeolQ4uK8XL0EbmZhLWLpF0ti?usp=sharing

* CLIP text detector, simply detects text or no text in images [implemented]
https://colab.research.google.com/drive/1M66t-lnd0QT-opdGS4Bc_zkdzdWAjYCa?usp=sharing 
