# OCR-ensemble

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