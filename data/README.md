# Data Description

The folder fpresents the eight post-correction benchmarks, each comprising two historical transcriptions: an automated system's output needing correction, and its corresponding ground truth. These versions are aligned at various levels without any images of the original documents. The selection of benchmarks was influenced by diversity and the need for transcripts of adequate length to offer sufficient context for Large Language Models (LLMs).


### Organisation
Below is a sample structure for organizing a repository with multiple folders and subfolders, represented in Markdown format. This structure is designed to illustrate a generic project setup, which can be customized according to specific project requirements.

# Repository Structure

```
data/
├── asr/
│   ├── original/
│   │   └── ina/*txt,*xml
│   └── converted/
│   │   └── ina.jsonl
├── htr/
│   ├── original/
│   │   ├── htrec/*csv
│   └── converted/
│   │   └── htrec.jsonl
├── ocr/
│   ├── original/
│   │   ├── ajmc/*tsv
│   │   ├── icdar-2017/*txt
│   │   ├── icdar-2019/*txt
│   │   ├── impresso-nzz/*xml
│   │   └── overproof/*txt
└── └── converted/
        ├── ajmc.jsonl
        ├── icdar-2017.jsonl
        ├── icdar-2019.jsonl
        └── impresso-nzz.jsonl
```

## Top-Level Folders

- `docs/`: Contains documentation files like setup instructions, usage examples, and FAQs.
- `src/`: Houses the source code, including main application code and tests, organized by programming language.
- `data/`: Stores data files, organized into raw, processed, and final datasets.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training.
- `scripts/`: Shell scripts for tasks like preprocessing data and deploying applications.
- `config/`: Configuration files for different environments (development, production, etc.).










| Dataset       | Document Type            | Origin | Time Period | Language               | # Lines | # Sentences | # Regions |
|---------------|--------------------------|--------|-------------|------------------------|---------|-------------|-----------|
| icdar-2017    | newspapers, monographies | OCR    | 17C-20C     | en, fr                 | 0       | 461         | 28        |
| icdar-2019    |                          | OCR    | not specified | bg, cz, en, fr, de, pl, sl | 0   | 404         | 41        |
| overproof     | newspaper                | OCR    | 19-20C      | en                     | 2,278   | 399         | 41        |
| impresso-nzz  | newspaper                | OCR    | 18-20C      | de                     | 1,256   | 577         | 203       |
| ajmc-mixed    | class. commentaries      | OCR    | 19C         | grc, de, en, fr        | 535     | 379         | 33        |
| ajmc-primary  | class. commentaries      | OCR    | 19C         | grc, de, en, fr        | 40      | 27          | 9         |
| htrec         | papyri and manuscripts   | HTR    | 10C-16C     | grc                    | 180     | 8           | 8         |
| ina           | radio programs           | ASR    | 20C         | fr                     | 201     | 290         | 6         |
