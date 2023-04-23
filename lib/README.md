## 

## Converters

The data format in `jsonl` is:
```
{
Const.FILE: the file path of the concerned text,
Const.OCR: {Const.LINE: the output of an OCR text of a line if exists otherwise None,
            Const.SENTENCE: the output of an OCR text of a sentence (that contains the line),
            Const.REGION: the output of an OCR of the full text},
Const.GROUND: {Const.LINE: the groundtruth text line,
               Const.SENTENCE: the grountruth of a sentence,
               Const.REGION: the grountruth full text}}

```

All converters have the same parameters (`input_dir` and `output_dir`) and generate a dataset `$DATASET$` in `$OUTPUT_DIR` (e.g., `data/ocr/converted`).

### Converter ICDAR 2017 & 2019
```
python icdar_converter.py --input_dir ../../data/datasets/ocr/original/icdar-2017/ \
                          --output_dir ../../data/datasets/ocr/converted
```

### Converter impresso-nzz
impresso-nzz has the XLM files in the groundtruth folder and the OCRed text by its internal ABBYY FineReader Server 11. The converter maps the files in both folder, along with the `region_id` and `line_id` in every region. If the OCRed region or text line is not found, it is disconsired.

```
python icdar_converter.py --input_dir ../../data/datasets/ocr/original/icdar-2017/ \
                          --output_dir ../../data/datasets/ocr/converted
```
