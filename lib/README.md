## 

## Converters

The data format in `jsonl` is:
```
{Const.OCR: {Const.LINE: the output of an OCR text of a line if exists otherwise None,
            Const.SENTENCE: the output of an OCR text of a sentence (that contains the line),
            Const.REGION: the output of an OCR of the full text},
Const.GROUND: {Const.LINE: the groundtruth text line,
               Const.SENTENCE: the grountruth of a sentence,
               Const.REGION: the grountruth full text}}

```

All converters have the same parameters:

```
python icdar_converter.py --input_dir ../../data/datasets/ocr/original/icdar-2017/ \
                          --output_dir ../../data/datasets/ocr/converted \
                          --extraction_type line \
                          --language en
```
