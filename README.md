# Manga Translator

## Why ?

A long, long, long time ago, a wee lad was reading a manga. It was such a blissfull read for this wee lad before he realized and before he realized it he had reached the last page.Curious on the next release this lad hoped on the web to find answers only to that the manga was no longer being translated. Stricken by grief this wee lad set out to right this wrong.

## How ?

- [Yolo](https://github.com/ultralytics/ultralytics) for bubble recognition and text segmentation
- Open CV for bubble clean up and masking
- PIL for text replacement
- Have not decided on ocr tech or translation method

## Progress

- [x] Bubble recognition (should improve with more training data)
- [x] Free text recognition (should improve with more training data)
- [x] Bubble text extraction
- [x] Bubble masking
- [x] Bubble text insertion
- [ ] Bubble Text ocr
- [x] Bubble Text translation
- [ ] Free text ocr
- [ ] Free text translation
- [ ] Improve text resize algorithm, some texts are too small/big

## Examples

### Original

<img src="ja_oshi_no_ko_1_4.png" width="500"/><img src="ja_oshi_no_ko_1_28.png" width="500"/>

### Converted

<img src="ja_oshi_no_ko_1_4_converted.png" width="500"/><img src="ja_oshi_no_ko_1_28_converted.png" width="500"/>

## Glossary

- Bubble: a speech bubble
- Free text: text found on pages but not in speech bubbles
- Bubble Text: text within speech bubbles
