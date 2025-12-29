# Manga Translator

<img src="assets\examples\ui_2025-12-27.png"/>

## What
Extensible manga translator

## Install
- This repo uses [uv](https://github.com/astral-sh/uv) for package management, see individual parts below for specifics
  
## Parts
- [Actual Translator](./translator/README.md)
- A [UI](./ui/README.md) for translating individual images
- A basic [CLI](./cli/README.md) example
  
## Limitations
- Yolo by ultralytics is AGPL, need to find alternative detector and segmenter (or maybe remove from repo so repo is not AGPL)
- Currently no form of text color detection so text will always be black
- Only horizontal left to right text layout is supported for output
- Very large vertical images are not split so either they work or they dont
- Dataset is only japanese text so detection might not work on other languages
- Aside from the yolo models, all other models are general-purpose and may or may not work well on manga

## Models Included
- Yolo detection and segmentation
- deepfillv2 inpainting
- lama inpainting

<!--
Links are currently down, and I'm not sure when I will be able to bring them back up
## Datasets

### Detection

<a href="https://universe.roboflow.com/tarehimself/manga-translator-detection">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>

### Segmentation

<a href="https://universe.roboflow.com/tarehimself/manga-translator-segmentation">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>
-->
##  Examples

<table>
   <thead>
      <tr>
         <th align="center" width="50%">Original</th>
         <th align="center" width="50%">Translated</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td align="center" width="50%">
            <img alt="Original" src="assets/examples/jujutsu_kaisen.png" width="100%"/>
         </td>
         <td align="center" width="50%">
            <img alt="Result" src="assets/examples/jujutsu_kaisen_converted.png" width="100%"/>
         </td>
      </tr>
      <tr>
         <td colspan=2 align="center">Japanese => English</br>Jujutsu Kaisen</td>
      </tr>
      <tr>
         <td align="center" width="50%">
            <img alt="Original" src="assets/examples/solo_leveling.png" width="100%"/>
         </td>
         <td align="center" width="50%">
            <img alt="Result" src="assets/examples/solo_leveling_converted.png" width="100%"/>
         </td>
      </tr>
      <tr>
         <td colspan=2 align="center">Japanese => "Meow"</br>Solo Leveling</td>
      </tr>
      <tr>
         <td align="center" width="50%">
            <img alt="Original" src="assets/examples/the_rising_of_the_sheild_hero.jpg" width="100%"/>
         </td>
         <td align="center" width="50%">
            <img alt="Result" src="assets/examples/the_rising_of_the_sheild_hero_converted.jpg" width="100%"/>
         </td>
      </tr>
      <tr>
         <td colspan=2 align="center">Japanese => Clean</br>The Rising of the Shield Hero</td>
      </tr>
      <tr>
         <td align="center" width="50%">
            <img alt="Original" src="assets/examples/ja_a_certain_scientific_accelerator.png" width="100%"/>
         </td>
         <td align="center" width="50%">
            <img alt="Result" src="assets/examples/ja_a_certain_scientific_accelerator_converted.png" width="100%"/>
         </td>
      </tr>
      <tr>
         <td colspan=2 align="center">Japanese => English</br>A Certain Scientific Accelerator</td>
      </tr>
      <tr>
         <td align="center" width="50%">
            <img alt="Original" src="assets/examples/ja_one_punch_man.jpg" width="100%"/>
         </td>
         <td align="center" width="50%">
            <img alt="Result" src="assets/examples/ja_one_punch_man_converted.jpg" width="100%" />
         </td>
      </tr>
      <tr>
         <td colspan=2 align="center">Japanese => English</br>One Punch Man</td>
      </tr>
      <tr>
         <td align="center" width="50%">
            <img alt="Original" src="assets/examples/ja_oshi_no_ko.png" width="100%"/>
         </td>
         <td align="center" width="50%">
            <img alt="Result" src="assets/examples/ja_oshi_no_ko_converted.png" width="100%"/>
         </td>
      </tr>
      <tr>
         <td colspan=2 align="center">Japanese => English</br>Oshi No Ko</td>
      </tr>
   </tbody>
</table>
<!-- 
## Glossary

- Bubble: a speech bubble
- Free text: text found on pages but not in speech bubbles
- Bubble Text: text within speech bubbles -->
