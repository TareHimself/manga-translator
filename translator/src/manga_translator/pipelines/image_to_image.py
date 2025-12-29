from manga_translator.core.pipeline import Pipeline
from manga_translator.core.plugin import DetectionResult, Translator, Detector, Segmenter, SegmentationResult, Cleaner, Drawer, OCR, ColorDetector
import numpy as np
import cv2
import asyncio
import torch
from manga_translator.core.typing import Vector4i
from manga_translator.utils import has_white, draw_area_bbox


class FrameSection:
    def __init__(self,source_index: int,source: np.ndarray,section: np.ndarray,cleaned_section: np.ndarray,mask: np.ndarray,text: np.ndarray,bbox: Vector4i,draw_section: np.ndarray,draw_bbox: Vector4i):
        self.source_index = source_index
        self.source = source
        self.section = section
        self.cleaned_section = cleaned_section
        self.mask = mask
        self.text = text
        self.bbox = bbox
        self.draw_section = draw_section
        self.draw_bbox = draw_bbox
        
class TranslatableFrame:
    def __init__(self,index: int,frame: np.ndarray,detections: list[DetectionResult],segments: list[SegmentationResult]):
        self.index = index
        self.frame = frame
        self.detections = detections
        self.segments = segments
         
class ImageToImagePipeline(Pipeline):
    def __init__(self,detector: Detector = Detector(), segmenter: Segmenter = Segmenter(),translator: Translator = Translator(),cleaner: Cleaner = Cleaner,drawer: Drawer = Drawer(),ocr: OCR = OCR(),color_detector: ColorDetector = ColorDetector()):
        super().__init__()
        self.translator = translator
        self.detector = detector
        self.segmenter = segmenter
        self.cleaner = cleaner
        self.drawer = drawer
        self.ocr = ocr
        self.color_detector = color_detector

    def make_mask(self,frame: np.ndarray,segments: list[SegmentationResult]):
        h, w = frame.shape[:2]
        result = np.zeros((h,w),dtype=np.uint8)

        if len(segments) == 0:
            return result
        segments_list = list(map(lambda a: a.points,segments))
        cv2.fillPoly(result, segments_list, (255, 255, 255))

        return result
            
        
    # async def extract_frames(self,frames: list[np.ndarray],
    #     cleaned_frames: list[np.ndarray],
    #     masks: list[np.ndarray],
    #     segments: list[list[SegmentationResult]] = [],
    #     detections: list[list[DetectionResult]] = []):
    #     pass
    
    # not sure if we could still optimize here ?
    def extract_detected_sections(self,index: int ,frame: np.ndarray,cleaned_frame: np.ndarray, mask: np.ndarray,detection: list[DetectionResult])  -> list[FrameSection]:
        sections = []
        text_only = cv2.bitwise_and(frame,frame,None,mask)
        for result in detection:
            x1,y1,x2,y2 = result.bbox
            draw_bbox = draw_area_bbox(cleaned_frame[y1:y2,x1:x2])
            draw_bbox += np.array([x1,y1,x1,y1],dtype=np.int32)
            dx1,dy1,dx2,dy2 = draw_bbox
            sections.append(FrameSection(index,frame,frame[y1:y2,x1:x2],cleaned_frame[y1:y2,x1:x2],mask[y1:y2,x1:x2],text_only[y1:y2,x1:x2],result.bbox.copy(),cleaned_frame[dy1:dy2,dx1:dx2],draw_bbox))

        return sections
            

    async def extract_detected_sections_batched(self,frames: list[np.ndarray],cleaned_frames: list[np.ndarray],masks: list[np.ndarray],detections: list[list[DetectionResult]]) -> list[FrameSection]:
        result = []
        for sections in await asyncio.gather(*[asyncio.to_thread(self.extract_detected_sections,x,frame,cleaned_frame,mask,detection) for x,frame,cleaned_frame,mask,detection in zip(range(len(frames)),frames,cleaned_frames,masks,detections)]):
            result += sections

        return result
            

    def composite_drawn_sections(self,cleaned_frame: np.ndarray,sections: list[tuple[FrameSection,np.ndarray,np.ndarray]]):
        for section,drawn,drawn_mask in sections:
            x1,y1,x2,y2 = section.draw_bbox
            dest = cleaned_frame[y1:y2,x1:x2]
            # Replace masked regions with white
            a = cv2.bitwise_and(dest,dest, mask=cv2.bitwise_not(drawn_mask))
            b = cv2.bitwise_and(drawn, drawn, mask=drawn_mask)
            cleaned_frame[y1:y2,x1:x2] = cv2.add(a,b)

    async def __call__(
        self,
        batch: list[np.ndarray]
    ) -> list[np.ndarray]:
        
        input_images = [frame.copy() for frame in batch]

        detections,segments = await asyncio.gather(self.detector(input_images),self.segmenter(input_images))

        results = [frame.copy() for frame in input_images]

        translatable_frames: list[TranslatableFrame] = []

        for i in range(len(input_images)):
            if len(detections[i]) == 0:
                # print(f"Skipping {i}, no detections")
                continue

            translatable_frames.append(TranslatableFrame(i,input_images[i],detections[i],segments[i]))

        if len(translatable_frames) == 0:
            return results
        
        raw_translatable_frames = [item.frame for item in translatable_frames]
        detections = [item.detections for item in translatable_frames]
        segments = [item.segments for item in translatable_frames]

        # create 1d segmentation masks for inpainting
        masks = await asyncio.gather(*[asyncio.to_thread(self.make_mask,item.frame,item.segments)  for item in translatable_frames])

        cleaned_frames = await self.cleaner(raw_translatable_frames,masks,segments,detections)

        for i in range(len(cleaned_frames)):
            results[translatable_frames[i].index] = cleaned_frames[i].copy()

        sections = await self.extract_detected_sections_batched(raw_translatable_frames,cleaned_frames,masks,detections)

        ocr_results = await self.ocr([section.text for section in sections])

        valid_ocr_indices = [i for i in range(len(ocr_results)) if len(ocr_results[i].text.strip()) > 0]
        valid_ocr_indices_len = len(valid_ocr_indices)

        sections_dirty = False
        if valid_ocr_indices_len != len(ocr_results):
            x = [ocr_results[i] for i in valid_ocr_indices]
            ocr_results = x
            sections_dirty = True

        if valid_ocr_indices_len > 0:
            translation_results = await self.translator(ocr_results)
            
            valid_translations_incides = [i for i in range(len(translation_results)) if len(translation_results[i].text.strip()) > 0]
            valid_translations_incides_len = len(valid_translations_incides)

            if valid_translations_incides_len != len(translation_results):
                x = [translation_results[i] for i in valid_translations_incides]
                translation_results = x
                sections_dirty = True

            if valid_translations_incides_len > 0:
                if sections_dirty:
                    x = [sections[valid_ocr_indices[valid_translations_incides[i]]] for i in valid_translations_incides]
                    sections = x
                    
                drawn_results = await self.drawer([section.draw_section for section in sections],translation_results)

                frame_sections = [[] for _ in range(len(raw_translatable_frames))]

                for section,drawn_result in zip(sections,drawn_results):
                    frame_sections[section.source_index].append((section,*drawn_result))


                await asyncio.gather(*[asyncio.to_thread(self.composite_drawn_sections,cleaned_frames[i],frame_sections[i]) for i in range(len(raw_translatable_frames))])

                for item,result_frame in zip(translatable_frames,cleaned_frames):
                    results[item.index] = result_frame

        # for x in to_translate:
        #     segments_list = list(map(lambda a: a.points,x.segments))
        #     results[x.index] = cv2.fillPoly(results[x.index], segments_list, (0, 0, 255,50))

        return results
            
        




        
