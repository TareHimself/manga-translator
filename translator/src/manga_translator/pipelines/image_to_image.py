from manga_translator.core.pipeline import Pipeline
from manga_translator.core.plugin import DetectionResult, Translator, Detector, Segmenter, SegmentationResult, Cleaner, Drawer, OCR, ColorDetector
import numpy as np
import cv2
import asyncio
import torch
from manga_translator.core.typing import Vector4i
from manga_translator.utils import  compute_draw_bbox, perf_async


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

# TODO: Currently text that is not translated is still erased, might wan to change this in the future
class ImageToImagePipeline(Pipeline):
    def __init__(self,detector: Detector = Detector(), segmenter: Segmenter = Segmenter(),translator: Translator = Translator(),cleaner: Cleaner = Cleaner(),drawer: Drawer = Drawer(),ocr: OCR = OCR(),color_detector: ColorDetector = ColorDetector()):
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
            draw_bbox = compute_draw_bbox(cleaned_frame[y1:y2,x1:x2])
            draw_bbox += np.array([x1,y1,x1,y1],dtype=np.int32)
            dx1,dy1,dx2,dy2 = draw_bbox
            sections.append(FrameSection(index,frame,frame[y1:y2,x1:x2],cleaned_frame[y1:y2,x1:x2],mask[y1:y2,x1:x2],text_only[y1:y2,x1:x2],result.bbox.copy(),cleaned_frame[dy1:dy2,dx1:dx2],draw_bbox))

        return sections
            

    async def extract_detected_sections_batched(self,frames: list[np.ndarray],cleaned_frames: list[np.ndarray],masks: list[np.ndarray],detections: list[list[DetectionResult]]) -> list[FrameSection]:
        result = []
        for sections in await asyncio.gather(*[asyncio.to_thread(self.extract_detected_sections,x,frame,cleaned_frame,mask,detection) for x,frame,cleaned_frame,mask,detection in zip(range(len(frames)),frames,cleaned_frames,masks,detections)]):
            result += sections

        return result
    

    
    def clean_frame_using_masks_and_detections(self,frame: np.ndarray,cleaned_frame: np.ndarray,mask: np.ndarray,frame_detections: list[DetectionResult]) -> np.ndarray:

        # make a mask of the detection boxes
        detections_mask = np.zeros_like(mask)
        for detection in frame_detections:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(detections_mask, (x1, y1), (x2, y2), 255, thickness=-1)
        
        # composite the frame with the cleaned frame using the detection mask
        a = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(detections_mask))
        b = cv2.bitwise_and(cleaned_frame, cleaned_frame, mask=detections_mask)
        return cv2.add(a, b)
    
    async def clean_frames_using_masks_and_detections(self,frames: list[np.ndarray],cleaned_frames: list[np.ndarray],masks: list[np.ndarray],detections: list[list[DetectionResult]]) -> list[np.ndarray]:
        return await asyncio.gather(*[asyncio.to_thread(self.clean_frame_using_masks_and_detections,*args) for args in zip(frames,cleaned_frames,masks,detections)])
            

    def composite_drawn_sections(self,sections: list[tuple[FrameSection,np.ndarray,np.ndarray]]):
        for section,_,_ in sections:
            section.section[:] = section.cleaned_section

        for section,drawn,drawn_mask in sections:
            x1,y1,x2,y2 = section.draw_bbox
            
            cleaned = section.source[y1:y2,x1:x2]
            # Replace masked regions with white
            a = cv2.bitwise_and(cleaned,cleaned, mask=cv2.bitwise_not(drawn_mask))
            b = cv2.bitwise_and(drawn, drawn, mask=drawn_mask)
            
            cv2.add(a,b,dst=section.source[y1:y2,x1:x2])

    @perf_async
    async def __call__(
        self,
        batch: list[np.ndarray]
    ) -> list[np.ndarray]:
        results = [frame.copy() for frame in batch]

        detections,segments = await asyncio.gather(self.detector(results),self.segmenter(results))

        translatable_frames: list[TranslatableFrame] = []

        # translate only input with detections
        for i in range(len(results)):
            if len(detections[i]) == 0:
                # print(f"Skipping {i}, no detections")
                continue

            translatable_frames.append(TranslatableFrame(i,results[i],detections[i],segments[i]))

        # if no detections no work
        if len(translatable_frames) == 0:
            return results
        
        # prep for other plugins
        raw_translatable_frames = [item.frame for item in translatable_frames]
        detections = [item.detections for item in translatable_frames]
        segments = [item.segments for item in translatable_frames]

        # produce segmentation masks that will be used for inpainting (Maybe we push this to the cleaner)
        # create 1d segmentation masks for inpainting
        masks = await asyncio.gather(*[asyncio.to_thread(self.make_mask,item.frame,item.segments)  for item in translatable_frames])

        # Produce cleaned frames
        cleaned_frames = await self.cleaner(raw_translatable_frames,masks,segments,detections)


        final_cleaned_frames = await self.clean_frames_using_masks_and_detections(raw_translatable_frames,cleaned_frames,masks,detections)

        cleaned_frames = final_cleaned_frames
        
        # if the default class is passed in, assume we just want the images cleaned
        if type(self.ocr) is OCR:
            for i in range(len(cleaned_frames)):
                results[translatable_frames[i].index] = cleaned_frames[i]
            
            return results
            
            

        # Removed because we only want to clean as much as we need to
        # for i in range(len(cleaned_frames)):
        #     results[translatable_frames[i].index] = cleaned_frames[i].copy()

        # each detection is a section so this is a flat list of all detections (maybe we do some kind of grouping here to help paralelism) in the future
        sections = await self.extract_detected_sections_batched(raw_translatable_frames,cleaned_frames,masks,detections)

        # perform ocr on each detection
        ocr_results = await self.ocr([section.text for section in sections])

        # no point in working on frames with no text detected
        valid_ocr_indices = [i for i in range(len(ocr_results)) if len(ocr_results[i].text.strip()) > 0]

        resize_sections = False

        # Reduce ocr results to only valid ones
        if len(valid_ocr_indices) != len(ocr_results):
            x = [ocr_results[i] for i in valid_ocr_indices]
            ocr_results = x
            resize_sections = True

        # if no ocr we can't translate anything
        if len(valid_ocr_indices) > 0:
            translation_results = await self.translator(ocr_results)

            valid_translation_indices = [i for i in range(len(translation_results)) if len(translation_results[i].text.strip()) > 0]

            # Reduce translation results and ocr_indices to only valid ones
            if len(valid_translation_indices) != len(translation_results):
                x = [translation_results[i] for i in valid_translation_indices]
                #y = [valid_ocr_indices[i] for i in valid_translation_indices]
                translation_results = x
                #valid_ocr_indices = y
                resize_sections = True
            
            if len(valid_translation_indices) > 0:

                # reduce the sections to valid ones
                if resize_sections:
                    x = [sections[valid_ocr_indices[i]] for i in valid_translation_indices]
                    sections = x
                    
                # only do color detection on valid ocr and translation results
                color_detection_results = await self.color_detector([x.text for x in sections])

                drawn_results = await self.drawer([section.draw_section for section in sections],translation_results,color_detection_results)

                frame_sections = [[] for _ in range(len(raw_translatable_frames))]

                for section,drawn_result in zip(sections,drawn_results):
                    frame_sections[section.source_index].append((section,*drawn_result))

                await asyncio.gather(*[asyncio.to_thread(self.composite_drawn_sections,frame_sections[i]) for i in range(len(raw_translatable_frames))])

                # for item,result_frame in zip(translatable_frames,cleaned_frames):
                #     results[item.index] = result_frame

        # for x in to_translate:
        #     segments_list = list(map(lambda a: a.points,x.segments))
        #     results[x.index] = cv2.fillPoly(results[x.index], segments_list, (0, 0, 255,50))

        return results
            
        




        
