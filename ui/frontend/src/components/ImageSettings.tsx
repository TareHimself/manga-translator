import {  useRef } from "react";
import { EAppOperation, EImageFit } from "../types";
import { FaFileUpload } from "react-icons/fa";
import TileRow from "./TileRow";
import ArgsTileColumn from "./ArgsTileColumn";
import { useAppStore } from "../useAppStore";
import { Button, Flex, Select } from "@mantine/core";
import PluginSelect from "./PluginSelect";
import { download } from "../utils";

export default function ImageSettings() {
  const operation = useAppStore((s) => s.operation);
  const detectorIndex = useAppStore((s) => s.detectorIndex);
  const detectors = useAppStore((s) => s.detectors);
  const segmenterIndex = useAppStore((s) => s.segmenterIndex);
  const segmenters = useAppStore((s) => s.segmenters);
  const translatorIndex = useAppStore((s) => s.translatorIndex);
  const translators = useAppStore((s) => s.translators);
  const ocrIndex = useAppStore((s) => s.ocrIndex);
  const ocrs = useAppStore((s) => s.ocrs);
  const drawerIndex = useAppStore((s) => s.drawerIndex);
  const drawers = useAppStore((s) => s.drawers);
  const cleanerIndex = useAppStore((s) => s.cleanerIndex);
  const cleaners = useAppStore((s) => s.cleaners);
  const imageFit = useAppStore((s) => s.imageFit);
  const ocrArgs = useAppStore((s) => s.ocrArgs);
  const detectorArgs = useAppStore((s) => s.detectorArgs);
  const segmenterArgs = useAppStore((s) => s.segmenterArgs);
  const translatorArgs = useAppStore((s) => s.translatorArgs);
  const drawerArgs = useAppStore((s) => s.drawerArgs);
  const cleanerArgs = useAppStore((s) => s.cleanerArgs);
  const setImageAddress = useAppStore((s) => s.setImageAddress);
  const setImageFit = useAppStore((s) => s.setImageFit);
  const setSelectedOperation = useAppStore((s) => s.setSelectedOperation);
  const setCleanerIndex = useAppStore((s) => s.setCleanerIndex);
  const setCleanerArgument = useAppStore((s) => s.setCleanerArgument);
  const setDetectorIndex = useAppStore((s) => s.setDetectorIndex);
  const setDetectorArgument = useAppStore((s) => s.setDetectorArgument);
  const setSegmenterIndex = useAppStore((s) => s.setSegmenterIndex);
  const setSegmenterArgument = useAppStore((s) => s.setSegmenterArgument);
  const setTranslatorIndex = useAppStore((s) => s.setTranslatorIndex);
  const setTranslatorArgument = useAppStore((s) => s.setTranslatorArgument);
  const setDrawerIndex = useAppStore((s) => s.setDrawerIndex);
  const setDrawerArgument = useAppStore((s) => s.setDrawerArgument);
  const setOcrIndex = useAppStore((s) => s.setOcrIndex);
  const setOcrArgument = useAppStore((s) => s.setOcrArgument);
  const performCurrentOperation = useAppStore((s) => s.performCurrentOperation);
  const loadConfig = useAppStore((s) => s.loadConfig)
  const exportConfig = useAppStore((s) => s.exportConfig)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const configInputRef = useRef<HTMLInputElement>(null)

  return (
    <div className="tile">
      <TileRow name="Upload Image">
        {/* <input
          value={imageToLoad}
          type="text"
          onChange={(e) => setImageToLoad(e.target.value)}
        /> */}
        <input
          ref={imageInputRef}
          type="file"
          onChange={(e) => {
            const result = e.target.files?.item(0);
            if (result instanceof Blob) {
              setImageAddress(URL.createObjectURL(result));
            }
          }}
          style={{ display: "none" }}
          accept="image/*"
        />
        <input
          ref={configInputRef}
          type="file"
          onChange={(e) => {
            if(e.target.files !== null){
              loadConfig(e.target.files[0])
              e.target.value = ""
            }
          }}
          style={{ display: "none" }}
          accept=".yaml,.yml,application/x-yaml"
        />
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            width: "100%",
            height: "100%",
            gap: "20px",
          }}
        >
          <Flex>
            <Button
            fullWidth
            className="upload"
            onClick={() => imageInputRef.current?.click()}
          >
            <FaFileUpload />
          </Button>
          </Flex>
          <Flex gap="lg">
            <Button
            fullWidth
            onClick={() => configInputRef.current?.click()}
          >
            Load Config
          </Button>
          <Button
            fullWidth
            onClick={() => {
              download("config.yaml",exportConfig())
            }}
          >
            Save Config
          </Button>
          </Flex>
        </div>
      </TileRow>

      <Select
        allowDeselect={false}
        label="Image Fit"
        data={[
          {
            label: "Fit To Page",
            value: EImageFit.FIT_TO_PAGE.toString(),
          },
          {
            label: "Scroll",
            value: EImageFit.SCROLL.toString(),
          },
        ]}
        value={imageFit.toString()}
        onChange={(value) => {
          setImageFit(parseInt(value ?? "0"));
        }}
        style={{ width: "60%" }}
      />

      <Select
        allowDeselect={false}
        label="Operation"
        data={[
          {
            label: "Clean",
            value: EAppOperation.CLEANING.toString(),
          },
          {
            label: "Translate",
            value: EAppOperation.TRANSLATION.toString(),
          },
        ]}
        value={operation.toString()}
        onChange={(value) => {
          setSelectedOperation(parseInt(value ?? "0"));
        }}
        style={{ width: "60%" }}
      />

      <PluginSelect
        name="Detector"
        selected={detectorIndex}
        items={detectors}
        onChange={setDetectorIndex}
      />

      {detectorArgs.length > 0 && (
        <ArgsTileColumn
          category="Detector"
          args={detectorArgs}
          argsInfo={detectors[detectorIndex].args}
          onArgumentUpdated={(idx, val) => setDetectorArgument(idx, val)}
        />
      )}

      <PluginSelect
        name="Segmenter"
        selected={segmenterIndex}
        items={segmenters}
        onChange={setSegmenterIndex}
      />

      {segmenterArgs.length > 0 && (
        <ArgsTileColumn
          category="Segmenter"
          args={segmenterArgs}
          argsInfo={segmenters[segmenterIndex].args}
          onArgumentUpdated={(idx, val) => setSegmenterArgument(idx, val)}
        />
      )}

      <PluginSelect
        name="Cleaner"
        items={cleaners}
        selected={cleanerIndex}
        onChange={setCleanerIndex}
      />

      {cleanerArgs.length > 0 && (
        <ArgsTileColumn
          category="Cleaner"
          args={cleanerArgs}
          argsInfo={cleaners[cleanerIndex].args}
          onArgumentUpdated={(idx, val) => setCleanerArgument(idx, val)}
        />
      )}

      {operation === EAppOperation.TRANSLATION && (
        <>
          <PluginSelect
            name="Ocr"
            items={ocrs}
            selected={ocrIndex}
            onChange={setOcrIndex}
          />

          {ocrArgs.length > 0 && (
            <ArgsTileColumn
              category="Ocr"
              args={ocrArgs}
              argsInfo={ocrs[ocrIndex].args}
              onArgumentUpdated={(idx, val) => setOcrArgument(idx, val)}
            />
          )}

          <PluginSelect
            name="Translator"
            items={translators}
            selected={translatorIndex}
            onChange={setTranslatorIndex}
          />

          {translatorArgs.length > 0 && (
            <ArgsTileColumn
              category="Translator"
              args={translatorArgs}
              argsInfo={translators[translatorIndex].args}
              onArgumentUpdated={(idx, val) => setTranslatorArgument(idx, val)}
            />
          )}

          <PluginSelect
            name="Drawer"
            items={drawers}
            selected={drawerIndex}
            onChange={setDrawerIndex}
          />

          {drawerArgs.length > 0 && (
            <ArgsTileColumn
              category="Drawer"
              args={drawerArgs}
              argsInfo={drawers[drawerIndex].args}
              onArgumentUpdated={(idx, val) => setDrawerArgument(idx, val)}
            />
          )}
        </>
      )}
      <div className="tile-row">
        <div className="tile-row-content">
          <Button
            fullWidth
            onClick={() => {
              performCurrentOperation();
            }}
          >
            {operation === EAppOperation.CLEANING ? "Clean" : "Translate"}
          </Button>
        </div>
      </div>
    </div>
  );
}
