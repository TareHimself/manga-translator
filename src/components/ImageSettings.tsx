import { useState } from "react";
import { useAppDispatch, useAppSelector } from "../redux/hooks";
import { EAppOperation, EImageFit } from "../types";
import { AiOutlineCloudUpload } from "react-icons/ai";
import {
  performCurrentOperation,
  setFontId,
  setImageAddress,
  setImageFit,
  setOcrArgument,
  setOcrId,
  setSelectedOperation,
  setTranslatorArgument,
  setTranslatorId,
} from "../redux/slices/app";
import TileRow from "./TileRow";
import SelectTileRow from "./SelectTileRow";
import ArgsTileColumn from "./ArgsTileColumn";

export default function ImageSettings() {
  const dispatch = useAppDispatch();

  const operation = useAppSelector((store) => store.app.operation);
  const translatorId = useAppSelector((a) => a.app.translatorId);
  const translators = useAppSelector((a) => a.app.translators);
  const ocrId = useAppSelector((a) => a.app.ocrId);
  const ocrs = useAppSelector((a) => a.app.ocrs);
  const fontId = useAppSelector((a) => a.app.fontId);
  const fonts = useAppSelector((a) => a.app.fonts);
  const imageAddress = useAppSelector((a) => a.app.originalImageAddress);
  const imageFit = useAppSelector((a) => a.app.imageFit);
  const ocrArgs = useAppSelector((a) => a.app.ocrArgs);
  const translatorArgs = useAppSelector((a) => a.app.translatorArgs);

  const [imageToLoad, setImageToLoad] = useState<string>("");

  return (
    <div className="tile">
      <TileRow name="Image Address Or Path">
        <input
          value={imageToLoad}
          type="text"
          onChange={(e) => setImageToLoad(e.target.value)}
        />
        <AiOutlineCloudUpload
          color="white"
          onClick={() =>
            dispatch(setImageAddress(encodeURI(imageToLoad.trim())))
          }
        />
      </TileRow>

      <SelectTileRow
        name="Image Fit"
        items={[EImageFit.FIT_TO_PAGE, EImageFit.SCROLL]}
        value={imageFit}
        toSelectValue={(a) => `${a}`}
        toOriginalValue={parseInt}
        toLabel={(a) =>
          a === EImageFit.FIT_TO_PAGE ? "Fit To Page" : "Scroll"
        }
        onSelected={(a) => {
          dispatch(setImageFit(a));
        }}
      />
      <SelectTileRow
        name="Operation"
        items={[EAppOperation.CLEANING, EAppOperation.TRANSLATION]}
        value={operation}
        toSelectValue={(a) => `${a}`}
        toOriginalValue={parseInt}
        toLabel={(a) => (a === EAppOperation.CLEANING ? "Clean" : "Translate")}
        onSelected={(a) => {
          dispatch(setSelectedOperation(a));
        }}
      />

      {operation === EAppOperation.TRANSLATION && (
        <>
          <SelectTileRow
            name="Font"
            items={fonts.map((a, idx) => idx)}
            value={fontId}
            toSelectValue={(a) => `${a}`}
            toOriginalValue={parseInt}
            toLabel={(a) => fonts[a]?.name ?? "Loading"}
            onSelected={(a) => {
              dispatch(setFontId(a));
            }}
          />

          <SelectTileRow
            name="Ocr"
            items={ocrs.map((a, idx) => idx)}
            value={ocrId}
            toSelectValue={(a) => `${a}`}
            toOriginalValue={parseInt}
            toLabel={(a) => ocrs[a]?.name ?? "Loading"}
            onSelected={(a) => {
              dispatch(setOcrId(a));
            }}
          />

          {ocrArgs.length > 0 && (
            <ArgsTileColumn
              category="Ocr"
              args={ocrArgs}
              argsInfo={ocrs[ocrId].args}
              onArgumentUpdated={(idx, val) =>
                dispatch(setOcrArgument({ index: idx, value: val }))
              }
            />
          )}

          <SelectTileRow
            name="Translator"
            items={translators.map((a, idx) => idx)}
            value={translatorId}
            toSelectValue={(a) => `${a}`}
            toOriginalValue={parseInt}
            toLabel={(a) => translators[a]?.name ?? "Loading"}
            onSelected={(a) => {
              dispatch(setTranslatorId(a));
            }}
          />

          {translatorArgs.length > 0 && (
            <ArgsTileColumn
              category="Translator"
              args={translatorArgs}
              argsInfo={translators[translatorId].args}
              onArgumentUpdated={(idx, val) =>
                dispatch(setTranslatorArgument({ index: idx, value: val }))
              }
            />
          )}
        </>
      )}
      <div className="tile-row">
        <div className="tile-row-content">
          <button
            onClick={() => {
              const currentAddressInUiEncoded = encodeURI(imageToLoad.trim());

              if (currentAddressInUiEncoded.length === 0) {
                dispatch(setImageAddress(""));
                return;
              }

              if (currentAddressInUiEncoded != imageAddress) {
                dispatch(setImageAddress(currentAddressInUiEncoded));
              }

              dispatch(performCurrentOperation());
            }}
          >
            {operation === EAppOperation.CLEANING ? "Clean" : "Translate"}
          </button>
        </div>
      </div>
    </div>
  );
}
