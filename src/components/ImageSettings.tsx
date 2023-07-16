import { useCallback, useEffect, useState } from "react";
import { useAppDispatch, useAppSelector } from "../redux/hooks";
import {
  EAppOperation,
  EImageFit,
  IImageSettings,
  IServerArgument,
  IToServerArgument,
} from "../types";
import { AiOutlineCloudUpload } from "react-icons/ai";
import {
  getServerInfo,
  setConvertedAddress,
  setConvertedImageLoaded,
  setFontId,
  setImageAddress,
  setImageFit,
  setOcrId,
  setSelectedOperation,
  setTranslatorId,
} from "../redux/slices/app";
import TileRow from "./TileRow";
import SelectTileRow from "./SelectTileRow";
import ArgsTileColumn from "./ArgsTileColumn";

function toDefaultArgs(args: IServerArgument[]): IToServerArgument[] {
  return args.map((a) => ({ name: a.name, value: "" }));
}

function argsToString(total: string, current: IToServerArgument) {
  return total + "$" + current.name + "=" + current.value;
}

export default function ImageSettings() {
  const dispatch = useAppDispatch();

  const serverUrl = useAppSelector((store) => store.app.serverAddress);
  const operation = useAppSelector((store) => store.app.operation);
  const translatorId = useAppSelector((a) => a.app.translatorId);
  const translators = useAppSelector((a) => a.app.translators);
  const ocrId = useAppSelector((a) => a.app.ocrId);
  const ocrs = useAppSelector((a) => a.app.ocrs);
  const fontId = useAppSelector((a) => a.app.fontId);
  const fonts = useAppSelector((a) => a.app.fonts);
  const imageAddress = useAppSelector((a) => a.app.originalImageAddress);
  const imageFit = useAppSelector((a) => a.app.imageFit);

  const [imageSettings, setImageSettings] = useState<IImageSettings>({
    imageAddress: imageAddress,
    translatorsArgs: toDefaultArgs(translators[translatorId]?.args ?? []),
    ocrArgs: toDefaultArgs(ocrs[ocrId]?.args ?? []),
  });

  const generateImageUrl = useCallback(
    (address: string) => {
      let serverRequestUrl =
        serverUrl +
        `/${operation === EAppOperation.CLEANING ? "clean" : "translate"}`;

      if (operation === EAppOperation.TRANSLATION) {
        serverRequestUrl += `/id=${translatorId}${imageSettings.translatorsArgs.reduce(
          argsToString,
          ""
        )}/id=${ocrId}${imageSettings.ocrArgs.reduce(
          argsToString,
          ""
        )}/id=${fontId}`;
      }

      return (
        serverRequestUrl +
        `/${address}${
          address.includes("?") ? "&" : "?"
        }dummy_timestamp_for_new_data=${Date.now().toFixed()}`
      );
    },
    [
      fontId,
      imageSettings.ocrArgs,
      imageSettings.translatorsArgs,
      ocrId,
      operation,
      serverUrl,
      translatorId,
    ]
  );

  useEffect(() => {
    dispatch(getServerInfo());
  }, [dispatch]);

  return (
    <div className="tile">
      <TileRow name="Image Address Or Path">
        <input
          value={imageSettings.imageAddress}
          type="text"
          onChange={(e) =>
            setImageSettings((a) => ({
              ...a,
              imageAddress: e.target.value.trim(),
            }))
          }
        />
        <AiOutlineCloudUpload
          color="white"
          onClick={() => dispatch(setImageAddress(imageSettings.imageAddress))}
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
              setImageSettings((cur) => ({
                ...cur,
                ocrArgs: toDefaultArgs(ocrs[a].args),
              }));
            }}
          />

          {imageSettings.ocrArgs.length > 0 && (
            <ArgsTileColumn
              category="Ocr"
              args={imageSettings.ocrArgs}
              onArgumentUpdated={(idx, val) => {
                setImageSettings((a) => {
                  a.ocrArgs[idx].value = val;
                  return { ...a };
                });
              }}
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
              setImageSettings((cur) => ({
                ...cur,
                translatorsArgs: toDefaultArgs(translators[a].args),
              }));
            }}
          />

          {imageSettings.translatorsArgs.length > 0 && (
            <ArgsTileColumn
              category="Translator"
              args={imageSettings.translatorsArgs}
              onArgumentUpdated={(idx, val) => {
                setImageSettings((a) => {
                  a.translatorsArgs[idx].value = val;
                  return { ...a };
                });
              }}
            />
          )}
        </>
      )}
      <div className="tile-row">
        <div className="tile-row-content">
          <button
            onClick={() => {
              if (imageAddress != imageSettings.imageAddress) {
                dispatch(setImageAddress(imageSettings.imageAddress));
              }

              if (imageSettings.imageAddress.trim() === "") {
                return;
              }

              dispatch(setConvertedImageLoaded(false));
              dispatch(
                setConvertedAddress(
                  generateImageUrl(imageSettings.imageAddress)
                )
              );
            }}
          >
            {operation === EAppOperation.CLEANING ? "Clean" : "Translate"}
          </button>
        </div>
      </div>
    </div>
  );
}
