import { useState } from "react";
import { useAppDispatch, useAppSelector } from "../redux/hooks";
import { setConvertedImageLoaded } from "../redux/slices/app";
import { EImageFit } from "../types";

export default function ImageConverter() {
  const dispatch = useAppDispatch();
  const originalImageAddress = useAppSelector((a) =>
    a.app.originalImageAddress.length == 0
      ? ""
      : a.app.serverAddress + "/images/" + a.app.originalImageAddress
  );
  const convertedImageAddress = useAppSelector(
    (a) => a.app.convertedImageAddress
  );

  const convertedImageLoaded = useAppSelector(
    (a) => a.app.convertedImageLoaded
  );

  const imageFit = useAppSelector((a) => a.app.imageFit);

  const [hasMainImageLoaded, setHasMainImageLoaded] = useState(false);

  if (originalImageAddress.length > 0) {
    return (
      <div
        className="tile"
        data-fit={imageFit === EImageFit.FIT_TO_PAGE ? "page" : "scroll"}
      >
        <img
          className={`original${
            convertedImageLoaded || convertedImageAddress.length === 0
              ? ""
              : " loading"
          }`}
          src={originalImageAddress}
          alt="original"
          onLoadStart={() => {
            setHasMainImageLoaded(false);
          }}
          onLoad={() => {
            setTimeout(() => {
              setHasMainImageLoaded(true);
            }, 1000);
          }}
        />
        {hasMainImageLoaded && (
          <img
            className="converted"
            src={convertedImageAddress}
            alt="converted"
            onLoad={() => {
              if (hasMainImageLoaded) {
                dispatch(setConvertedImageLoaded(true));
              }
            }}
            style={
              convertedImageLoaded && hasMainImageLoaded
                ? undefined
                : { opacity: 0 }
            }
          />
        )}
      </div>
    );
  }

  return <div className="tile"></div>;
}
