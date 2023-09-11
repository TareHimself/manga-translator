import { useState } from "react";
import { useAppDispatch, useAppSelector } from "../redux/hooks";
import { setConvertedImageLoading } from "../redux/slices/app";
import { EImageFit } from "../types";

export default function ImageConverter() {
  const dispatch = useAppDispatch();
  const originalImageAddress = useAppSelector((a) => a.app.originalImageAddress);
  const convertedImageAddress = useAppSelector(
    (a) => a.app.convertedImageAddress
  );

  const convertedImageLoading = useAppSelector(
    (a) => a.app.convertedImageLoading
  );

  const imageFit = useAppSelector((a) => a.app.imageFit);

  const [hasMainImageLoaded, setHasMainImageLoaded] = useState(false);

  if (originalImageAddress.length > 0) {
    return (
      <div
        className="tile"
        data-fit={imageFit === EImageFit.FIT_TO_PAGE ? "page" : "scroll"}
        style={{ justifyContent: "flex-start"}}
      >
        <img
          className={`original${!convertedImageLoading ? "" : " loading"}`}
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
        {hasMainImageLoaded && convertedImageAddress.length > 0 && (
          <img
            className="converted"
            src={convertedImageAddress}
            alt="converted"
            onLoad={() => {
              if (hasMainImageLoaded) {
                dispatch(setConvertedImageLoading(false));
              }
            }}
            style={
              !convertedImageLoading &&
              hasMainImageLoaded &&
              convertedImageAddress.length > 0
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
