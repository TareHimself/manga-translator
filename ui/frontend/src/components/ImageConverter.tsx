import { useState } from "react";
import { EImageFit } from "../types";
import { useAppStore } from "../useAppStore";

export default function ImageConverter() {
  const originalImageAddress = useAppStore((s) => s.originalImageAddress)
  const convertedImageAddress = useAppStore((s) => s.convertedImageAddress)

  const convertedImageLoading = useAppStore((s) => s.convertedImageLoading)
  const setConvertedImageLoading = useAppStore((s) => s.setConvertedImageLoading)

  const imageFit = useAppStore((s) => s.imageFit)

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
                setConvertedImageLoading(false)
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
