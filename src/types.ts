export interface IToServerArgument {
  name: string;
  value: string;
}

export enum EServerArgumentType {
  TEXT = 0,
  SELECT = 1,
}

export type IServerArgument = {
  name: string;
  description: string;
} & (
  | {
      type: EServerArgumentType.TEXT;
    }
  | {
      type: EServerArgumentType.SELECT;
      options: IToServerArgument[];
    }
);


export interface IServerItem {
  id: number;
  name: string;
  description: string;
  args: IServerArgument[];
}

export type IFontItem = Pick<IServerItem, "id" | "name">;

export const enum EAppOperation {
  TRANSLATION,
  CLEANING,
}

export const enum EImageFit {
  FIT_TO_PAGE,
  SCROLL,
}

export interface IServerInfoResponse {
  translators: IServerItem[];
  ocr: IServerItem[];
  fonts: IFontItem[];
}

export interface IImageSettings {
  imageAddress: string;
  translatorsArgs: IToServerArgument[];
  ocrArgs: IToServerArgument[];
}

export interface IAppSliceState {
  serverAddress: string;
  originalImageAddress: string;
  convertedImageAddress: string;
  convertedImageLoaded: boolean;
  translatorId: number;
  translators: IServerItem[];
  ocrId: number;
  fontId: number;
  fonts: IFontItem[];
  ocrs: IServerItem[];
  operation: EAppOperation;
  imageFit: EImageFit;
}

export type IAppStore = {
  state: {
    app: IAppSliceState;
  };
};

export {};
