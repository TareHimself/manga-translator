export interface IPluginArgumentInfo {
  id: string;
  value: string;
}

export interface IPluginSelectArgument {
  name: string;
  value: string;
}

export enum EServerArgumentType {
  TEXT = 0,
  SELECT = 1,
}

export type IPluginArgument = {
  id: string;
  name: string;
  description: string;
  default: string;
} & (
  | {
      type: EServerArgumentType.TEXT;
    }
  | {
      type: EServerArgumentType.SELECT;
      options: IPluginSelectArgument[];
    }
);

export interface IServerItem {
  id: number;
  name: string;
  description: string;
  args: IPluginArgument[];
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
  translatorsArgs: IPluginArgumentInfo[];
  ocrArgs: IPluginArgumentInfo[];
}

export interface IAppSliceState {
  serverAddress: string;
  originalImageAddress: string;
  convertedImageAddress: string;
  convertedImageLoading: boolean;
  translators: IServerItem[];
  fonts: IFontItem[];
  ocrs: IServerItem[];
  translatorId: number;
  ocrId: number;
  fontId: number;
  translatorArgs: IPluginArgumentInfo[];
  ocrArgs: IPluginArgumentInfo[];
  operation: EAppOperation;
  imageFit: EImageFit;
}

export interface IServerPayload {
  image: string;
  translator: number;
  ocr: number;
  translatorArgs: Record<string, string>;
  ocrArgs: Record<string, string>;
  font: number;
}

export type IAppStore = {
  state: {
    app: IAppSliceState;
  };
};

export {};
