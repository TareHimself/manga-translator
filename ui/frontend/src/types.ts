export interface IPluginArgumentInfo {
  id: string;
  value: unknown;
}

export interface IPluginSelectArgument {
  name: string;
  value: string;
}

export enum EServerArgumentType {
  STRING = 0,
  SELECT = 1,
  INT = 2,
  BOOLEAN = 3,
}

export type IPluginArgument = {
  id: string;
  name: string;
  description: string;
} & (
  | {
      type: EServerArgumentType.STRING;
      default: string;
    }
  | {
      type: EServerArgumentType.SELECT;
      options: IPluginSelectArgument[];
      default: string;
    }
    | {
      type: EServerArgumentType.INT;
      default: number;
    }
    | {
      type: EServerArgumentType.BOOLEAN;
      default: boolean;
    }
);

export interface IPlugin {
  id: string;
  name: string;
  description: string;
  args: IPluginArgument[];
}

export type IFontItem = Pick<IPlugin, "id" | "name">;

export const enum EAppOperation {
  TRANSLATION,
  CLEANING,
}

export const enum EImageFit {
  FIT_TO_PAGE,
  SCROLL,
}

export interface IServerInfoResponse {
  detectors: IPlugin[];
  segmenters: IPlugin[];
  translators: IPlugin[];
  ocrs: IPlugin[];
  cleaners: IPlugin[];
  drawers: IPlugin[];
}

// export interface IImageSettings {
//   imageAddress: string;
//   translatorsArgs: IPluginArgumentInfo[];
//   cleaner: IPluginArgumentInfo[];
//   ocrArgs: IPluginArgumentInfo[];
// }

export interface IServerPayloadComponent {
  id: string
  args: Record<string, unknown>
}

export interface IServerPayload {
  detector: IServerPayloadComponent;
  segmenter: IServerPayloadComponent;
  translator: IServerPayloadComponent;
  ocr: IServerPayloadComponent;
  drawer: IServerPayloadComponent;
  cleaner: IServerPayloadComponent;
}


export {};
