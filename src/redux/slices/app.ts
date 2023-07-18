import { createAsyncThunk, createSlice, PayloadAction } from "@reduxjs/toolkit";
import {
  IAppSliceState,
  EAppOperation,
  IAppStore,
  IServerInfoResponse,
  EImageFit,
  IPluginArgument,
  IPluginArgumentInfo,
  IServerPayload,
} from "../../types";

function toDefaultArgs(args: IPluginArgument[]): IPluginArgumentInfo[] {
  return args.map((a) => ({ id: a.id, value: a.default }));
}

function argsToPayloadReduce(
  total: Record<string, string>,
  current: IPluginArgumentInfo
) {
  total[current.id] = current.value;
  return total;
}

const initialState: IAppSliceState = {
  serverAddress: "http://127.0.0.1:5000",
  translatorId: 0,
  translators: [],
  ocrId: 0,
  ocrs: [],
  fontId: 0,
  fonts: [],
  translatorArgs: [],
  ocrArgs: [],
  operation: EAppOperation.CLEANING,
  originalImageAddress: "",
  convertedImageAddress: "",
  convertedImageLoading: false,
  imageFit: EImageFit.FIT_TO_PAGE,
};

const getServerInfo = createAsyncThunk<
  IServerInfoResponse | undefined,
  undefined,
  IAppStore
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
>("app/init", async (_, thunk) => {
  try {
    const currentState = thunk.getState();

    const serverAddress = currentState.app.serverAddress;

    const serverApiInfo: IServerInfoResponse = await fetch(
      serverAddress + "/info"
    ).then((a) => a.json());

    return serverApiInfo;
  } catch (e: unknown) {
    // eslint-disable-next-line no-console
    console.error(e);
    return undefined;
  }
});

const performCurrentOperation = createAsyncThunk<
  string | undefined,
  undefined,
  IAppStore
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
>("app/perform", async (_, thunk) => {
  try {
    const store = thunk.getState();
    const state = store.app;

    const data: IServerPayload = {
      image: state.originalImageAddress,
      translator: state.translatorId,
      ocr: state.ocrId,
      translatorArgs: state.translatorArgs.reduce(argsToPayloadReduce, {}),
      ocrArgs: state.ocrArgs.reduce(argsToPayloadReduce, {}),
      font: state.fontId,
    };

    const serverAddress = state.serverAddress;
    return await fetch(
      serverAddress +
        (state.operation === EAppOperation.CLEANING ? "/clean" : "/translate"),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      }
    )
      .then((a) => {
        if (a.status === 500) {
          console.log("Error from server", a.text());
          return undefined;
        }

        return a.blob();
      })
      .then((a) => (a === undefined ? a : URL.createObjectURL(a)));
  } catch (e: unknown) {
    // eslint-disable-next-line no-console
    console.error(e);
    return undefined;
  }
});

export const AppSlice = createSlice({
  name: "app",
  // `createSlice` will infer the state type from the `initialState` argument
  initialState,
  reducers: {
    setTranslatorId: (state, action: PayloadAction<number>) => {
      state.translatorId = action.payload;
      state.translatorArgs = toDefaultArgs(
        state.translators[state.translatorId].args
      );
    },
    setOcrId: (state, action: PayloadAction<number>) => {
      state.ocrId = action.payload;
      state.ocrArgs = toDefaultArgs(state.ocrs[state.ocrId].args);
    },
    setFontId: (state, action: PayloadAction<number>) => {
      state.fontId = action.payload;
    },
    setServerAddress: (state, action: PayloadAction<string>) => {
      state.originalImageAddress = action.payload;
    },
    setImageAddress: (state, action: PayloadAction<string>) => {
      state.originalImageAddress = action.payload;
      state.convertedImageAddress = "";
      state.convertedImageLoading = false;
    },
    setConvertedAddress: (state, action: PayloadAction<string>) => {
      state.convertedImageAddress = action.payload;
    },
    setSelectedOperation: (state, action: PayloadAction<EAppOperation>) => {
      state.operation = action.payload;
    },
    setImageFit: (state, action: PayloadAction<EImageFit>) => {
      state.imageFit = action.payload;
    },
    setConvertedImageLoading: (state, action: PayloadAction<boolean>) => {
      state.convertedImageLoading = action.payload;
    },
    setTranslatorArgument: (
      state,
      action: PayloadAction<{ index: number; value: string }>
    ) => {
      state.translatorArgs[action.payload.index].value = action.payload.value;
    },
    setOcrArgument: (
      state,
      action: PayloadAction<{ index: number; value: string }>
    ) => {
      state.ocrArgs[action.payload.index].value = action.payload.value;
    },
  },
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  extraReducers: (builder) => {
    builder.addCase(getServerInfo.fulfilled, (state, action) => {
      if (action.payload !== undefined) {
        state.translators = action.payload.translators;
        state.ocrs = action.payload.ocr;
        state.fonts = action.payload.fonts;
        state.translatorId = 0;
        state.ocrId = 0;
        state.fontId = 0;
        state.translatorArgs = toDefaultArgs(
          state.translators[state.translatorId].args
        );
        state.ocrArgs = toDefaultArgs(state.ocrs[state.ocrId].args);
      }
    });
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    builder.addCase(performCurrentOperation.pending, (state, _) => {
      if (state.convertedImageAddress != "") {
        URL.revokeObjectURL(state.convertedImageAddress);
        state.convertedImageAddress = "";
      }
      state.convertedImageLoading = true;
    });

    builder.addCase(performCurrentOperation.fulfilled, (state, action) => {
      if (action.payload !== undefined) {
        state.convertedImageAddress = action.payload;
      } else {
        state.convertedImageLoading = false;
      }
    });
  },
});

export const {
  setTranslatorId,
  setOcrId,
  setFontId,
  setServerAddress,
  setImageAddress,
  setSelectedOperation,
  setConvertedAddress,
  setConvertedImageLoading,
  setImageFit,
  setOcrArgument,
  setTranslatorArgument,
} = AppSlice.actions;
export { getServerInfo, performCurrentOperation };
export default AppSlice.reducer;
