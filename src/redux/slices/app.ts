import { createAsyncThunk, createSlice, PayloadAction } from "@reduxjs/toolkit";
import {
  IAppSliceState,
  EAppOperation,
  IAppStore,
  IServerInfoResponse,
  EImageFit,
} from "../../types";

const initialState: IAppSliceState = {
  serverAddress: "http://127.0.0.1:5000",
  translatorId: 0,
  translators: [],
  ocrId: 0,
  ocrs: [],
  fontId: 0,
  fonts: [],
  operation: EAppOperation.CLEANING,
  originalImageAddress: "",
  convertedImageAddress: "",
  convertedImageLoaded: false,
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

export const AppSlice = createSlice({
  name: "app",
  // `createSlice` will infer the state type from the `initialState` argument
  initialState,
  reducers: {
    setTranslatorId: (state, action: PayloadAction<number>) => {
      state.translatorId = action.payload;
    },
    setOcrId: (state, action: PayloadAction<number>) => {
      state.ocrId = action.payload;
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
      state.convertedImageLoaded = false;
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
    setConvertedImageLoaded: (state, action: PayloadAction<boolean>) => {
      state.convertedImageLoaded = action.payload;
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
  setConvertedImageLoaded,
  setImageFit,
} = AppSlice.actions;
export { getServerInfo };
export default AppSlice.reducer;
