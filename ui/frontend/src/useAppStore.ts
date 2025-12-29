import { create } from 'zustand'
// import { devtools, persist } from 'zustand/middleware'
import { EAppOperation, EImageFit, type IPluginArgument, type IPluginArgumentInfo, type IServerInfoResponse, type IPlugin, type IServerPayload } from './types';
import { parse as parseYaml,stringify as stringifyYaml } from 'yaml'

function toDefaultArgs(args: IPluginArgument[]): IPluginArgumentInfo[] {
  return args.map((a) => ({ id: a.id, value: a.default }));
}

function argsToPayloadReduce(
  total: Record<string, unknown>,
  current: IPluginArgumentInfo
) {
  total[current.id] = current.value;
  return total;
}

export type AppStoreState = {
  serverAddress: string;
  originalImageAddress: string;
  convertedImageAddress: string;
  convertedImageLoading: boolean;
  detectors: IPlugin[];
  segmenters: IPlugin[];
  translators: IPlugin[];
  drawers: IPlugin[];
  cleaners: IPlugin[];
  ocrs: IPlugin[];
  detectorIndex: number;
  segmenterIndex: number;
  translatorIndex: number;
  ocrIndex: number;
  drawerIndex: number;
  cleanerIndex: number;
  detectorArgs: IPluginArgumentInfo[];
  segmenterArgs: IPluginArgumentInfo[];
  translatorArgs: IPluginArgumentInfo[];
  ocrArgs: IPluginArgumentInfo[];
  drawerArgs: IPluginArgumentInfo[];
  cleanerArgs: IPluginArgumentInfo[];
  operation: EAppOperation;
  imageFit: EImageFit;
}

export type AppStoreActions = {
  setDetectorIndex: (id: number) => void
  setSegmenterIndex: (id: number) => void
  setTranslatorIndex: (id: number) => void
  setOcrIndex: (id: number) => void
  setDrawerIndex: (id: number) => void
  setCleanerIndex: (id: number) => void
  setServerAddress: (address: string) => void
  setImageAddress: (address: string) => void
  setConvertedAddress: (address: string) => void
  setSelectedOperation: (operation: EAppOperation) => void
  setImageFit: (fit: EImageFit) => void
  setConvertedImageLoading: (state: boolean) => void
  setDetectorArgument: (index: number, value: unknown) => void
  setSegmenterArgument: (index: number, value: unknown) => void
  setTranslatorArgument: (index: number, value: unknown) => void
  setOcrArgument: (index: number, value: unknown) => void
  setDrawerArgument: (index: number, value: unknown) => void
  setCleanerArgument: (index: number, value: unknown) => void
  getServerInfo: () => Promise<IServerInfoResponse>
  performCurrentOperation: () => Promise<string>
  loadConfig: (file: File) => Promise<void>
  makeServerPayload: () => IServerPayload
  exportConfig: () => string
}

export const useAppStore = create<AppStoreState & AppStoreActions>((set, get) => ({
  serverAddress: window.location.origin,
  detectorIndex: 0,
  detectors: [],
  segmenterIndex: 0,
  segmenters: [],
  translatorIndex: 0,
  translators: [],
  ocrIndex: 0,
  ocrs: [],
  drawerIndex: 0,
  drawers: [],
  cleanerIndex: 0,
  cleaners: [],
  detectorArgs: [],
  segmenterArgs: [],
  translatorArgs: [],
  ocrArgs: [],
  drawerArgs: [],
  cleanerArgs: [],
  operation: EAppOperation.CLEANING,
  originalImageAddress: "",
  convertedImageAddress: "",
  convertedImageLoading: false,
  imageFit: EImageFit.FIT_TO_PAGE,
  setDetectorIndex: (id) => set({ detectorIndex: id, translatorArgs: toDefaultArgs(get().detectors[id].args) }),
  setSegmenterIndex: (id) => set({ segmenterIndex: id, translatorArgs: toDefaultArgs(get().segmenters[id].args) }),
  setTranslatorIndex: (id) => set({ translatorIndex: id, translatorArgs: toDefaultArgs(get().translators[id].args) }),
  setOcrIndex: (id) => set({ ocrIndex: id, ocrArgs: toDefaultArgs(get().ocrs[id].args) }),
  setDrawerIndex: (id) => set({ drawerIndex: id, drawerArgs: toDefaultArgs(get().drawers[id].args) }),
  setCleanerIndex: (id) => set({ cleanerIndex: id, cleanerArgs: toDefaultArgs(get().cleaners[id].args) }),
  setServerAddress: (address) => set({ serverAddress: address }),
  setImageAddress: (address) => set({ originalImageAddress: address, convertedImageAddress: '', convertedImageLoading: false }),
  setConvertedAddress: (address) => set({ convertedImageAddress: address }),
  setSelectedOperation: (operation) => set({ operation }),
  setImageFit: (fit) => set({ imageFit: fit }),
  setConvertedImageLoading: (state) => set({ convertedImageLoading: state }),
  setDetectorArgument: (index, value) => {
    const args = get().detectorArgs
    args[index].value = value
    set({ detectorArgs: [...args] })
  },
  setSegmenterArgument: (index, value) => {
    const args = get().segmenterArgs
    args[index].value = value
    set({ segmenterArgs: [...args] })
  },
  setTranslatorArgument: (index, value) => {
    const args = get().translatorArgs
    args[index].value = value
    set({ translatorArgs: [...args] })
  },
  setOcrArgument: (index, value) => {
    const args = get().ocrArgs
    args[index].value = value
    set({ ocrArgs: [...args] })
  },
  setDrawerArgument: (index, value) => {
    const args = get().drawerArgs
    args[index].value = value
    set({ drawerArgs: [...args] })
  },
  setCleanerArgument: (index, value) => {
    const args = get().cleanerArgs
    args[index].value = value
    set({ cleanerArgs: [...args] })
  },
  getServerInfo: async () => {

    const { serverAddress } = get()
    const serverApiInfo: IServerInfoResponse = await fetch(
      serverAddress + "/info"
    ).then((a) => a.json());


    set({
      detectors: serverApiInfo.detectors,
      segmenters: serverApiInfo.segmenters,
      translators: serverApiInfo.translators,
      ocrs: serverApiInfo.ocrs,
      cleaners: serverApiInfo.cleaners,
      drawers: serverApiInfo.drawers,
      detectorIndex: 0,
      segmenterIndex: 0,
      translatorIndex: 0,
      ocrIndex: 0,
      cleanerIndex: 0,
      detectorArgs: toDefaultArgs(serverApiInfo.detectors[0].args),
      segmenterArgs: toDefaultArgs(serverApiInfo.segmenters[0].args),
      translatorArgs: toDefaultArgs(serverApiInfo.translators[0].args),
      ocrArgs: toDefaultArgs(serverApiInfo.ocrs[0].args),
      cleanerArgs: toDefaultArgs(serverApiInfo.cleaners[0].args),
      drawerArgs: toDefaultArgs(serverApiInfo.drawers[0].args)
    })

    console.log(serverApiInfo)

    return serverApiInfo;
  },
  performCurrentOperation: async () => {

    try {
      const { operation, serverAddress, originalImageAddress, convertedImageAddress } = get()

      if (convertedImageAddress !== "") {
        URL.revokeObjectURL(convertedImageAddress);
        set({ convertedImageAddress: "" })
      }
      set({ convertedImageLoading: true })

      const data = get().makeServerPayload()

      const formData = new FormData()
      formData.append('data', JSON.stringify(data))
      formData.append('file', await fetch(originalImageAddress).then(a => a.blob()))
      const convertedAddress = await fetch(
        serverAddress +
        (operation === EAppOperation.CLEANING ? "/clean" : "/translate"),
        {
          method: "POST",
          body: formData,
        }
      )
        .then(async (a) => {
          if (a.status === 500) {
            throw new Error(await a.text())
          }

          return a.blob();
        })
        .then(URL.createObjectURL);
      set({ convertedImageAddress: convertedAddress, convertedImageLoading: false })
      return convertedImageAddress
    } catch (error) {
      set({ convertedImageLoading: false })
      throw error
    }
  },
  loadConfig: async (file) => {
    const textData = await file.text()
    const yaml = parseYaml(textData)
    const data = yaml["pipeline"]

    const detector = data["detector"]
    const segmenter = data["segmenter"]
    const translator = data["translator"]
    const cleaner = data["cleaner"]
    const drawer = data["drawer"]
    const ocr = data["ocr"]
    const state = get()
    const newState: Partial<AppStoreState> = {}

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const tryUpdateState = (prefix: keyof IServerPayload, inputData: any, newState: any) => {
      const idx = state[`${prefix}s`].findIndex(c => c.id === inputData["class"])
      newState[`${prefix}Index`] = idx === -1 ? 0 : idx
      const plugin = state[`${prefix}s`][idx]
      if (plugin) {
        newState[`${prefix}Args`] = toDefaultArgs(state[`${prefix}s`][idx].args)
        if (inputData["args"]) {
          const stateArgs = newState[`${prefix}Args`] as IPluginArgumentInfo[]
          for (const arg of Object.entries(inputData["args"])) {
            const stateArg = stateArgs.find(c => c.id === arg[0])
            if (stateArg) {
              stateArg.value = arg[1]
            }
          }
        }
      }
    }
    if (detector) {
      tryUpdateState("detector", detector, newState)
    }
    if (segmenter) {
      tryUpdateState("segmenter", segmenter, newState)
    }
    if (translator) {
      tryUpdateState("translator", translator, newState)
    }
    if (cleaner) {
      tryUpdateState("cleaner", cleaner, newState)
    }
    if (drawer) {
      tryUpdateState("drawer", drawer, newState)
    }
    if (ocr) {
      tryUpdateState("ocr", ocr, newState)
    }
    console.log("YAML",data,newState)
    set(newState)
  },
  makeServerPayload: () => {
    const {  detectorIndex, segmenterIndex, translatorIndex, ocrIndex, drawerIndex, cleanerIndex, detectors, segmenters, translators, ocrs, drawers, cleaners, detectorArgs, segmenterArgs, translatorArgs, ocrArgs, drawerArgs, cleanerArgs } = get()
      const data: IServerPayload = {
        detector: {
          id: detectors[detectorIndex].id,
          args: detectorArgs.reduce(argsToPayloadReduce, {})
        },
        segmenter: {
          id: segmenters[segmenterIndex].id,
          args: segmenterArgs.reduce(argsToPayloadReduce, {})
        },
        translator: {
          id: translators[translatorIndex].id,
          args: translatorArgs.reduce(argsToPayloadReduce, {})
        },
        ocr: {
          id: ocrs[ocrIndex].id,
          args: ocrArgs.reduce(argsToPayloadReduce, {})
        },
        drawer: {
          id: drawers[drawerIndex].id,
          args: drawerArgs.reduce(argsToPayloadReduce, {})
        },
        cleaner: {
          id: cleaners[cleanerIndex].id,
          args: cleanerArgs.reduce(argsToPayloadReduce, {})
        }
      };

      return data
  },
  exportConfig: () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const data: any = get().makeServerPayload()
    for(const key of Object.keys(data)){
      data[key]["class"] = data[key]["id"]
      delete data[key]["id"]
    }
    return stringifyYaml({
      pipeline: data
    },{
      
    }).replaceAll("{}","")
  }
}))