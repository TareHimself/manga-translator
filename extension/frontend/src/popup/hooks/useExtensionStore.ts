import browser from "webextension-polyfill"
import { getStorageOrDefaults, StorageDefaults, StorageKeys } from "../../shared/storage"
import { create } from 'zustand'


export type ExtensionStoreState = {
    serverAddress: string
    batchSize: number
    batchTimeout: number
    loading: boolean
};

export type ExtensionStoreActions = {
    setServerAddress: (value: string) => void
    setBatchSize: (value: number) => void
    setBatchTimeout: (value: number) => void
};

export const useExtensionStore = create<ExtensionStoreState & ExtensionStoreActions>((set) => {
    getStorageOrDefaults().then((data) => {
        set({
            serverAddress: data[StorageKeys.ServerAddress],
            batchSize: data[StorageKeys.BatchSize],
            batchTimeout: data[StorageKeys.BatchTimeout],
            loading: false
        })
    })

    return {
        loading: true,
        serverAddress: StorageDefaults.ServerAddress,
        batchSize: StorageDefaults.BatchSize,
        batchTimeout: StorageDefaults.BatchTimeout,
        setServerAddress: (v) => {
            browser.storage.local.set({ [StorageKeys.ServerAddress]: v })
            set({ serverAddress: v })
        },
        setBatchSize: (v) => {
            browser.storage.local.set({ [StorageKeys.BatchSize]: v })
            set({ batchSize: v })
        },
        setBatchTimeout: (v) => {
            browser.storage.local.set({ [StorageKeys.BatchTimeout]: v })
            set({ batchTimeout: v })
        }
    }
})