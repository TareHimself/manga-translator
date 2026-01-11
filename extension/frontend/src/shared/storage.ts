import browser from "webextension-polyfill"
export const enum StorageKeys {
    ServerAddress = "serverAddress",
    BatchSize = "batchSize",
    BatchTimeout = "batchTimeout"
}

export const StorageDefaults = {
    ServerAddress : "http://127.0.0.1:9000/api/v1",
    BatchSize : 4,
    BatchTimeout : 500
}

interface IGetStorageOrDefaultsResult { [StorageKeys.ServerAddress]: string, [StorageKeys.BatchSize]: number, [StorageKeys.BatchTimeout]: number }
export const getStorageOrDefaults = (): Promise<IGetStorageOrDefaultsResult> => {
    return browser.storage.local.get({ [StorageKeys.ServerAddress]: StorageDefaults.ServerAddress, [StorageKeys.BatchSize]: StorageDefaults.BatchSize, [StorageKeys.BatchTimeout]: StorageDefaults.BatchTimeout }) as unknown as  Promise<IGetStorageOrDefaultsResult>
}
