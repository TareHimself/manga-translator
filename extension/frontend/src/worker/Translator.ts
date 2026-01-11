import browser from "webextension-polyfill"
import { getStorageOrDefaults, StorageDefaults, StorageKeys } from "../shared/storage"
import type { SerializedImage } from "../shared/types"

type TranslationJob = {
    id: string
    image: Blob
    resolve: (url: string) => void
    reject: (val: unknown) => void
}

interface IServerJsonResponse {
    urls: string[]
}
export class Translator {

    pendingJobs: TranslationJob[] = []
    jobSet: Set<string> = new Set()
    pendingTranslateTimeout: ReturnType<typeof setTimeout> | undefined = undefined
    batchSize: number = 4
    batchTimeout: number = 500
    serverAddress: string = StorageDefaults.ServerAddress
    isTranslating: boolean = false
    constructor() {

    }

    async translateBatch() {
        this.isTranslating = true
        const batch = this.pendingJobs.splice(0, this.batchSize);
        console.log("translating batch of size", batch.map(c => c.id))
        try {
            const formData = new FormData()
            for (const item of batch) {
                formData.append('file', item.image)
            }
            const response = await fetch(`${this.serverAddress}/translate`,
                {
                    method: "POST",
                    body: formData,
                }
            )
                .then(async (a) => {
                    if (a.status === 500) {
                        throw new Error(await a.text())
                    }

                    return a.json() as Promise<IServerJsonResponse>
                });

            for (let i = 0; i < batch.length; i++) {
                batch[i].resolve(response.urls[i])
            }
        } catch (error) {
            for (const job of batch) {
                this.jobSet.delete(job.id)
                job.reject(error)
            }
        }

        setTimeout(this.startTranslateTimeout.bind(this), 0)
        this.isTranslating = false
    }

    startTranslateTimeout() {
        if (this.isTranslating) {
            return
        }

        if (this.pendingTranslateTimeout !== undefined) {
            clearTimeout(this.pendingTranslateTimeout)
            this.pendingTranslateTimeout = undefined
        }

        if (this.pendingJobs.length > 0 && this.pendingJobs.length < this.batchSize) {
            // Try waiting for more
            this.pendingTranslateTimeout = setTimeout(this.translateBatch.bind(this), this.batchTimeout)
            console.log(`waiting full batch ${this.pendingJobs.length}/${this.batchSize}`)
        }
        else if (this.pendingJobs.length > 0) {
            // Translate batch
            this.translateBatch()
        }
    }

    enqueued(id: string): boolean {
        return this.jobSet.has(id)
    }

    async getImage(data: SerializedImage) {
        let result = await fetch(data.url, {
            headers: data.headers
        }
        ).then(c => c.blob()).catch((e) => {
            console.error(`Failed to fetch image with id:${data.id}`)
            console.error(e)
            throw new Error(`Failed to fetch image with id:${data.id}`)
        })

        if (!result.type.startsWith("image") && !data.url.startsWith("data")) {
            // If the regular fetch failed try using the proxy
            result = await fetch(`${this.serverAddress}/get-image`, {
                method: "POST",
                body: JSON.stringify({
                    url: data.url,
                    headers: data.headers
                }),
                headers: {
                    "Content-Type": "application/json"
                }
            }).then(async c => {
                const asBlob = await c.blob();
                if (!asBlob.type.startsWith("image")) {
                    throw new Error(`expected image from ${data.url} but got ${asBlob.type}`)
                }
                return asBlob
            }).catch((e) => {
                console.error(`Failed to fetch image with id:${data.id}`)
                console.error(e)
                throw new Error(`Failed to fetch image with id:${data.id}`)
            })
        }
        return result
    }

    // async enqueue(data: SerializedImage): Promise<string> {
    //     this.jobSet.add(data.id)
    //     const image = await this.getImage(data)

    //     if (image === undefined) {
    //         this.jobSet.delete(data.id)
    //         throw new Error("could not get image")
    //     }

    //     return new Promise((res, rej) => {
    //         const job: TranslationJob = {
    //             id: data.id,
    //             image: image,
    //             resolve: res,
    //             reject: rej
    //         }

    //         this.pendingJobs.push(job)
    //         this.startTranslateTimeout()
    //     })
    // }

    async enqueue(data: SerializedImage[]): Promise<Promise<string>[]> {
        for (const item of data) {
            this.jobSet.add(item.id)
        }

        const results = await Promise.allSettled(data.map(c => this.getImage(c)))

        // if (image === undefined) {
        //     this.jobSet.delete(data.id)
        //     throw new Error("could not get image")
        // }

        return results.map((c, idx) => {
            return new Promise((res, rej) => {
                if (c.status === "rejected") {
                    rej(c.reason)
                }
                else {
                    const job: TranslationJob = {
                        id: data[idx].id,
                        image: c.value,
                        resolve: res,
                        reject: rej
                    }

                    this.pendingJobs.push(job)
                    this.startTranslateTimeout()
                }
            })
        })
    }

    onStorageChanged(changes: browser.Storage.StorageAreaOnChangedChangesType) {
        const serverAddress = changes[StorageKeys.ServerAddress]?.newValue as string | undefined
        const batchSize = changes[StorageKeys.BatchSize]?.newValue as number | undefined
        const batchTimeout = changes[StorageKeys.BatchTimeout]?.newValue as number | undefined

        if (serverAddress !== undefined) {
            this.serverAddress = serverAddress
        }

        if (batchSize !== undefined) {
            this.batchSize = batchSize
        }

        if (batchTimeout !== undefined) {
            this.batchTimeout = batchTimeout
        }
    }

    async init() {
        const result = await getStorageOrDefaults()
        this.serverAddress = result[StorageKeys.ServerAddress]
        this.batchSize = result[StorageKeys.BatchSize]
        this.batchTimeout = result[StorageKeys.BatchTimeout]
        browser.storage.local.onChanged.addListener(this.onStorageChanged.bind(this))
    }
}