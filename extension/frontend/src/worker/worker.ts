import browser from "webextension-polyfill"
import { type CompletedTranslationMessage, type FailedTranslationMessage, type FetchOrRestoreImageMessage, type FetchImageResponse, type StartedTranslationMessage, MessageType } from "../shared/types"


const SERVER_URL = "http://127.0.0.1:9000/api/v1"
const CONTEXT_MENU_TRANSLATE_OR_RESTORE_ID = "manga-translator/context/translate_or_restore"
const PENDING_TRANSLATIONS = new Set<string>()

const sendTranslationStartedMessage = async (tabId: number,imageId: string) => {
    const message: StartedTranslationMessage = {
                    type: MessageType.TranslationStarted,
                    args: [imageId]
                }
                await browser.tabs.sendMessage(tabId, message)
}

const sendTranslationCompletedMessage = async (tabId: number,imageId: string,newUrl: string) => {
    const message: CompletedTranslationMessage = {
                    type: MessageType.TranslationComplete,
                    args: [imageId,newUrl]
                }
                await browser.tabs.sendMessage(tabId, message)
}

const sendTranslationFailedMessage = async (tabId: number,imageId: string,reason: string) => {
    const message: FailedTranslationMessage = {
                    type: MessageType.TranslationFailed,
                    args: [imageId,reason]
                }
                await browser.tabs.sendMessage(tabId, message)
}
const getImage = async (data: FetchImageResponse) => {
    let result = await fetch(data.url, {
        headers: data.headers
    }
    ).then(c => c.blob()).catch((e) => {
        console.error(`Failed to fetch image with id:${data.id}`)
        console.error(e)
        return undefined
    })

    if (result === undefined) return undefined

    if (!result.type.startsWith("image") && !data.url.startsWith("data")) {
        // If the regular fetch failed try using the proxy
        result = await fetch(`${SERVER_URL}/get-image`, {
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
            return undefined
        })
    }
    return result
}
const handleTranslation = async (image: Blob): Promise<string> => {
    const serverUrl = SERVER_URL
    const formData = new FormData()
    formData.append('file', image)
    return await fetch(`${serverUrl}/translate`,
        {
            method: "POST",
            body: formData,
        }
    )
        .then(async (a) => {
            if (a.status === 500) {
                throw new Error(await a.text())
            }

            return a.text();
        });
}

browser.contextMenus.onClicked.addListener(async (info, tab) => {
    switch (info.menuItemId) {
        case CONTEXT_MENU_TRANSLATE_OR_RESTORE_ID:
            {
                const message: FetchOrRestoreImageMessage = {
                    type: MessageType.FetchOrRestoreImage,
                    args: []
                }
                const result: FetchImageResponse | undefined = await browser.tabs.sendMessage(tab?.id ?? 0, message)
                if (result === undefined) return // if result is undefined we did a resore or something else went wrong
                if (PENDING_TRANSLATIONS.has(result.id)) return
                PENDING_TRANSLATIONS.add(result.id)
                await sendTranslationStartedMessage(tab?.id ?? 0,result.id)

                const data = await getImage(result)
                if (data === undefined) {
                    PENDING_TRANSLATIONS.delete(result.id)
                    await sendTranslationFailedMessage(tab?.id ?? 0,result.id,"could not get image")
                    return
                }
                handleTranslation(data).then(async (newImageUrl) => {
                    await sendTranslationCompletedMessage(tab?.id ?? 0,result.id,newImageUrl)
                    PENDING_TRANSLATIONS.delete(result.id)
                }).catch((e) => {
                    console.error("Failed to translate", result)
                    console.error(e)
                    PENDING_TRANSLATIONS.delete(result.id)
                    sendTranslationFailedMessage(tab?.id ?? 0,result.id,e.toString())
                })
            }
            break;
    }
})
browser.runtime.onInstalled.addListener(() => {
    browser.contextMenus.create({
        title: "Translate / Restore",
        contexts: ["image"],
        id: CONTEXT_MENU_TRANSLATE_OR_RESTORE_ID
    })
})