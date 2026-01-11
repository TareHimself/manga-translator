import browser from "webextension-polyfill"
import { type CompletedTranslationMessage, type FailedTranslationMessage, type FetchOrRestoreImageMessage, type InspectMessage, type MessageResponse, type SerializedImage, type StartedTranslationMessage, MessageType } from "../shared/types"
import { Translator } from "./Translator"


const CONTEXT_MENU_TRANSLATE_OR_RESTORE_ID = "manga-translator/context/translate_or_restore"
//const CONTEXT_MENU_TRANSLATE_ALL = "manga-translator/context/translate_all"
const CONTEXT_MENU_INSPECT = "manga-translator/context/inspect"
const PENDING_TRANSLATIONS = new Set<string>()
const TRANSLATOR = new Translator()

const sendTranslationStartedMessage = async (tabId: number, imageId: string) => {
    const message: StartedTranslationMessage = {
        type: MessageType.TranslationStarted,
        args: [imageId]
    }
    await browser.tabs.sendMessage(tabId, message)
}

const sendTranslationCompletedMessage = async (tabId: number, imageId: string, newUrl: string) => {
    const message: CompletedTranslationMessage = {
        type: MessageType.TranslationComplete,
        args: [imageId, newUrl]
    }
    await browser.tabs.sendMessage(tabId, message)
}

const sendTranslationFailedMessage = async (tabId: number, imageId: string, reason: string) => {
    const message: FailedTranslationMessage = {
        type: MessageType.TranslationFailed,
        args: [imageId, reason]
    }
    await browser.tabs.sendMessage(tabId, message)
}


browser.contextMenus.onClicked.addListener(async (info, tab) => {
    switch (info.menuItemId) {
        case CONTEXT_MENU_TRANSLATE_OR_RESTORE_ID:
            {
                const tabId = tab?.id ?? 0
                const message: FetchOrRestoreImageMessage = {
                    type: MessageType.FetchOrRestoreImage,
                    args: []
                }
                const result: MessageResponse<SerializedImage[] | undefined> = await browser.tabs.sendMessage(tabId, message)
                if (result.error) {
                    console.error(result.error)
                    return
                }
                if (result.response === undefined) return

                await Promise.allSettled(result.response.map(c => sendTranslationStartedMessage(tabId, c.id)))

                const promises = await TRANSLATOR.enqueue(result.response)
                promises.forEach((p, idx) => {
                    p.then(async (newImageUrl) => {
                        await sendTranslationCompletedMessage(tabId, result.response![idx].id, newImageUrl)
                    }).catch((e) => {
                        const image = result.response![idx]
                        console.error("Failed to translate")
                        console.error(image)
                        console.error(e)
                        PENDING_TRANSLATIONS.delete(image.id)
                        sendTranslationFailedMessage(tabId, image.id, e.toString())
                    })
                })
            }
            break;
        case CONTEXT_MENU_INSPECT:
            {
                const message: InspectMessage = {
                    type: MessageType.InspectMessage,
                    args: []
                }
                const result: MessageResponse<string> = await browser.tabs.sendMessage(tab?.id ?? 0, message)
                console.log("inspect result:")
                console.log(result)
            }
            break;
    }
})
browser.runtime.onInstalled.addListener(() => {
    browser.contextMenus.create({
        title: "Translate / Restore",
        contexts: ["all"],
        id: CONTEXT_MENU_TRANSLATE_OR_RESTORE_ID
    })
    // browser.contextMenus.create({
    //     title: "Try Translate All",
    //     contexts: ["all"],
    //     id: CONTEXT_MENU_TRANSLATE_ALL
    // })
    // browser.contextMenus.create({
    //     title: "Inspect",
    //     contexts: ["all"],
    //     id: CONTEXT_MENU_INSPECT
    // })
    TRANSLATOR.init()
})