import browser from "webextension-polyfill"
import { type Message, type FetchOrRestoreImageMessage as FetchOrRestoreImageMessage, type FetchImageResponse, type CompletedTranslationMessage, type StartedTranslationMessage, type FailedTranslationMessage, MessageType } from "../shared/types"
import { getContextMenuSourceTarget, getSourceById, SourceTags } from "./sources";



const fetchOrRestoreImage: (...args: FetchOrRestoreImageMessage["args"]) => FetchImageResponse | undefined = () => {
    const source = getContextMenuSourceTarget()

    if (source === undefined) return

    if(source.hasTranslation()){
        source.toggleTranslationVisible()
        return undefined
    }

    return source.getImageInfo()
}

const onTranslationStarted: (...args: StartedTranslationMessage["args"]) => void = (imageId) => {
    const source = getSourceById(imageId)

    if (source === undefined) return

    source.onTranslationStarted()
}

const onTranslationCompleted: (...args: CompletedTranslationMessage["args"]) => void = (sourceId, newUrl) => {
    const source = getSourceById(sourceId)

    if (source === undefined) return

    source.onTranslationCompleted(newUrl)
}

const onTranslationFailed: (...args: FailedTranslationMessage["args"]) => void = (imageId, reason) => {
    const source = getSourceById(imageId)

    if (source === undefined) return

    source.onTranslationFailed()

    console.error(`Failed to translate image with ${SourceTags.Id}="${imageId}": ${reason}`)
}

browser.runtime.onMessage.addListener((data, _, respond) => {
    const message = data as Message
    switch (message.type) {
        case MessageType.FetchOrRestoreImage:
            {
                try {
                    respond(fetchOrRestoreImage(...message.args))
                }
                catch{
                    respond(undefined)
                }
            }
            break;
        case MessageType.TranslationStarted:
            {
                respond(undefined)
                onTranslationStarted(...message.args)
            }
            break;
        case MessageType.TranslationComplete:
            {
                respond(undefined)
                onTranslationCompleted(...message.args)
            }
            break;
        case MessageType.TranslationFailed:
            {
                respond(undefined)
                onTranslationFailed(...message.args)
            }
    }

    return true;
})