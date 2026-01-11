import browser from "webextension-polyfill"
import { type Message, type FetchOrRestoreImageMessage as FetchOrRestoreImageMessage, type SerializedImage, type CompletedTranslationMessage, type StartedTranslationMessage, type FailedTranslationMessage, MessageType } from "../shared/types"
import { getContextMenuSourceTarget, getSourceById, getTargetElement, SourceTags } from "./sources";
import { makeError, makeResponse } from "../shared/response";


const fetchOrRestoreImage: (...args: FetchOrRestoreImageMessage["args"]) => SerializedImage[] | undefined = () => {
    const sources = getContextMenuSourceTarget()

    if (sources === undefined){
        throw  { 
            error: "element not supported for translation",
            element: getTargetElement()?.outerHTML
        }
    }

    if (sources[0].hasTranslation()) {
        sources[0].toggleTranslationVisible()
        return undefined
    }

    return sources.filter(c => !c.hasTranslation()).map(c => c.getImageInfo())
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
                    respond(makeResponse(fetchOrRestoreImage(...message.args)))
                }
                catch(e) {
                    respond(makeError(e))
                }
            }
            break;
        case MessageType.TranslationStarted:
            {
                respond(makeResponse(undefined))
                onTranslationStarted(...message.args)
            }
            break;
        case MessageType.TranslationComplete:
            {
                respond(makeResponse(undefined))
                onTranslationCompleted(...message.args)
            }
            break;
        case MessageType.TranslationFailed:
            {
                respond(makeResponse(undefined))
                onTranslationFailed(...message.args)
            }
            break;
        case MessageType.InspectMessage:
            {
                respond(makeResponse({
                    html: document.documentElement.innerHTML,
                    element: getTargetElement()?.outerHTML ?? ""
                }))
            }
            break;
    }

    return true;
})