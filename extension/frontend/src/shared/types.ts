export const enum MessageType {
    FetchOrRestoreImage  = "image-fetch-or-restore",
    RestoreImage = "image-restore",
    TranslationStarted  = "translation-started",
    TranslationComplete  = "translation-completed",
    TranslationFailed  = "translation-failed",
    /**
     * Fetch the set A which is the result of filtering the set B for images not translated where B is the set of the image under the mouse and all other images in the list on the page i.e. a manga strip
     */
    FetchGroupNotTranslated = "image-fetch-group-not-translated",
    InspectMessage = "inspect"
}

export type FetchOrRestoreImageMessage = {
    type: MessageType.FetchOrRestoreImage
    args: []
}

export type StartedTranslationMessage = {
    type: MessageType.TranslationStarted
    args: [imageId: string]
}

export type CompletedTranslationMessage = {
    type: MessageType.TranslationComplete
    args: [imageId: string,new_url: string]
}

export type FailedTranslationMessage = {
    type: MessageType.TranslationFailed
    args: [imageId: string,reason: string]
}

export type FetchGroupNotTranslatedMessage = {
    type: MessageType.FetchGroupNotTranslated
    args: [images: SerializedImage]
}

export type InspectMessage = {
    type: MessageType.InspectMessage
    args: []
}


export type Message = FetchOrRestoreImageMessage | StartedTranslationMessage | CompletedTranslationMessage | FailedTranslationMessage | FetchGroupNotTranslatedMessage | InspectMessage

export type SerializedImage = {
    url: string
    id: string
    headers: Record<string,string>
}

export type MessageResponse<T = undefined> = {
    response: T
    error?: unknown
}