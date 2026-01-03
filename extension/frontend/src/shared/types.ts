export const enum MessageType {
    FetchOrRestoreImage  = "image-fetch-or-restore",
    RestoreImage = "image-restore",
    TranslationStarted  = "translation-started",
    TranslationComplete  = "translation-completed",
    TranslationFailed  = "translation-failed"
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

export type Message = FetchOrRestoreImageMessage | StartedTranslationMessage | CompletedTranslationMessage | FailedTranslationMessage

export type FetchImageResponse = {
    url: string
    id: string
    headers: Record<string,string>
}