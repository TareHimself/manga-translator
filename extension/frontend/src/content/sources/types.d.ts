import type { FetchImageResponse } from "../../shared/types"

// Anything we can translate (I found out that <picture/> tag exists)
export interface ISource {
    getId(): string
    hasTranslation(): boolean
    isShowingTranslation(): boolean
    toggleTranslationVisible()
    getImageInfo(): FetchImageResponse
    onTranslationCompleted(url: string)
    onTranslationFailed()
    onTranslationStarted()
}

