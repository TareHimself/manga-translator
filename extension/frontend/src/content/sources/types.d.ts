import type { SerializedImage } from "../../shared/types"

// Anything we can translate (I found out that <picture/> tag exists)
export abstract class ISource {
    getId(): string
    hasTranslation(): boolean
    isShowingTranslation(): boolean
    toggleTranslationVisible()
    getImageInfo(): SerializedImage
    onTranslationCompleted(url: string)
    onTranslationFailed()
    onTranslationStarted()
}

export interface IPageSourceInfo {
    element: HTMLElement
    source: ISource
}
export interface ISourceFactoryResult {
    context: IPageSourceInfo
    related: IPageSourceInfo[]
}

export type SourceFactory = (element: HTMLElement,existing: Map<HTMLElement,ISource>) => ISourceFactoryResult | undefined