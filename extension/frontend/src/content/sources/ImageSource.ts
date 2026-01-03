import type { FetchImageResponse } from "../../shared/types";
import { SourceTags } from "./SourceTags";
import type { ISource } from "./types";
import { v4 as uuidv4 } from 'uuid';

export class ImageSource implements ISource {
    source: HTMLImageElement
    id: string
    constructor(source: HTMLImageElement) {
        this.source = source
        this.id = this.source.getAttribute(SourceTags.Id) ?? uuidv4()
        this.source.setAttribute(SourceTags.Id,this.id)
        this.source.setAttribute(SourceTags.OriginalSrc,this.source.src)
    }
    getId(): string {
        return this.id
    }
    getImageInfo(): FetchImageResponse {
        
        return {
            url: this.source.src,
            id: this.id,
            headers: {
                referer: window.location.origin,
                "User-Agent": navigator.userAgent
            }
        }
    }
    hasTranslation(): boolean {
        return this.source.hasAttribute(SourceTags.TranslatedSrc)
    }
    isShowingTranslation(): boolean {
        return this.source.hasAttribute(SourceTags.TranslationVisible)
    }
    toggleTranslationVisible() {
        if (this.hasTranslation()) {
            if (this.isShowingTranslation()) {
                this.source.src = this.source.getAttribute(SourceTags.OriginalSrc) ?? ""
                this.source.toggleAttribute(SourceTags.TranslationVisible, false)
            }
            else {
                this.source.src = this.source.getAttribute(SourceTags.TranslatedSrc) ?? ""
                this.source.toggleAttribute(SourceTags.TranslationVisible, true)
            }
        }
    }
    onTranslationCompleted(url: string) {
        this.source.setAttribute(SourceTags.TranslatedSrc,url)
        this.toggleTranslationVisible()
        this.source.toggleAttribute(SourceTags.PendingTranslation,false)
    }
    onTranslationFailed() {
        this.source.toggleAttribute(SourceTags.PendingTranslation,false)
    }
    onTranslationStarted() {
        this.source.toggleAttribute(SourceTags.PendingTranslation,true)
    }
}