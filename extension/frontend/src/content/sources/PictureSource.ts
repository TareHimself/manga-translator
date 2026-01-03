import type { FetchImageResponse } from "../../shared/types";
import { SourceTags } from "./SourceTags";
import type { ISource } from "./types";
import { v4 as uuidv4 } from 'uuid';

export class PictureSource implements ISource {
    source: HTMLPictureElement
    id: string
    sourceElements: Node[]
    sourceImage: HTMLImageElement
    constructor(source: HTMLPictureElement) {
        this.source = source
        this.id = this.source.getAttribute(SourceTags.Id) ?? uuidv4()
        this.source.setAttribute(SourceTags.Id,this.id)

        const imageElement = this.source.querySelector("img")
        if(imageElement === null){
            throw new Error(`Picture element with id ${this.id} does not have an image, what even is this website?`)
        }
        this.sourceImage = imageElement
        this.source.setAttribute(SourceTags.OriginalSrc,this.sourceImage.currentSrc ?? this.sourceImage.src)

        this.sourceElements = Array.from(this.source.children).filter(c => !(c instanceof HTMLImageElement)).map(c => {
            c.remove()
            return c
        })
    }
    getId(): string {
        return this.id
    }
    getImageInfo(): FetchImageResponse {
        return {
            url: this.sourceImage.src,
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
                this.sourceImage.src = this.source.getAttribute(SourceTags.OriginalSrc) ?? ""
                this.source.toggleAttribute(SourceTags.TranslationVisible, false)
            }
            else {
                this.sourceImage.src = this.source.getAttribute(SourceTags.TranslatedSrc) ?? ""
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