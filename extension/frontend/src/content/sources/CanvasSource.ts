import type { SerializedImage } from "../../shared/types";
import { SourceTags } from "./SourceTags";
import type { ISource } from "./types";
import { v4 as uuidv4 } from 'uuid';

function loadAsync(src: string) {
    return new Promise<HTMLImageElement>((res, rej) => {
        const img = new Image()
        img.onload = () => {
            res(img)
        }
        img.onerror = rej
        img.src = src
    })
}
export class CanvasSource implements ISource {
    source: HTMLCanvasElement
    id: string
    constructor(source: HTMLCanvasElement) {
        this.source = source
        this.id = this.source.getAttribute(SourceTags.Id) ?? uuidv4()
        this.source.setAttribute(SourceTags.Id, this.id)
        this.source.setAttribute(SourceTags.OriginalSrc, source.toDataURL())
    }

    getId(): string {
        return this.id
    }
    getImageInfo(): SerializedImage {

        return {
            url: this.source.getAttribute(SourceTags.OriginalSrc) ?? "",
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
    async toggleTranslationVisible() {
        if (this.hasTranslation()) {
            if (this.isShowingTranslation()) {
                const img = await loadAsync(this.source.getAttribute(SourceTags.OriginalSrc) ?? "")
                this.source.getContext('2d')?.drawImage(img, 0, 0)
                this.source.toggleAttribute(SourceTags.TranslationVisible, false)
            }
            else {
                const img = await loadAsync(this.source.getAttribute(SourceTags.TranslatedSrc) ?? "")
                this.source.getContext('2d')?.drawImage(img, 0, 0)
                this.source.toggleAttribute(SourceTags.TranslationVisible, true)
            }
        }
    }

    async onTranslationCompleted(url: string) {
        this.source.setAttribute(SourceTags.TranslatedSrc, url)
        this.toggleTranslationVisible().then(() => {
            this.source.toggleAttribute(SourceTags.PendingTranslation, false)
        }).catch((e) => {
            console.error(e)
            this.source.toggleAttribute(SourceTags.PendingTranslation, false)
        })
        
    }
    onTranslationFailed() {
        this.source.toggleAttribute(SourceTags.PendingTranslation, false)
    }
    onTranslationStarted() {
        this.source.toggleAttribute(SourceTags.PendingTranslation, true)
    }
}

