import type { SerializedImage} from "../../shared/types";
import { SourceTags } from "./SourceTags";
import type { ISource } from "./types";
import { v4 as uuidv4 } from 'uuid';

const X_IMAGE_PATTERN = new RegExp(/url\(["'](.*)["']\)/)

/**
 * For images on https://x.com
 */
export class XImageSource implements ISource {
    source: HTMLDivElement
    image?: HTMLImageElement
    id: string
    constructor(source: HTMLDivElement) {
        this.source = source
        this.id = this.source.getAttribute(SourceTags.Id) ?? uuidv4()
        this.source.setAttribute(SourceTags.Id,this.id)
        this.source.setAttribute(SourceTags.OriginalSrc,this.backgroundImageCssToResource(this.source.style.backgroundImage))
        this.image = this.source.parentElement?.children[1]  as HTMLImageElement | undefined
    }

    backgroundImageCssToResource(backgroundImageCss: string){
        return X_IMAGE_PATTERN.exec(backgroundImageCss)?.[1] ?? ""
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
    toggleTranslationVisible() {
        if (this.hasTranslation()) {
            if (this.isShowingTranslation()) {
                const src = this.source.getAttribute(SourceTags.OriginalSrc) ?? ''
                this.source.style.backgroundImage = `url("${src}")`
                this.image?.setAttribute("src",src)
                this.source.toggleAttribute(SourceTags.TranslationVisible, false)
            }
            else {
                const src = this.source.getAttribute(SourceTags.TranslatedSrc) ?? ''
                this.source.style.backgroundImage = `url("${src}")`
                this.image?.setAttribute("src",src)
                this.source.toggleAttribute(SourceTags.TranslationVisible, true)
            }
        }
    }
    async convertTranslation(url: string){
        const res = await fetch(url);
        const blob = await res.blob();
        const blobUrl = URL.createObjectURL(blob)
        this.source.setAttribute(SourceTags.TranslatedSrc,blobUrl)
        this.toggleTranslationVisible()
        this.source.toggleAttribute(SourceTags.PendingTranslation,false)
    }
    onTranslationCompleted(url: string) {
        this.convertTranslation(url).catch((e) => {
            this.onTranslationFailed()
            console.error(e)
        })
    }
    onTranslationFailed() {
        this.source.toggleAttribute(SourceTags.PendingTranslation,false)
    }
    onTranslationStarted() {
        this.source.toggleAttribute(SourceTags.PendingTranslation,true)
    }
}

