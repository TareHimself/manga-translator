import type { SerializedImage} from "../../shared/types";
import { SourceTags } from "./SourceTags";
import type { ISource } from "./types";
import { v4 as uuidv4 } from 'uuid';

const X_IMAGE_PATTERN = new RegExp(/url\(["'](.*)["']\)/)

/**
 * For divs that act as images like on https://x.com
 */
export class DivSource implements ISource {
    source: HTMLDivElement
    id: string
    constructor(source: HTMLDivElement) {
        this.source = source
        this.id = this.source.getAttribute(SourceTags.Id) ?? uuidv4()
        this.source.setAttribute(SourceTags.Id,this.id)
        this.source.setAttribute(SourceTags.OriginalSrc,this.backgroundImageCssToResource(this.source.style.backgroundImage))
        console.log("Div source",this)
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
                this.source.style.backgroundImage = `url("${this.source.getAttribute(SourceTags.OriginalSrc) ?? ''}")`
                this.source.toggleAttribute(SourceTags.TranslationVisible, false)
            }
            else {
                this.source.style.backgroundImage = `url("${this.source.getAttribute(SourceTags.TranslatedSrc) ?? ''}")`
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

