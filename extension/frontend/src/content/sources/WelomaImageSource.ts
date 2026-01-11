import { ImageSource } from "./ImageSource";


export class WelomaImageSource extends ImageSource {
    constructor(source: HTMLImageElement) {
        super(source)
    }

    getInitialSrc() {
        return this.source.getAttribute("srcset") ?? this.source.src
    }

    updateSrc(src: string) {
        this.source.setAttribute("srcset",src) 
    }
}