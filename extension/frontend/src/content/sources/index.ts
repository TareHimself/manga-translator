import { ImageSource } from "./ImageSource"
import { PictureSource } from "./PictureSource"
import { SourceTags } from "./SourceTags"
import type { ISource } from "./types"

const ELEMENTS_TO_SOURCES = new Map<HTMLElement,ISource>()
const ELEMENT_IDS_TO_SOURCES = new Map<string,ISource>()

let targetSource: ISource | undefined = undefined
document.addEventListener("contextmenu", (e) => {
    let contextTarget = e.target as HTMLElement | null ?? undefined
    if(contextTarget instanceof HTMLImageElement){
        if(contextTarget.parentElement instanceof HTMLPictureElement){
            contextTarget = contextTarget.parentElement
        }
    }
    else
    {
        contextTarget = undefined
    }

    if(contextTarget === undefined) {
        targetSource = undefined
        return
    }

    if(!ELEMENTS_TO_SOURCES.has(contextTarget)){
        let newSource: ISource
        if(contextTarget instanceof HTMLImageElement){
            newSource = new ImageSource(contextTarget)
        }
        else if(contextTarget instanceof HTMLPictureElement){
            newSource = new PictureSource(contextTarget)
        }
        else{
            throw new Error("Unknown source type, how did we even get here ?")
        }
        ELEMENTS_TO_SOURCES.set(contextTarget,newSource)
        ELEMENT_IDS_TO_SOURCES.set(newSource.getId(),newSource)
        targetSource = newSource
    }
    else{
        targetSource = ELEMENTS_TO_SOURCES.get(contextTarget)
    }
})

export const getContextMenuSourceTarget = () => {
    return targetSource
}

export const getSourceById = (id: string) => {
    const target = ELEMENT_IDS_TO_SOURCES.get(id)
    if (target === undefined) {
        console.error(`Failed to find source with ${SourceTags.Id}="${id}"`)
        return undefined
    }
    return target
}

export { SourceTags }