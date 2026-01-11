import { getSingleFactories } from "./factories"
import { SourceTags } from "./SourceTags"
import type { ISource } from "./types"


const ELEMENTS_TO_SOURCES = new Map<HTMLElement, ISource>()
const ELEMENT_IDS_TO_SOURCES = new Map<string, ISource>()
let targetElement: HTMLElement | undefined = undefined
document.addEventListener("contextmenu", (e) => {
    const target = e.target
    if (target === null) return
    if (!(target instanceof HTMLElement)) return
    targetElement = target

})

export const getContextMenuSourceTarget = () => {
    if (targetElement === undefined) return undefined

    const factories = getSingleFactories()

    for (const factory of factories) {
        const result = factory(targetElement, ELEMENTS_TO_SOURCES);
        if (result !== undefined) {
            ELEMENTS_TO_SOURCES.set(result.context.element, result.context.source)
            ELEMENT_IDS_TO_SOURCES.set(result.context.source.getId(), result.context.source)
            for (const item of result.related) {
                ELEMENTS_TO_SOURCES.set(item.element, item.source)
                ELEMENT_IDS_TO_SOURCES.set(item.source.getId(), item.source)
            }
            return [result.context.source,...result.related.map(c => c.source)]
        }
    }

    return undefined
}

export const getSourceById = (id: string) => {
    const target = ELEMENT_IDS_TO_SOURCES.get(id)
    if (target === undefined) {
        console.error(`Failed to find source with ${SourceTags.Id}="${id}"`)
        return undefined
    }
    return target
}

export const getTargetElement = () => targetElement
export { SourceTags }