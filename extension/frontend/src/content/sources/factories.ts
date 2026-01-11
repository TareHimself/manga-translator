import { CanvasSource } from "./CanvasSource"
import { ImageSource } from "./ImageSource"
import { PictureSource } from "./PictureSource"
import type { SourceFactory } from "./types"
import { WelomaImageSource } from "./WelomaImageSource"
import { XImageSource } from "./XImageSource"

// Theres a lot of weird shit in this file, we basically want to get the item that was right clicked and any related items i.e. all other pages on a manga strip

// Given [0,1,2,3,4,5] and 2 we get [1,3,0,4,5] where items are sorted based on their distance to the first item and the first item is removed
function sortAndInterleave<T>(items: T[], target: T, removeTarget = true) {
    const targetItemIndex = items.indexOf(target)
    const sorted = items.map((c, index) => {
        return {
            element: c,
            index
        }
    }).sort((a, b) => {
        const aDist = Math.abs(targetItemIndex - a.index)
        const bDist = Math.abs(targetItemIndex - b.index)
        return aDist - bDist
    })
    if (removeTarget) {
        sorted.shift()
    }
    return sorted.map(c => c.element)
}

const pictureListFactory: SourceFactory = (element, existing) => {
    if (element instanceof HTMLImageElement && element.parentElement instanceof HTMLPictureElement) {
        return {
            context: {
                element: element,
                source: existing.get(element) ?? new PictureSource(element.parentElement)
            },
            related: []
        }
    }
    return undefined
}

const xImageFactory: SourceFactory = (element, existing) => {
    if (element instanceof HTMLImageElement && window.location.origin === "https://x.com") {
        const div = element.parentElement?.children[0]
        if (div instanceof HTMLDivElement) {
            return {
                context: {
                    element: element,
                    source: existing.get(element) ?? new XImageSource(div)
                },
                related: []
            }
        }
    }
    return undefined
}

const imageFactory: SourceFactory = (element, existing) => {
    if (element instanceof HTMLImageElement) {
        return {
            context: {
                element,
                source: existing.get(element) ?? new ImageSource(element)
            },
            related: sortAndInterleave(Array.from((element.parentElement?.children ?? [])).filter(c => c instanceof HTMLImageElement) as HTMLImageElement[], element).map((e) => {
                return {
                    element: e,
                    source: existing.get(e) ?? new ImageSource(e)
                }
            })
        }
    }
    return undefined
}

const welomaImageFactory: SourceFactory = (element, existing) => {
    if (element instanceof HTMLImageElement && element.hasAttribute("srcset") && window.location.origin.includes("https://weloma.")) {
        return {
            context: {
                element,
                source: existing.get(element) ?? new WelomaImageSource(element)
            },
            related: sortAndInterleave(Array.from((element.parentElement?.children ?? [])).filter(c => c instanceof HTMLImageElement && c.hasAttribute("srcset")) as HTMLImageElement[], element).map((e) => {
                return {
                    element: e,
                    source: existing.get(e) ?? new WelomaImageSource(e)
                }
            })
        }
    }
    return undefined
}

const mangaFireImageFactory: SourceFactory = (element, existing) => {
    if (element instanceof HTMLDivElement && window.location.origin.includes("https://mangafire.") && element.children[0]?.children[0] instanceof HTMLImageElement) {
        const image = element.children[0]?.children[0] as HTMLImageElement
        if (!image.hasAttribute("data-number")) return undefined
        return {
            context: {
                element: image,
                source: existing.get(element) ?? new ImageSource(image)
            },
            related: sortAndInterleave(Array.from((element.parentElement?.children ?? [])).filter(c => c.children[0]?.children[0] instanceof HTMLImageElement && (c.children[0]?.children[0].hasAttribute("data-number") ?? false)) as HTMLImageElement[], element).map((e) => {
                const eImage = e.children[0]?.children[0] as HTMLImageElement
                return {
                    element: e,
                    source: existing.get(e) ?? new ImageSource(eImage)
                }
            })
        }
    }
    return undefined
}

const canvasFactory: SourceFactory = (element, existing) => {
    if (element instanceof HTMLCanvasElement) {
        return {
            context: {
                element,
                source: existing.get(element) ?? new CanvasSource(element)
            },
            related: sortAndInterleave(Array.from((element.parentElement?.children ?? [])).filter(c => c instanceof HTMLCanvasElement) as HTMLCanvasElement[], element).map((e) => {
                return {
                    element: e,
                    source: existing.get(e) ?? new CanvasSource(e)
                }
            })
        }
    }
    return undefined
}

// for https://comic-walker.com/
const comicWalkerFactory: SourceFactory = (element, existing) => {
    if (window.location.origin.startsWith("https://comic-walker.") && element instanceof HTMLDivElement && element.parentElement?.children[0] instanceof HTMLCanvasElement) {
        return {
            context: {
                element,
                source: existing.get(element) ?? new CanvasSource(element.parentElement?.children[0])
            },
            related: sortAndInterleave(Array.from((element.parentElement?.parentElement?.parentElement?.children ?? [])).filter(c => c.children[0].children[0] instanceof HTMLCanvasElement) as HTMLDivElement[], element.parentElement.parentElement as HTMLDivElement).map((e) => {
                const contextElement = e.children[0].children[1] as HTMLDivElement
                return {
                    element: contextElement,
                    source: existing.get(contextElement) ?? new CanvasSource(e.children[0].children[0] as HTMLCanvasElement)
                }
            })
            // This will collect everything on the page but does not actually work that well since some canvas elements might not have any info in them
            // related: sortAndInterleave(Array.from((element.parentElement?.parentElement?.parentElement?.parentElement?.parentElement?.children ?? [])).reduce((all,c) => {
            //     all.push(c.children[0].children[0] as HTMLDivElement)
            //     all.push(c.children[0].children[1] as HTMLDivElement)
            //     return all;
            // },[] as HTMLDivElement[]).filter(c => c.children[0].children[0] instanceof HTMLCanvasElement) as HTMLDivElement[], element.parentElement.parentElement as HTMLDivElement).map((e) => {
            //     const contextElement = e.children[0].children[1] as HTMLDivElement
            //     return {
            //         element: contextElement,
            //         source: existing.get(contextElement) ?? new CanvasSource(e.children[0].children[0] as HTMLCanvasElement)
            //     }
            // })
        }
    }
    return undefined
}

/**
 * Factories that return single sources
 * @returns 
 */
export const getSingleFactories = () => {
    return [
        pictureListFactory,
        xImageFactory,
        welomaImageFactory,
        imageFactory,
        mangaFireImageFactory,
        canvasFactory,
        comicWalkerFactory
    ]
}

