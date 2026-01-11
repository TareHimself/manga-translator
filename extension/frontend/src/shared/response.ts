import type { MessageResponse } from "./types";

export function makeResponse<T>(data: T): MessageResponse<T> {
    return {
        response: data,
        error: undefined
    }
}

export function makeError<T = unknown>(error: T): MessageResponse<undefined>{
    return {
        response: undefined,
        error
    }
}