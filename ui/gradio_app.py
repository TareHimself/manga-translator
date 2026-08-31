"""Gradio front-end for the comic-localizer pipeline.

Replaces the blacksheep server + React app. Every plugin's options are rendered
from its ``get_arguments()``, so new plugins need no UI work. Config load / save
round-trips the same ``pipeline:`` YAML that
``construct_image_to_image_pipeline_from_config`` reads.

    uv run python gradio_app.py        (from ui/)
"""

import os
import tempfile

import gradio as gr
import numpy as np
import yaml
from dotenv import load_dotenv

from comic_localizer.core.plugin import PluginArgumentType as AT
from comic_localizer.cleaning.get import get_cleaners
from comic_localizer.detection.get import get_detectors
from comic_localizer.drawing.get import get_drawers
from comic_localizer.get import construct_plugin_by_name
from comic_localizer.ocr.get import get_ocrs
from comic_localizer.pipelines.image_to_image import ImageToImagePipeline
from comic_localizer.segmentation.get import get_segmenters
from comic_localizer.translation.get import get_translators

load_dotenv()

CLEAN_STAGES = ["detector", "segmenter", "cleaner"]
TRANSLATE_STAGES = CLEAN_STAGES + ["ocr", "translator", "drawer"]
GETTERS = {
    "detector": get_detectors,
    "segmenter": get_segmenters,
    "cleaner": get_cleaners,
    "ocr": get_ocrs,
    "translator": get_translators,
    "drawer": get_drawers,
}
# {stage: {class_name: cls}}
PLUGINS = {s: {c.__name__: c for c in g()} for s, g in GETTERS.items()}


def _defaults(cls) -> dict:
    return {a.get()["id"]: a.get()["default"] for a in cls.get_arguments()}


def _default_config() -> dict:
    out = {}
    for stage in GETTERS:
        name = next(iter(PLUGINS[stage]))
        out[stage] = {"class": name, "args": _defaults(PLUGINS[stage][name])}
    return out


def _choices(stage: str):
    return [(cls.get_name(), name) for name, cls in PLUGINS[stage].items()]


def _arg_component(meta: dict, value):
    kind, name, info = meta["type"], meta["name"], meta["description"] or None
    if kind == AT.SELECT:
        opts = [(o["name"], o["value"]) for o in meta["options"]]
        return gr.Dropdown(opts, value=value, label=name, info=info, interactive=True)
    if kind == AT.INT:
        return gr.Number(value=value, precision=0, label=name, info=info, interactive=True)
    if kind == AT.BOOLEAN:
        return gr.Checkbox(value=bool(value), label=name, info=info, interactive=True)
    return gr.Textbox(value=value, label=name, info=info, interactive=True)


def _assemble(selected: dict, flat: list[tuple[str, str]], values: tuple) -> dict:
    config = {}
    for (stage, arg_id), value in zip(flat, values):
        config.setdefault(stage, {"class": selected[stage], "args": {}})
        config[stage]["args"][arg_id] = value
    for stage in selected:  # stages with no args
        config.setdefault(stage, {"class": selected[stage], "args": {}})
    return config


async def _run(operation, image, config: dict):
    if image is None:
        raise gr.Error("Upload an image first.")
    stages = CLEAN_STAGES if operation == "Clean" else TRANSLATE_STAGES
    kwargs = {
        s: construct_plugin_by_name(config[s]["class"], config[s].get("args") or {})
        for s in stages
    }
    out = await ImageToImagePipeline(**kwargs)([np.ascontiguousarray(image)])
    return (image, out[0])


def _to_yaml(config: dict, operation) -> str:
    stages = CLEAN_STAGES if operation == "Clean" else TRANSLATE_STAGES
    pipeline = {
        s: {"class": config[s]["class"], "args": config[s].get("args") or None}
        for s in stages
    }
    path = os.path.join(tempfile.mkdtemp(), "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"pipeline": pipeline}, f, sort_keys=False)
    return path


def _from_yaml(file, config: dict):
    with open(file, encoding="utf-8") as fh:
        loaded = (yaml.safe_load(fh) or {}).get("pipeline", {})
    config = {s: dict(v) for s, v in config.items()}
    for stage, data in loaded.items():
        cls_name = (data or {}).get("class")
        if stage not in PLUGINS or cls_name not in PLUGINS[stage]:
            continue
        cls = PLUGINS[stage][cls_name]
        config[stage] = {
            "class": cls_name,
            "args": {**_defaults(cls), **(data.get("args") or {})},
        }
    return [config, *[gr.update(value=config[s]["class"]) for s in GETTERS]]


def build() -> gr.Blocks:
    with gr.Blocks(title="comic-localizer") as demo:
        # config holds only what "Load config" restores; live edits are read
        # straight off the rendered components at run / save time.
        config = gr.State(_default_config())

        gr.Markdown("## comic-localizer")
        with gr.Row():
            left = gr.Column(scale=1)
            right = gr.Column(scale=3)

        with right:
            image_in = gr.Image(type="numpy", label="Page", height=460)
            result = gr.ImageSlider(label="Before / after", height=620)

        with left:
            operation = gr.Radio(
                ["Clean", "Translate"], value="Clean", label="Operation"
            )
            dropdowns = {
                s: gr.Dropdown(
                    _choices(s), value=next(iter(PLUGINS[s])), label=s.capitalize()
                )
                for s in GETTERS
            }

            # Everything below is built inside the render so its event listeners
            # stay valid across re-renders.
            @gr.render(inputs=[*dropdowns.values(), operation, config])
            def _render(*vals):
                *picks, operation_val, cfg = vals
                selected = dict(zip(GETTERS, picks))
                stages = (
                    CLEAN_STAGES if operation_val == "Clean" else TRANSLATE_STAGES
                )

                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    save_btn = gr.DownloadButton("Save config")
                    load_btn = gr.UploadButton(
                        "Load config", file_types=[".yaml", ".yml"]
                    )

                flat: list[tuple[str, str]] = []
                comps: list = []
                for stage in stages:
                    name = selected[stage]
                    cls = PLUGINS[stage][name]
                    seed = (
                        cfg[stage]["args"]
                        if cfg[stage]["class"] == name
                        else _defaults(cls)
                    )
                    args = cls.get_arguments()
                    if not args:
                        continue
                    with gr.Accordion(
                        f"{stage.capitalize()} · {cls.get_name()}", open=True
                    ):
                        for arg in args:
                            m = arg.get()
                            comps.append(
                                _arg_component(m, seed.get(m["id"], m["default"]))
                            )
                            flat.append((stage, m["id"]))

                async def _do_run(image, op, *values, sel=selected, fl=flat):
                    return await _run(op, image, _assemble(sel, fl, values))

                def _do_save(op, *values, sel=selected, fl=flat):
                    return _to_yaml(_assemble(sel, fl, values), op)

                run_btn.click(_do_run, [image_in, operation, *comps], result)
                save_btn.click(_do_save, [operation, *comps], save_btn)
                load_btn.upload(
                    _from_yaml, [load_btn, config], [config, *dropdowns.values()]
                )

    return demo


if __name__ == "__main__":
    build().launch(inbrowser=True, server_port=5000, theme=gr.themes.Soft())
