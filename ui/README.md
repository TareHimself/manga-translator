# UI

A [Gradio](https://www.gradio.app/) front-end for the comic-localizer pipeline.
Every plugin's options are rendered from its `get_arguments()`, so new plugins
need no UI work.

## Install

```bash
uv sync
```

## Usage

```bash
uv run python gradio_app.py     # opens a browser tab on http://localhost:5000
```

or `poe start`.

Pick a plugin per stage (detector / segmenter / cleaner, plus OCR / translator /
drawer in Translate mode), fill in its options, upload a page, and hit **Run**.
The result comes back as a before/after slider.

**Save config** downloads the current selection as a `pipeline:` YAML; **Load
config** restores it. That file is the same format `comic_localizer` reads via
`construct_image_to_image_pipeline_from_config`, so it works with the CLI too.
