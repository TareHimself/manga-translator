from dataclasses import dataclass
from typing import Any

from rectpack import newPacker


@dataclass
class Placement:
    id: Any
    x: int
    y: int
    width: int
    height: int


@dataclass
class AtlasLayout:
    width: int
    height: int
    placements: list[Placement]


def _trim_to_content(placements: list[Placement]) -> tuple[int, int, list[Placement]]:
    min_x = min(p.x for p in placements)
    min_y = min(p.y for p in placements)
    max_x = max(p.x + p.width for p in placements)
    max_y = max(p.y + p.height for p in placements)

    trimmed = [
        Placement(id=p.id, x=p.x - min_x, y=p.y - min_y, width=p.width, height=p.height)
        for p in placements
    ]
    return max_x - min_x, max_y - min_y, trimmed


def _raw_byte_estimate(width: int, height: int, channels: int) -> int:
    return width * height * channels


def _split_by_byte_cap(
    width: int,
    height: int,
    placements: list[Placement],
    byte_cap: int,
    channels: int,
) -> list[AtlasLayout]:
    """Pop reading-order-last placements into their own solo atlases until the
    remainder's raw pixel estimate fits under byte_cap (or only one is left)."""
    remaining = sorted(placements, key=lambda p: (p.y, p.x))
    solo_atlases: list[AtlasLayout] = []

    while len(remaining) > 1 and _raw_byte_estimate(width, height, channels) > byte_cap:
        popped = remaining.pop()
        solo_atlases.append(
            AtlasLayout(
                width=popped.width,
                height=popped.height,
                placements=[Placement(popped.id, 0, 0, popped.width, popped.height)],
            )
        )
        width, height, remaining = _trim_to_content(remaining)

    main_width, main_height, main_placements = _trim_to_content(remaining)
    return [
        AtlasLayout(width=main_width, height=main_height, placements=main_placements)
    ] + solo_atlases


def pack_into_atlases(
    items: list[tuple[Any, int, int]],
    bin_size: int,
    padding: int,
    byte_cap: int,
    channels: int = 3,
) -> list[AtlasLayout]:
    """Pack (id, width, height) items into as few 2D atlases as possible.

    Returns layout only (positions/sizes) - the caller is responsible for
    rendering/encoding actual pixels from this.
    """
    if not items:
        return []

    fits: list[tuple[Any, int, int]] = []
    oversized: list[tuple[Any, int, int]] = []
    for item_id, width, height in items:
        if width + padding > bin_size or height + padding > bin_size:
            oversized.append((item_id, width, height))
        else:
            fits.append((item_id, width, height))

    atlases: list[AtlasLayout] = [
        AtlasLayout(width=w, height=h, placements=[Placement(item_id, 0, 0, w, h)])
        for item_id, w, h in oversized
    ]

    if fits:
        sizes = {item_id: (w, h) for item_id, w, h in fits}

        packer = newPacker(rotation=False)
        for item_id, w, h in fits:
            packer.add_rect(w + padding, h + padding, rid=item_id)
        packer.add_bin(bin_size, bin_size, count=float("inf"))
        packer.pack()

        for atlas_bin in packer:
            placements = [
                Placement(rect.rid, rect.x, rect.y, *sizes[rect.rid])
                for rect in atlas_bin
            ]
            width, height, trimmed = _trim_to_content(placements)
            if _raw_byte_estimate(width, height, channels) <= byte_cap or len(trimmed) <= 1:
                atlases.append(AtlasLayout(width=width, height=height, placements=trimmed))
            else:
                atlases.extend(
                    _split_by_byte_cap(width, height, trimmed, byte_cap, channels)
                )

    return atlases
