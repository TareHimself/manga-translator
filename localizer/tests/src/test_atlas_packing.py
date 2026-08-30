"""Tests for the reusable atlas_packing.pack_into_atlases geometry."""

from comic_localizer.atlas_packing import pack_into_atlases


def _placements_by_id(layouts):
    return {p.id: (layout, p) for layout in layouts for p in layout.placements}


def test_oversized_item_gets_its_own_solo_layout():
    items = [(0, 100, 100), (1, 2000, 100)]
    layouts = pack_into_atlases(items, bin_size=300, padding=20, byte_cap=10**9)

    by_id = _placements_by_id(layouts)
    oversized_layout, oversized_placement = by_id[1]

    assert len(oversized_layout.placements) == 1
    assert oversized_placement.width == 2000
    assert oversized_placement.height == 100


def test_every_item_appears_exactly_once_across_multiple_atlases():
    items = [(i, 150, 150) for i in range(10)]
    layouts = pack_into_atlases(items, bin_size=300, padding=10, byte_cap=10**9)

    assert len(layouts) > 1

    all_ids = [p.id for layout in layouts for p in layout.placements]
    assert sorted(all_ids) == sorted(i for i, _, _ in items)
    assert len(all_ids) == len(set(all_ids))


def test_no_two_placements_overlap_once_padded():
    items = [(i, 80, 60) for i in range(12)]
    layouts = pack_into_atlases(items, bin_size=400, padding=20, byte_cap=10**9)

    for layout in layouts:
        placements = layout.placements
        for i in range(len(placements)):
            for j in range(i + 1, len(placements)):
                a, b = placements[i], placements[j]
                a_rect = (a.x, a.y, a.x + a.width, a.y + a.height)
                b_rect = (b.x, b.y, b.x + b.width, b.y + b.height)
                # inflate both by half the padding - true crop content, unpadded,
                # must not overlap even after eating into half the gap each
                pad = 10
                a_inflated = (
                    a_rect[0] - pad,
                    a_rect[1] - pad,
                    a_rect[2] + pad,
                    a_rect[3] + pad,
                )
                overlap_x = a_inflated[0] < b_rect[2] and b_rect[0] < a_inflated[2]
                overlap_y = a_inflated[1] < b_rect[3] and b_rect[1] < a_inflated[3]
                assert not (overlap_x and overlap_y)


def test_splits_atlas_once_over_byte_cap():
    # Two 100x100 crops packed together would be ~ (100+pad)*2 * 100 * 3 bytes,
    # comfortably over a byte_cap set far below that.
    items = [(0, 100, 100), (1, 100, 100)]
    tiny_cap = 1000  # bytes, forces a split
    layouts = pack_into_atlases(items, bin_size=400, padding=10, byte_cap=tiny_cap)

    assert len(layouts) == 2
    all_ids = [p.id for layout in layouts for p in layout.placements]
    assert sorted(all_ids) == [0, 1]


def test_single_item_over_byte_cap_is_still_sent_alone():
    items = [(0, 500, 500)]
    tiny_cap = 1000  # far smaller than 500*500*3
    layouts = pack_into_atlases(items, bin_size=1000, padding=10, byte_cap=tiny_cap)

    assert len(layouts) == 1
    assert len(layouts[0].placements) == 1
    assert layouts[0].placements[0].id == 0


def test_empty_input_returns_no_layouts():
    assert pack_into_atlases([], bin_size=300, padding=10, byte_cap=10**9) == []
