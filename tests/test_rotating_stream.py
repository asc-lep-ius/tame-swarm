"""The stream is newer than the last update, and the schedule is not the reader's.

Three properties #41 turns on, each with the failure it exists to make loud: an
item the checkpoint could have trained on is dropped and said, a manifest that
stopped rotating is refused rather than quietly read as a fixed corpus, and the
canaries sit where their own fingerprint puts them rather than where a caller asks.
"""

import json
from dataclasses import replace
from datetime import date

import pytest
import torch

from rotating_stream import (
    DEFAULT_REFRESH_DAYS,
    MANIFEST_VERSION,
    DatedItem,
    DatedManifest,
    RotationRecord,
    StaleStreamError,
    build_rotating_stream,
    load_manifest,
    place_canaries,
)

from .conftest import FakeTokenizer
from .rotating_fixtures import (
    CANARY_COUNT,
    CANARY_PLANTED,
    CUTOFF,
    MAX_SEQ_LENGTH,
    REFRESHED,
    STREAM_PUBLISHED,
    TODAY,
    canary_manifest,
    dated_items,
    stream_manifest,
)


def _build(tokenizer, stream=None, canaries=None, **kwargs):
    return build_rotating_stream(
        stream or stream_manifest(),
        canaries or canary_manifest(),
        tokenizer,
        MAX_SEQ_LENGTH,
        cutoff=CUTOFF,
        today=TODAY,
        **kwargs,
    )


# --- Newer than the last update ----------------------------------------------------------


def test_items_dated_before_the_cutoff_are_dropped_and_said(caplog):
    """A manifest ages into the training set one row at a time as checkpoints move."""
    stale = stream_manifest(count=6, published=date(2025, 5, 1))
    mixed = replace(stale, items=stale.items + dated_items("fresh", 4, STREAM_PUBLISHED))

    with caplog.at_level("WARNING"):
        fresh = mixed.newer_than(CUTOFF)

    assert len(fresh.items) == 4
    assert all(item.published > CUTOFF for item in fresh.items)
    assert "6 of 10 items are dated on or before the checkpoint cutoff" in caplog.text


def test_a_manifest_entirely_before_the_cutoff_is_refused():
    """Empty is not a small margin; it is no measurement."""
    with pytest.raises(ValueError, match="no item published after the checkpoint cutoff"):
        stream_manifest(published=date(2025, 5, 1)).newer_than(CUTOFF)


def test_the_built_stream_carries_only_post_cutoff_documents(fake_tokenizer):
    stale = stream_manifest(count=5, published=date(2025, 5, 1))
    mixed = replace(stale, items=stale.items + dated_items("fresh", 3, STREAM_PUBLISHED))

    built = _build(fake_tokenizer, stream=mixed)

    assert built.num_items == 3 + built.num_canaries
    assert built.rotation.stream_fingerprint == mixed.newer_than(CUTOFF).fingerprint


# --- A schedule the system does not control ------------------------------------------------


def test_a_stream_past_its_refresh_date_is_refused():
    """A stream that stopped rotating is a fixed corpus wearing a rotating name."""
    late = stream_manifest(refreshed=date(2026, 6, 1), refresh_days=30)

    with pytest.raises(StaleStreamError, match="71 days overdue"):
        _build(FakeTokenizer(), stream=late)


def test_a_stale_stream_is_readable_only_deliberately(fake_tokenizer, caplog):
    """The pairing: the refusal is a refusal, not a warning dressed as one.

    And the override says so twice -- once to whoever is watching the run, once on
    the record every margin is quoted beside. An override that only logged would
    be an override that vanished by the time the number was read.
    """
    late = stream_manifest(refreshed=date(2026, 6, 1), refresh_days=30)

    with caplog.at_level("WARNING"):
        built = _build(fake_tokenizer, stream=late, allow_stale=True)

    assert built.rotation.stream_refreshed == "2026-06-01"
    assert built.rotation.stream_days_overdue == 71
    assert "STALE+71d" in built.rotation.label
    assert "71 days past its refresh" in caplog.text


def test_a_fresh_stream_is_labelled_without_a_staleness_note(fake_tokenizer, caplog):
    with caplog.at_level("WARNING"):
        built = _build(fake_tokenizer)

    assert built.rotation.stream_days_overdue == 0
    assert "STALE" not in built.rotation.label
    assert built.rotation.stream_cutoff == CUTOFF.isoformat()
    assert caplog.text == ""


def test_overdue_counts_from_the_cadence_the_manifest_declares():
    manifest = stream_manifest(refreshed=date(2026, 9, 1), refresh_days=30)

    assert manifest.days_overdue(date(2026, 10, 1)) == 0
    assert manifest.days_overdue(date(2026, 10, 2)) == 1
    assert replace(manifest, refresh_days=7).days_overdue(date(2026, 10, 1)) == 23


def test_a_cadence_that_never_comes_due_is_refused():
    with pytest.raises(ValueError, match="not a schedule"):
        stream_manifest(refresh_days=0)


def test_an_empty_manifest_is_refused():
    with pytest.raises(ValueError, match="carries no items"):
        DatedManifest(name="empty", items=(), refreshed=REFRESHED)


# --- Canaries, at positions nothing in the system chooses ------------------------------------


def test_canaries_are_placed_by_their_own_fingerprint():
    """Deterministic given the canary set, and nothing else can ask for an arrangement."""
    stream = stream_manifest()
    canaries = canary_manifest()

    first = place_canaries(stream.items, canaries)[1]
    second = place_canaries(stream.items, canaries)[1]

    assert first == second
    assert sum(first) == len(canaries.items)


def test_rotating_the_canary_set_moves_them():
    """The pairing: positions that never moved would be positions to learn around."""
    stream = stream_manifest(count=40)
    here = place_canaries(stream.items, canary_manifest())[1]
    there = place_canaries(stream.items, canary_manifest(refreshed=date(2026, 8, 1)))[1]

    assert here != there


def test_every_canary_is_marked_and_no_stream_item_is(fake_tokenizer):
    built = _build(fake_tokenizer)

    pairs = zip(built.item_ids, built.is_canary.tolist(), strict=True)
    marked = {item_id for item_id, flag in pairs if flag}

    assert marked == {item.item_id for item in canary_manifest().items}
    assert built.num_canaries == CANARY_COUNT
    assert built.num_items == 12 + CANARY_COUNT


# --- The fingerprint names the rotation ------------------------------------------------------


def test_the_fingerprint_moves_with_the_rows_the_date_and_the_cadence():
    manifest = stream_manifest()
    prints = {
        manifest.fingerprint,
        stream_manifest(count=11).fingerprint,
        stream_manifest(refreshed=date(2026, 8, 1)).fingerprint,
        stream_manifest(refresh_days=14).fingerprint,
    }

    assert len(prints) == 4
    assert manifest.fingerprint == stream_manifest().fingerprint


def test_the_rotation_record_carries_both_schedules_and_reads_as_one_line(fake_tokenizer):
    built = _build(fake_tokenizer)
    rotation = built.rotation

    assert rotation.stream_refreshed == REFRESHED.isoformat()
    assert rotation.stream_refresh_days == 30
    assert rotation.canary_refreshed == REFRESHED.isoformat()
    assert rotation.canary_refresh_days == 90
    assert rotation.canary_fingerprint != rotation.stream_fingerprint
    assert rotation.stream_fingerprint is not None
    assert rotation.stream_fingerprint in rotation.label
    assert REFRESHED.isoformat() in rotation.label


def test_an_absent_rotation_records_nothing_rather_than_a_placeholder():
    assert RotationRecord() == RotationRecord(stream_fingerprint=None)
    assert RotationRecord().stream_fingerprint is None


# --- The answer is located, never shown ------------------------------------------------------


def test_the_prompt_is_all_the_model_sees_and_the_answer_is_the_next_token(fake_tokenizer):
    built = _build(fake_tokenizer)

    for row, item_id in enumerate(built.item_ids):
        length = int(built.attention_mask[row].sum().item())
        assert int(built.answer_positions[row].item()) == length - 1, item_id
        assert bool((built.input_ids[row, length:] == 0).all()), item_id

    expected = fake_tokenizer(["stream item 00 ends in alpha"], max_length=MAX_SEQ_LENGTH)
    first_stream = built.item_ids.index("stream-00")
    position = int(built.answer_positions[first_stream].item())
    assert built.answer_ids[first_stream] == expected["input_ids"][0, position + 1]


def test_a_tokenizer_that_does_not_extend_the_prompt_is_refused():
    """A target token that is silently the wrong one is an accuracy wrong by a constant."""

    class ReversingTokenizer(FakeTokenizer):
        def __call__(self, texts, **kwargs):
            return super().__call__([text[::-1] for text in texts], **kwargs)

    with pytest.raises(ValueError, match="does not encode its prompt as a prefix"):
        _build(ReversingTokenizer())


def test_a_left_padding_tokenizer_is_refused(fake_tokenizer):
    """The prefix check alone passes vacuously here: pad compared against pad.

    Under left padding the first ``length`` positions of both encodings are
    padding, so ``torch.equal`` agrees, the target becomes the pad token and the
    answer position points into the padding. Every item then scores "predict pad
    from pad" -- the accuracy wrong by an invisible constant that the prefix
    assertion exists to refuse, reached through the door it does not cover.
    """

    class LeftPaddingTokenizer(FakeTokenizer):
        def __call__(self, texts, **kwargs):
            encoded = super().__call__(texts, **kwargs)
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            width = ids.shape[1]
            for row in range(ids.shape[0]):
                used = int(mask[row].sum())
                pad = torch.zeros(width - used, dtype=torch.long)
                ids[row] = torch.cat([pad, ids[row, :used]])
                mask[row] = torch.cat([pad, mask[row, :used]])
            return {"input_ids": ids, "attention_mask": mask}

    with pytest.raises(ValueError, match="pads on the left"):
        _build(LeftPaddingTokenizer())


def test_an_answer_that_tokenises_to_nothing_is_refused(fake_tokenizer):
    """A blank answer is a row that cannot be answered, not a row that is wrong."""
    stream = stream_manifest()
    blank = replace(stream, items=(replace(stream.items[0], answer=""),) + stream.items[1:])

    with pytest.raises(ValueError, match="adds no token to its prompt"):
        _build(fake_tokenizer, stream=blank)


def test_an_item_with_no_prompt_is_refused(fake_tokenizer):
    stream = stream_manifest()
    blank = replace(stream, items=(replace(stream.items[0], prompt=""),) + stream.items[1:])

    with pytest.raises(ValueError, match="empty prompt"):
        _build(fake_tokenizer, stream=blank)


def test_a_prompt_that_fills_the_window_is_refused(fake_tokenizer):
    stream = stream_manifest()
    long_prompt = replace(stream.items[0], prompt="x" * (MAX_SEQ_LENGTH + 4))
    crowded = replace(stream, items=(long_prompt,) + stream.items[1:])

    with pytest.raises(ValueError, match="fills the whole"):
        _build(fake_tokenizer, stream=crowded)


# --- The manifest file is validated at the boundary --------------------------------------------


def _write(tmp_path, **overrides):
    payload = {
        "version": MANIFEST_VERSION,
        "name": "rotating-stream",
        "refreshed": REFRESHED.isoformat(),
        "refresh_days": 30,
        "items": [
            {
                "item_id": item.item_id,
                "published": item.published.isoformat(),
                "prompt": item.prompt,
                "answer": item.answer,
            }
            for item in dated_items("stream", 3, STREAM_PUBLISHED)
        ],
    }
    payload.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_a_manifest_round_trips_through_its_file(tmp_path):
    loaded = load_manifest(_write(tmp_path))

    assert loaded.name == "rotating-stream"
    assert loaded.refreshed == REFRESHED
    assert loaded.refresh_days == 30
    assert loaded.items[0] == DatedItem(
        item_id="stream-00",
        published=STREAM_PUBLISHED,
        prompt="stream item 00 ends in",
        answer=" alpha",
    )


def test_a_manifest_without_a_cadence_takes_the_written_down_default(tmp_path):
    payload = json.loads(_write(tmp_path).read_text())
    del payload["refresh_days"]
    path = tmp_path / "no_cadence.json"
    path.write_text(json.dumps(payload))

    assert load_manifest(path).refresh_days == DEFAULT_REFRESH_DAYS


def test_a_manifest_of_another_version_is_refused(tmp_path):
    with pytest.raises(ValueError, match="declares version"):
        load_manifest(_write(tmp_path, version=MANIFEST_VERSION + 1))


def test_a_manifest_row_missing_a_date_is_refused(tmp_path):
    path = _write(tmp_path)
    payload = json.loads(path.read_text())
    del payload["items"][1]["published"]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=r"item 1: missing keys \['published'\]"):
        load_manifest(path)


def test_the_canaries_need_not_be_newer_than_the_cutoff(fake_tokenizer):
    """Static by construction: a canary that rotated could not show memorisation."""
    built = _build(fake_tokenizer)

    assert CANARY_PLANTED < CUTOFF
    assert built.num_canaries == CANARY_COUNT
    assert torch.equal(built.is_canary.sum(), torch.tensor(CANARY_COUNT))
