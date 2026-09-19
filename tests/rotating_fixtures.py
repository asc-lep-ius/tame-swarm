"""One rotation small enough to read, for the tests of #41's stream and its margins.

The dates are the thing to read here rather than the text. The stream's items are
published after the checkpoint's cutoff, because that is the whole claim the stream
makes; the canaries' are published well *before* it, because a canary is static by
construction and its date is when it was planted, not evidence of anything held out.
"""

from datetime import date

from rotating_stream import CANARY_MANIFEST, STREAM_MANIFEST, DatedItem, DatedManifest

# The checkpoint's data cutoff, the day both manifests were last rotated, and the
# day the test reads them on -- nine days into a thirty-day cadence, so the stream
# is comfortably in date and a test that moves either end is testing the schedule.
CUTOFF = date(2026, 1, 1)
REFRESHED = date(2026, 9, 1)
TODAY = date(2026, 9, 10)

STREAM_PUBLISHED = date(2026, 6, 14)
CANARY_PLANTED = date(2025, 3, 2)

# Long enough that every prompt and its answer fit with room to spare, short enough
# that a tiny model's forward is free. ``FakeTokenizer`` is one id per character.
MAX_SEQ_LENGTH = 48

ANSWERS = (" alpha", " beta", " gamma", " delta")

# Eight, not four: ``viability_margins.MIN_CANARIES`` refuses a set small enough
# that one canary moves the accuracy by more than the divergence tolerance, and a
# fixture below that floor could not exercise the detector it is there to test.
CANARY_COUNT = 8


def dated_items(prefix: str, count: int, published: date) -> tuple[DatedItem, ...]:
    return tuple(
        DatedItem(
            item_id=f"{prefix}-{index:02d}",
            published=published,
            prompt=f"{prefix} item {index:02d} ends in",
            answer=ANSWERS[index % len(ANSWERS)],
        )
        for index in range(count)
    )


def stream_manifest(
    count: int = 12,
    refreshed: date = REFRESHED,
    refresh_days: int = 30,
    published: date = STREAM_PUBLISHED,
) -> DatedManifest:
    return DatedManifest(
        name=STREAM_MANIFEST,
        items=dated_items("stream", count, published),
        refreshed=refreshed,
        refresh_days=refresh_days,
    )


def canary_manifest(
    count: int = CANARY_COUNT,
    refreshed: date = REFRESHED,
    refresh_days: int = 90,
    published: date = CANARY_PLANTED,
) -> DatedManifest:
    return DatedManifest(
        name=CANARY_MANIFEST,
        items=dated_items("canary", count, published),
        refreshed=refreshed,
        refresh_days=refresh_days,
    )
