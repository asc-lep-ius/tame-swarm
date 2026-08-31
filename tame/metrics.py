"""One place every measured number leaves the training loop through.

#12 asked for held-out loss and perplexity to be logged "as first-class metrics
via #7" and, since #7 -- MLflow tracking -- wasn't built yet, routed everything
through this sink instead: the number is what has value, the backend is where
it is filed. #7 now exists, and it plugs in here rather than at each call site
-- every ``log()`` forwards its metrics to ``tracking.log_step`` after writing
the JSONL line, so nothing upstream of this file had to change to gain MLflow
curves. ``tracking.log_step`` is a no-op with no active or pending run (no
``mlflow`` installed, or nobody called ``tracking.init_tracking``), which is
what keeps a bare ``MetricSink`` -- the shape every test below still
constructs -- from starting a run of its own.

Names are namespaced and the namespaces are load-bearing rather than tidy:
``train/`` is a statistic of the batch the model just fit, ``eval/`` is the
held-out number, ``spec/`` is a functional specialisation probe and ``wealth/`` is
an economy diagnostic. The defect #12 exists to correct is precisely that a
training-batch perplexity was read as though it were a held-out one, so the two
can never again share a name.
"""

import json
import logging
from pathlib import Path
from typing import Any

from tracking import log_step

logger = logging.getLogger(__name__)


class MetricSink:
    """Append-only JSONL metric log, one object per step per namespace group.

    Opened lazily and flushed per write: a training run that dies at step 4000 must
    leave the 3999 steps before it readable, which a buffered handle does not
    guarantee.
    """

    def __init__(self, path: str | Path, run_tags: dict[str, Any] | None = None):
        self.path = Path(path)
        self.run_tags = dict(run_tags or {})
        self._handle = None

    def __enter__(self) -> "MetricSink":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Record one measurement group. Non-finite values are written as-is.

        A NaN loss is a fact about the run and dropping it would make the log
        disagree with the console, which is where a diverged run is diagnosed from.
        ``json.dumps`` emits bare ``NaN``/``Infinity``, which Python reads back and
        strict JSON parsers reject -- an acceptable trade for a file whose reader is
        this project.
        """
        if self._handle is None:
            # Created here rather than in __init__: the trainer builds its sink
            # while constructing, and a constructor that makes a directory leaves
            # an empty ./tame_checkpoints behind for anything that merely imports
            # and instantiates -- a test, a --help, a packaging tool walking the
            # tree. Nothing exists on disk until there is something to write.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")

        record = {"step": step, **self.run_tags, **metrics}
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

        # run_tags (router, seed) are dimensions of the run, not metrics of it --
        # MLflow already has them as params from tracking.init_tracking, so only
        # the measurements themselves go to log_step.
        log_step(step, metrics)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
