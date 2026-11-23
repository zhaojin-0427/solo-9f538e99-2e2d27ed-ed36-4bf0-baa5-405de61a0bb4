"""SQLite storage for the battery follow-up ledger and survival summaries.

Three pieces of state are written together by every import:

``battery_ledger``
    The auditable ledger: one current row per battery.
``battery_summary``
    The compact per-battery summary used to maintain the aggregates.  It is a
    single row per battery, updated in place on a correction and deleted on a
    delete.
``battery_summary_usage``
    Lifetime count of summary records written for a battery (insert, correction
    and delete all count).  A battery may produce at most
    ``MAX_SUMMARY_RECORDS_PER_BATTERY`` (6) of them; further changes are
    rejected with an actionable error.  This counter survives deletion so a
    delete/re-insert cycle still counts against the same budget.
``summary_counts``
    Pre-aggregated counts keyed by activation date and day offsets.  This is the
    truncated follow-up view used while the cutoff is earlier than some
    follow-up end date (a subject's age at the cutoff depends on its activation
    date, so activation must stay in the key there).
``age_counts``
    Counts keyed only by (entry_age, end_age, outcome).  Once the cutoff covers
    every follow-up end date no clipping can happen, so activation dates no
    longer matter: the analysis page reads this table instead and its row count
    depends only on the number of distinct age combinations.
``followup_end_counts``
    Subject counts per absolute follow-up end date.  Used to answer "is any
    follow-up still open past the cutoff?" and to count the subjects ending on
    the cutoff day in the cutoff-day risk set.
``aggregate_state``
    One-row marker recording that the two aggregate tables above have been
    backfilled for an existing database volume.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

MAX_SUMMARY_RECORDS_PER_BATTERY = 6
AGGREGATES_KEY = "age_aggregates_v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS battery_ledger (
    battery_no      TEXT PRIMARY KEY,
    activation_date TEXT NOT NULL,
    followup_start  TEXT NOT NULL,
    followup_end    TEXT NOT NULL,
    outcome         TEXT NOT NULL CHECK (outcome IN ('failure', 'censored')),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- One compact summary record per battery.  Corrections overwrite this row and
-- bump ``revision`` instead of appending, which keeps the per-battery summary
-- record count at one (well under the six-record bound).
CREATE TABLE IF NOT EXISTS battery_summary (
    battery_no      TEXT PRIMARY KEY,
    activation_date TEXT NOT NULL,
    entry_age       INTEGER NOT NULL CHECK (entry_age >= 0),
    end_age         INTEGER NOT NULL,
    event           INTEGER NOT NULL CHECK (event IN (0, 1)),
    revision        INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    CHECK (end_age >= entry_age)
);

-- Lifetime summary-record usage per battery.  Insert/correct/delete each
-- count once; the counter is never reset, including after a delete.
CREATE TABLE IF NOT EXISTS battery_summary_usage (
    battery_no      TEXT PRIMARY KEY,
    summary_records INTEGER NOT NULL CHECK (summary_records >= 0)
);

-- Aggregate counts read by the survival query.  The activation date stays in
-- the key because a subject's age on a calendar cutoff depends on its own
-- activation date.  ``entry_age``/``end_age`` are whole days from activation.
CREATE TABLE IF NOT EXISTS summary_counts (
    activation_date TEXT NOT NULL,
    entry_age       INTEGER NOT NULL,
    end_age         INTEGER NOT NULL,
    event           INTEGER NOT NULL CHECK (event IN (0, 1)),
    subject_count   INTEGER NOT NULL CHECK (subject_count > 0),
    PRIMARY KEY (activation_date, entry_age, end_age, event)
);

-- Cutoff-safe aggregate: keyed only by the age interval and outcome.  Read by
-- the survival query when the cutoff is at or after every follow-up end date,
-- so no clipping is possible and the activation date is irrelevant.
CREATE TABLE IF NOT EXISTS age_counts (
    entry_age       INTEGER NOT NULL,
    end_age         INTEGER NOT NULL,
    event           INTEGER NOT NULL CHECK (event IN (0, 1)),
    subject_count   INTEGER NOT NULL CHECK (subject_count > 0),
    PRIMARY KEY (entry_age, end_age, event)
);

-- Subject counts per absolute follow-up end date, for cutoff detection and the
-- cutoff-day risk set.
CREATE TABLE IF NOT EXISTS followup_end_counts (
    followup_end    TEXT NOT NULL PRIMARY KEY,
    subject_count   INTEGER NOT NULL CHECK (subject_count > 0)
);

-- Marks that age_counts / followup_end_counts were backfilled for an existing
-- database volume, so the (possibly large) backfill runs at most once.
CREATE TABLE IF NOT EXISTS aggregate_state (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_batches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    imported_at     TEXT NOT NULL,
    total_rows      INTEGER NOT NULL,
    inserted        INTEGER NOT NULL,
    updated         INTEGER NOT NULL,
    deleted         INTEGER NOT NULL DEFAULT 0,
    unchanged       INTEGER NOT NULL,
    summary_records INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class SummaryRecord:
    activation_date: str
    followup_end: str
    entry_age: int
    end_age: int
    event: int


def summary_from_values(values: tuple[str, str, str, str, str]) -> SummaryRecord:
    """Map ledger values to a summary record (day offsets from activation)."""

    _, activation_text, start_text, end_text, outcome = values
    activation = date.fromisoformat(activation_text)
    return SummaryRecord(
        activation_date=activation_text,
        followup_end=end_text,
        entry_age=(date.fromisoformat(start_text) - activation).days,
        end_age=(date.fromisoformat(end_text) - activation).days,
        event=1 if outcome == "failure" else 0,
    )


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    def connect(self) -> sqlite3.Connection:
        parent = Path(self.path).expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def initialize(self) -> None:
        conn = self.connect()
        try:
            # Multiple gunicorn workers may initialize at the same time; retry
            # the idempotent schema setup instead of failing worker startup.
            for attempt in range(6):
                try:
                    conn.execute("PRAGMA journal_mode = WAL")
                    conn.executescript(SCHEMA)
                    _migrate(conn)
                    # Bring an existing volume's follow-ups into the cutoff-safe
                    # aggregates once, without re-reading follow-up history.
                    ensure_aggregates(conn, own_transaction=True)
                    return
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                        raise
                    if attempt == 5:
                        raise
                    time.sleep(0.2 * (attempt + 1))
        finally:
            conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after an initial deployment, if needed."""

    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(import_batches)")
    }
    if "deleted" not in columns:
        conn.execute(
            "ALTER TABLE import_batches ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0"
        )


def _adjust_count(conn: sqlite3.Connection, record: SummaryRecord, delta: int) -> None:
    key = (record.activation_date, record.entry_age, record.end_age, record.event)
    if delta > 0:
        conn.execute(
            "INSERT INTO summary_counts "
            "(activation_date, entry_age, end_age, event, subject_count) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (activation_date, entry_age, end_age, event) DO UPDATE "
            "SET subject_count = subject_count + excluded.subject_count",
            (*key, delta),
        )
    elif delta < 0:
        # Drop a bucket when the decrement would empty it (the positive-count
        # CHECK forbids storing zero), then decrement the remaining buckets.
        conn.execute(
            "DELETE FROM summary_counts WHERE activation_date = ? AND entry_age = ? "
            "AND end_age = ? AND event = ? AND subject_count <= ?",
            (*key, -delta),
        )
        conn.execute(
            "UPDATE summary_counts SET subject_count = subject_count + ? "
            "WHERE activation_date = ? AND entry_age = ? AND end_age = ? AND event = ?",
            (delta, *key),
        )


def _adjust_age_count(
    conn: sqlite3.Connection, record: SummaryRecord, delta: int
) -> None:
    """Adjust ``age_counts`` (age interval + outcome only)."""

    key = (record.entry_age, record.end_age, record.event)
    if delta > 0:
        conn.execute(
            "INSERT INTO age_counts (entry_age, end_age, event, subject_count) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT (entry_age, end_age, event) DO UPDATE "
            "SET subject_count = subject_count + excluded.subject_count",
            (*key, delta),
        )
    elif delta < 0:
        conn.execute(
            "DELETE FROM age_counts WHERE entry_age = ? AND end_age = ? AND event = ? "
            "AND subject_count <= ?",
            (*key, -delta),
        )
        conn.execute(
            "UPDATE age_counts SET subject_count = subject_count + ? "
            "WHERE entry_age = ? AND end_age = ? AND event = ?",
            (delta, *key),
        )


def _adjust_followup_end_count(
    conn: sqlite3.Connection, followup_end: str, delta: int
) -> None:
    """Adjust ``followup_end_counts`` for one absolute follow-up end date."""

    if delta > 0:
        conn.execute(
            "INSERT INTO followup_end_counts (followup_end, subject_count) "
            "VALUES (?, ?) "
            "ON CONFLICT (followup_end) DO UPDATE "
            "SET subject_count = subject_count + excluded.subject_count",
            (followup_end, delta),
        )
    elif delta < 0:
        conn.execute(
            "DELETE FROM followup_end_counts WHERE followup_end = ? AND subject_count <= ?",
            (followup_end, -delta),
        )
        conn.execute(
            "UPDATE followup_end_counts SET subject_count = subject_count + ? "
            "WHERE followup_end = ?",
            (delta, followup_end),
        )


def _apply_record_deltas(
    conn: sqlite3.Connection, record: SummaryRecord, delta: int
) -> None:
    _adjust_count(conn, record, delta)
    _adjust_age_count(conn, record, delta)
    _adjust_followup_end_count(conn, record.followup_end, delta)


def apply_summary_changes(
    conn: sqlite3.Connection,
    changes: list[tuple[SummaryRecord | None, SummaryRecord | None]],
) -> None:
    """Incrementally apply (old, new) summary transitions.

    Only the buckets touched by this batch are updated; existing follow-up rows
    are never re-read.  Runs inside the caller's import transaction, so any
    later failure rolls every aggregate back together with the ledger.
    """

    # Release old buckets before acquiring new ones.  A same-key correction may
    # briefly empty a bucket, but it is re-inserted later in this transaction.
    for old, _ in changes:
        if old is not None:
            _apply_record_deltas(conn, old, -1)
    for _, new in changes:
        if new is not None:
            _apply_record_deltas(conn, new, 1)


def usage_for(
    conn: sqlite3.Connection, battery_nos: Sequence[str]
) -> dict[str, int]:
    """Lifetime summary-record counts for the given batteries."""

    usage: dict[str, int] = {}
    nos = list(battery_nos)
    for start in range(0, len(nos), 400):
        chunk = nos[start : start + 400]
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            "SELECT battery_no, summary_records FROM battery_summary_usage "
            f"WHERE battery_no IN ({placeholders})",
            chunk,
        ):
            usage[row[0]] = int(row[1])
    return usage


def record_usage(conn: sqlite3.Connection, battery_no: str) -> int:
    """Consume one summary-record slot and return the new usage count."""

    conn.execute(
        "INSERT INTO battery_summary_usage (battery_no, summary_records) "
        "VALUES (?, 1) "
        "ON CONFLICT (battery_no) DO UPDATE "
        "SET summary_records = battery_summary_usage.summary_records + 1",
        (battery_no,),
    )
    return int(
        conn.execute(
            "SELECT summary_records FROM battery_summary_usage WHERE battery_no = ?",
            (battery_no,),
        ).fetchone()[0]
    )


def rebuild_summary(conn: sqlite3.Connection) -> None:
    """Recompute every aggregate from the ledger.

    Maintenance/recovery helper; normal imports update each aggregate
    incrementally and never re-read historical follow-up rows.
    """

    from collections import Counter

    conn.execute("DELETE FROM battery_summary")
    conn.execute("DELETE FROM summary_counts")
    conn.execute("DELETE FROM age_counts")
    conn.execute("DELETE FROM followup_end_counts")

    counts: Counter[tuple[str, int, int, int]] = Counter()
    age_counts: Counter[tuple[int, int, int]] = Counter()
    end_counts: Counter[str] = Counter()
    now = conn.execute("SELECT datetime('now')").fetchone()[0]
    for row in conn.execute(
        "SELECT battery_no, activation_date, followup_start, followup_end, outcome, "
        "created_at FROM battery_ledger ORDER BY battery_no"
    ):
        record = summary_from_values(tuple(row[:5]))
        conn.execute(
            "INSERT INTO battery_summary "
            "(battery_no, activation_date, entry_age, end_age, event, revision, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (
                row[0],
                record.activation_date,
                record.entry_age,
                record.end_age,
                record.event,
                row[5] or now,
                now,
            ),
        )
        counts[
            (
                record.activation_date,
                record.entry_age,
                record.end_age,
                record.event,
            )
        ] += 1
        age_counts[(record.entry_age, record.end_age, record.event)] += 1
        end_counts[record.followup_end] += 1

    conn.executemany(
        "INSERT INTO summary_counts "
        "(activation_date, entry_age, end_age, event, subject_count) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (activation_date, entry_age, end_age, event, count)
            for (activation_date, entry_age, end_age, event), count in sorted(
                counts.items()
            )
        ],
    )
    conn.executemany(
        "INSERT INTO age_counts (entry_age, end_age, event, subject_count) "
        "VALUES (?, ?, ?, ?)",
        [
            (entry_age, end_age, event, count)
            for (entry_age, end_age, event), count in sorted(age_counts.items())
        ],
    )
    conn.executemany(
        "INSERT INTO followup_end_counts (followup_end, subject_count) VALUES (?, ?)",
        sorted(end_counts.items()),
    )
    _mark_aggregates_ready(conn)


def _aggregates_ready(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM aggregate_state WHERE key = ?", (AGGREGATES_KEY,)
    ).fetchone()
    return row is not None


def _mark_aggregates_ready(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO aggregate_state (key, value) VALUES (?, '1') "
        "ON CONFLICT (key) DO NOTHING",
        (AGGREGATES_KEY,),
    )


def ensure_aggregates(
    conn: sqlite3.Connection, *, own_transaction: bool = False
) -> None:
    """Backfill the cutoff-safe aggregates once for a pre-existing volume.

    Called during startup and, defensively, at the start of every import.  A
    volume created before these tables existed is incorporated transparently on
    first use.  The backfill derives everything from the pre-existing
    activation-keyed ``summary_counts`` (the table the page read before this
    optimisation), falling back to the equally pre-aggregated
    ``battery_summary``.  Neither is the raw follow-up history, so an existing
    SQLite volume needs no re-import and its follow-up history is never
    re-read.

    With ``own_transaction`` the check and backfill run in one ``BEGIN
    IMMEDIATE`` transaction so concurrent workers initializing together only
    backfill once and cannot observe a half-built aggregate.
    """

    if own_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        if _aggregates_ready(conn):
            if own_transaction:
                conn.execute("COMMIT")
            return
        conn.execute("DELETE FROM age_counts")
        conn.execute("DELETE FROM followup_end_counts")

        has_summary_counts = conn.execute(
            "SELECT 1 FROM summary_counts LIMIT 1"
        ).fetchone()
        if has_summary_counts:
            conn.execute(
                "INSERT INTO age_counts (entry_age, end_age, event, subject_count) "
                "SELECT entry_age, end_age, event, SUM(subject_count) "
                "FROM summary_counts GROUP BY entry_age, end_age, event"
            )
            conn.execute(
                "INSERT INTO followup_end_counts (followup_end, subject_count) "
                "SELECT date(activation_date, '+' || end_age || ' days'), "
                "SUM(subject_count) FROM summary_counts GROUP BY 1"
            )
        else:
            conn.execute(
                "INSERT INTO age_counts (entry_age, end_age, event, subject_count) "
                "SELECT entry_age, end_age, event, COUNT(*) "
                "FROM battery_summary GROUP BY entry_age, end_age, event"
            )
            # The absolute follow-up end is only stored as a day offset;
            # rebuild it from activation + end age (exact by construction).
            conn.execute(
                "INSERT INTO followup_end_counts (followup_end, subject_count) "
                "SELECT date(activation_date, '+' || end_age || ' days'), COUNT(*) "
                "FROM battery_summary GROUP BY 1"
            )
        _mark_aggregates_ready(conn)
        if own_transaction:
            conn.execute("COMMIT")
    except BaseException:
        if own_transaction:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        raise


def load_survival(
    conn: sqlite3.Connection, cutoff_text: str
) -> tuple[str, list, int, int]:
    """Read one consistent snapshot and describe the curve for ``cutoff_text``.

    Returns ``(mode, rows, total_subjects, risk_at_cutoff)``:

    * ``mode == "fast"`` and ``rows`` are ``age_counts`` rows when the cutoff
      covers every follow-up end date.
    * otherwise ``mode == "clipped"`` and ``rows`` are the original
      activation-keyed ``summary_counts`` rows.

    The "any follow-up past the cutoff" test and the aggregate read happen in
    the same deferred read transaction, so the chosen mode and the metrics
    always reflect one committed state.  In WAL mode this read does not block a
    concurrent import.
    """

    conn.execute("BEGIN")
    try:
        row = conn.execute(
            "SELECT 1 FROM followup_end_counts WHERE followup_end > ? LIMIT 1",
            (cutoff_text,),
        ).fetchone()
        if row is None:
            risk_row = conn.execute(
                "SELECT subject_count FROM followup_end_counts WHERE followup_end = ?",
                (cutoff_text,),
            ).fetchone()
            rows = [
                (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
                for r in conn.execute(
                    "SELECT entry_age, end_age, event, subject_count FROM age_counts"
                )
            ]
            conn.execute("COMMIT")
            return (
                "fast",
                rows,
                sum(r[3] for r in rows),
                int(risk_row[0]) if risk_row is not None else 0,
            )

        rows = [
            (
                r["activation_date"],
                int(r["entry_age"]),
                int(r["end_age"]),
                int(r["event"]),
                int(r["subject_count"]),
            )
            for r in conn.execute(
                "SELECT activation_date, entry_age, end_age, event, subject_count "
                "FROM summary_counts"
            )
        ]
        conn.execute("COMMIT")
        return "clipped", rows, sum(r[4] for r in rows), 0
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise


def ledger_rows(conn: sqlite3.Connection):
    return conn.execute(
        "SELECT battery_no, activation_date, followup_start, followup_end, outcome "
        "FROM battery_ledger ORDER BY battery_no"
    )
