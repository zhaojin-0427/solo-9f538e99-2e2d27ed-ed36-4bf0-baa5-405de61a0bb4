"""Aggregate optimisation: equivalence, incremental maintenance and concurrency.

The fast (cutoff-covering) path must return exactly the metrics of the original
activation-keyed path, imports must only touch the buckets they change, and a
survival query must neither block nor observe a partial import.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from analysis import summarize, summarize_ages
from app import create_app
from db import Database, load_survival
from importer import apply_import, parse_csv

CSV_HEADER = (
    "battery_no,activation_date,followup_start,followup_end,outcome,action\n"
)


def csv_bytes(*lines: str) -> bytes:
    return (CSV_HEADER + "\n".join(lines) + "\n").encode("utf-8")


def import_lines(database, *lines: str):
    conn = database.connect()
    try:
        return apply_import(conn, "batch.csv", parse_csv(csv_bytes(*lines)))
    finally:
        conn.close()


def _slow_rows(database):
    conn = database.connect()
    try:
        return [
            (
                row["activation_date"],
                int(row["entry_age"]),
                int(row["end_age"]),
                int(row["event"]),
                int(row["subject_count"]),
            )
            for row in conn.execute(
                "SELECT activation_date, entry_age, end_age, event, subject_count "
                "FROM summary_counts"
            )
        ]
    finally:
        conn.close()


def _count(database, table: str) -> int:
    conn = database.connect()
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _snapshot(database):
    """All aggregate tables, for exact incremental-delta assertions."""

    conn = database.connect()
    try:
        return {
            "age_counts": [
                tuple(row)
                for row in conn.execute(
                    "SELECT entry_age, end_age, event, subject_count "
                    "FROM age_counts ORDER BY 1, 2, 3"
                )
            ],
            "end_counts": [
                tuple(row)
                for row in conn.execute(
                    "SELECT followup_end, subject_count "
                    "FROM followup_end_counts ORDER BY 1"
                )
            ],
        }
    finally:
        conn.close()


@pytest.fixture()
def database(tmp_path):
    database = Database(tmp_path / "battery.db")
    database.initialize()
    return database


@pytest.fixture()
def client(database, tmp_path):
    app = create_app(
        {"TESTING": True, "DATABASE": str(tmp_path / "battery.db"), "SECRET_KEY": "t"}
    )
    with app.test_client() as client:
        yield client


def _seed(database):
    import_lines(
        database,
        "B1,2026-01-01,2026-01-02,2026-02-01,failure",
        "B2,2026-01-01,2026-01-02,2026-03-01,censored",
        "B3,2026-01-11,2026-01-12,2026-02-10,failure",
        "B4,2026-02-01,2026-02-02,2026-02-20,censored",
    )


# --------------------------------------------------------------------------
# Equivalence of the fast and clipped paths
# --------------------------------------------------------------------------


def test_fast_and_slow_agree_for_every_cutoff(database):
    _seed(database)
    slow = _slow_rows(database)
    conn = database.connect()
    try:
        for offset in range(0, 90):
            cutoff = date(2026, 1, 1) + timedelta(days=offset)
            mode, rows, total, risk = load_survival(conn, cutoff.isoformat())
            expected = summarize(slow, cutoff)
            if mode == "fast":
                actual = summarize_ages(rows, cutoff, risk_at_cutoff=risk)
                assert actual.points == expected.points
                assert actual.enrolled_subjects == expected.enrolled_subjects
                assert actual.risk_set_at_cutoff == expected.risk_set_at_cutoff
                assert total == sum(row[4] for row in slow)
            else:
                assert summarize(rows, cutoff).points == expected.points
    finally:
        conn.close()


def test_cutoff_at_last_end_date_uses_fast_path(database):
    _seed(database)
    conn = database.connect()
    try:
        # 2026-03-01 is the latest follow-up end across the ledger.
        assert load_survival(conn, "2026-03-01")[0] == "fast"
        assert load_survival(conn, "2026-02-28")[0] == "clipped"
        # A battery whose follow-up ends on the cutoff day is still counted in
        # the cutoff-day risk set (B2 ends exactly on 2026-03-01).
        _, _, _, risk = load_survival(conn, "2026-03-01")
        assert risk == 1
        # Nothing ends on 2026-03-02, so the risk set is empty there.
        _, _, _, risk = load_survival(conn, "2026-03-02")
        assert risk == 0
    finally:
        conn.close()


def test_cutoff_day_end_counted_in_risk_set_for_same_activation(database):
    # Two batteries share an activation date and both end on the cutoff day;
    # both must remain in that day's risk set even on the fast path.
    import_lines(
        database,
        "C1,2026-01-01,2026-01-02,2026-02-01,censored",
        "C2,2026-01-01,2026-01-02,2026-02-01,failure",
    )
    conn = database.connect()
    try:
        mode, rows, _total, risk = load_survival(conn, "2026-02-01")
        assert mode == "fast"
        assert risk == 2
        result = summarize_ages(rows, date(2026, 2, 1), risk_at_cutoff=risk)
    finally:
        conn.close()
    assert result.risk_set_at_cutoff == 2
    assert result.points[0].at_risk == 2


def test_fast_path_reads_age_combinations_not_activation_groups(client, database):
    # Many activation dates, few distinct age/outcome combinations.
    lines = []
    for index in range(60):
        activation = date(2026, 1, 1) + timedelta(days=index)
        end = activation + timedelta(days=30)
        lines.append(
            f"B{index},{activation.isoformat()},{activation.isoformat()},"
            f"{end.isoformat()},failure"
        )
    import_lines(database, *lines)
    assert _count(database, "summary_counts") == 60  # one per activation date
    assert _count(database, "age_counts") == 1  # one age/outcome combination

    cutoff = (date(2026, 1, 1) + timedelta(days=200)).isoformat()
    conn = database.connect()
    try:
        mode, rows, total, _risk = load_survival(conn, cutoff)
    finally:
        conn.close()
    assert mode == "fast"
    assert len(rows) == 1
    assert total == 60

    page = client.get(f"/survival?cutoff={cutoff}")
    assert page.status_code == 200


def test_fast_path_result_matches_original_page_metrics(client, database):
    _seed(database)
    slow = summarize(_slow_rows(database), date(2026, 12, 31))
    page = client.get("/survival?cutoff=2026-12-31")
    text = page.get_data(as_text=True)
    assert page.status_code == 200
    # The rendered KM value for the first failure age matches the reference
    # calculation from the original activation-keyed view.
    assert f'{slow.points[0].survival:.6f}' in text
    assert str(slow.points[0].at_risk) in text
    assert slow.points[0].age_days == 30  # B1: 2026-02-01 − 2026-01-01


def test_correction_of_activation_date_updates_ages_but_not_end_date(database):
    """Cross-activation correction: age offsets move, absolute end stays put."""

    import_lines(database, "E1,2026-01-01,2026-01-02,2026-02-01,failure")
    ends_before = dict(_snapshot(database)["end_counts"])
    # Same follow-up dates, different activation => smaller entry/end ages.
    import_lines(database, "E1,2026-01-11,2026-01-12,2026-02-01,failure")

    snapshot = _snapshot(database)
    ages = {row[:3]: row[3] for row in snapshot["age_counts"]}
    assert ages == {(1, 21, 1): 1}
    # The absolute follow-up end date is unchanged, so the end-date aggregate
    # (and therefore cutoff detection) must not move.
    assert dict(snapshot["end_counts"]) == ends_before

    conn = database.connect()
    try:
        mode, rows, total, risk = load_survival(conn, "2026-12-31")
        fast = summarize_ages(rows, date(2026, 12, 31), risk_at_cutoff=risk)
        expected = summarize(_slow_rows(database), date(2026, 12, 31))
    finally:
        conn.close()
    assert mode == "fast"
    assert total == 1
    assert fast.points == expected.points


def test_new_later_followup_flips_page_back_to_clipped_path(client, database):
    """A newly imported later follow-up must disable the fast path for cutoffs
    it no longer covers, while identical metrics continue to be served."""

    import_lines(
        database,
        "F1,2026-01-01,2026-01-02,2026-02-01,failure",
        "F2,2026-01-01,2026-01-02,2026-02-01,censored",
    )
    assert client.get("/survival?cutoff=2026-02-01").status_code == 200
    conn = database.connect()
    try:
        assert load_survival(conn, "2026-02-01")[0] == "fast"
    finally:
        conn.close()

    # A follow-up running past 2026-02-01 is added.
    import_lines(database, "F3,2026-01-01,2026-01-02,2026-03-01,censored")
    conn = database.connect()
    try:
        mode, rows, _total, _risk = load_survival(conn, "2026-02-01")
        assert mode == "clipped"
        actual = summarize(rows, date(2026, 2, 1))
        expected = summarize(_slow_rows(database), date(2026, 2, 1))
    finally:
        conn.close()
    assert actual.points == expected.points
    # F1 and F2 end exactly on the cutoff, so both remain at risk that day;
    # F3 is still open, so it is also at risk.
    assert actual.risk_set_at_cutoff == 3


def test_clipped_mode_counts_cutoff_day_end_in_risk_set(database):
    import_lines(
        database,
        "G1,2026-01-01,2026-01-02,2026-02-01,censored",  # ends exactly on cutoff
        "G2,2026-01-01,2026-01-02,2026-02-01,failure",   # fails exactly on cutoff
        "G3,2026-01-01,2026-01-02,2026-03-01,censored",  # open past cutoff
    )
    conn = database.connect()
    try:
        mode, rows, _total, _risk = load_survival(conn, "2026-02-01")
        assert mode == "clipped"
        result = summarize(rows, date(2026, 2, 1))
    finally:
        conn.close()
    assert result.risk_set_at_cutoff == 3
    # The failure on the cutoff day is counted.
    assert result.points[0].age_days == 31
    assert result.points[0].failures == 1


def test_same_age_across_activation_dates_merges_only_in_fast_path(database):
    """Cross-activation check: identical age records from different dates.

    The two batteries have different activation dates but the same age interval
    and outcome, so ``age_counts`` has one row while ``summary_counts`` has two.
    The fast path must still count both subjects at that age.
    """

    import_lines(
        database,
        "A1,2026-01-01,2026-01-02,2026-02-01,failure",
        "A2,2026-01-15,2026-01-16,2026-02-15,failure",
    )
    assert _count(database, "summary_counts") == 2
    assert _snapshot(database)["age_counts"] == [(1, 31, 1, 2)]

    conn = database.connect()
    try:
        # Covers both ends, so the activation date cannot matter.
        mode, rows, total, risk = load_survival(conn, "2026-02-15")
        fast = summarize_ages(rows, date(2026, 2, 15), risk_at_cutoff=risk)
        expected = summarize(_slow_rows(database), date(2026, 2, 15))
    finally:
        conn.close()
    assert mode == "fast"
    assert total == 2
    assert fast.points == expected.points
    # At age 31 both are exposed (A2's activation merely shifts the calendar).
    assert fast.points[0].at_risk == 2
    assert fast.points[0].failures == 2


def test_correction_across_cutoffs_matches_recomputation(database):
    """Cross-cutoff check: a correction before and after the cutoff.

    Correcting a battery changes the published curve on cutoffs on both sides of
    the change; the incrementally maintained aggregates must equal a full
    recomputation at every cutoff, under both the fast and clipped paths.
    """

    import_lines(
        database,
        "D1,2026-01-01,2026-01-02,2026-01-31,failure",
        "D2,2026-01-01,2026-01-02,2026-02-28,censored",
        "D3,2026-01-01,2026-01-02,2026-03-31,failure",
    )
    conn = database.connect()
    try:
        expected_before = {
            day: summarize(_slow_rows(database), date(2026, 1, 1) + timedelta(days=day))
            for day in (10, 40, 70, 120)
        }
    finally:
        conn.close()

    # Move D1's failure from age 30 to age 55 (2026-02-25).
    import_lines(database, "D1,2026-01-01,2026-01-02,2026-02-25,failure")
    # D2 becomes a failure at its existing end date (crosses the outcome axis).
    import_lines(database, "D2,2026-01-01,2026-01-02,2026-02-28,failure")

    conn = database.connect()
    try:
        for day in (10, 40, 70, 120):
            cutoff = date(2026, 1, 1) + timedelta(days=day)
            mode, rows, _total, risk = load_survival(conn, cutoff.isoformat())
            if mode == "fast":
                actual = summarize_ages(rows, cutoff, risk_at_cutoff=risk)
            else:
                actual = summarize(rows, cutoff)
            expected = summarize(_slow_rows(database), cutoff)
            assert actual.points == expected.points
            assert actual.risk_set_at_cutoff == expected.risk_set_at_cutoff
            assert actual.enrolled_subjects == expected.enrolled_subjects
            # The cutoff-10 curve is genuinely unchanged by a later correction.
            if day == 10:
                assert actual.points == expected_before[10].points
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Incremental aggregate maintenance
# --------------------------------------------------------------------------


def test_insert_only_touches_new_buckets(database):
    _seed(database)
    before = _snapshot(database)
    import_lines(database, "B9,2026-05-01,2026-05-02,2026-06-01,censored")
    after = _snapshot(database)
    assert [row for row in after["age_counts"] if row not in before["age_counts"]] == [
        (1, 31, 0, 1)
    ]
    # Every previous bucket is preserved unchanged; only the newest end date
    # gains a subject.  No historical follow-up was re-read or recomputed.
    assert set(before["age_counts"]).issubset(set(after["age_counts"]))
    assert ("2026-06-01", 1) in after["end_counts"]


def test_correction_moves_exactly_old_and_new_buckets(database):
    _seed(database)
    # Correct B1: same activation, different end/outcome.
    import_lines(database, "B1,2026-01-01,2026-01-02,2026-02-15,failure")
    snapshot = _snapshot(database)
    ages = {row[:3]: row[3] for row in snapshot["age_counts"]}
    ends = dict(snapshot["end_counts"])
    # The old (1, 31, event=1) bucket is gone, the new (1, 45, event=1) exists.
    assert ages[(1, 45, 1)] == 1
    assert ends["2026-02-15"] == 1
    assert "2026-02-01" not in ends

    # Correcting B2 (censored) to a failure at the same end date: only the
    # outcome dimension moves.
    import_lines(database, "B2,2026-01-01,2026-01-02,2026-03-01,failure")
    ages = {row[:3]: row[3] for row in _snapshot(database)["age_counts"]}
    assert (1, 59, 0) not in ages  # the old censored bucket is emptied and dropped
    assert ages[(1, 59, 1)] == 1


def test_delete_removes_buckets(database):
    _seed(database)
    import_lines(database, "B4,,,,,delete")
    snapshot = _snapshot(database)
    ends = dict(snapshot["end_counts"])
    assert ends.get("2026-02-20") is None


def test_correction_updates_only_changed_batteries_buckets(database):
    """A correction must move exactly the changed battery's buckets.

    After the import, every other battery's buckets are unchanged, showing that
    only the (old, new) delta for B3 was applied rather than a full recompute.
    """

    _seed(database)
    before = {row[:3]: row[3] for row in _snapshot(database)["age_counts"]}

    import_lines(database, "B3,2026-01-11,2026-01-12,2026-02-20,censored")

    snapshot = _snapshot(database)
    ages = {row[:3]: row[3] for row in snapshot["age_counts"]}
    ends = dict(snapshot["end_counts"])
    # B3's old failure bucket (1, 30, event=1) is gone; its new censored bucket
    # (1, 40, event=0) appears.
    assert ages.get((1, 30, 1)) is None
    assert ages[(1, 40, 0)] == 1
    assert "2026-02-10" not in ends
    # B4 also ends on 2026-02-20, so that day now has two subjects.
    assert ends["2026-02-20"] == 2
    # Every other battery's bucket is untouched.
    for key, count in before.items():
        if key == (1, 30, 1):
            continue
        assert ages[key] == count


# --------------------------------------------------------------------------
# Legacy volume ingestion and failure atomicity
# --------------------------------------------------------------------------


OLD_SCHEMA = """
CREATE TABLE battery_ledger (
    battery_no TEXT PRIMARY KEY, activation_date TEXT NOT NULL,
    followup_start TEXT NOT NULL, followup_end TEXT NOT NULL,
    outcome TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE battery_summary (
    battery_no TEXT PRIMARY KEY, activation_date TEXT NOT NULL,
    entry_age INTEGER NOT NULL, end_age INTEGER NOT NULL, event INTEGER NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE battery_summary_usage (
    battery_no TEXT PRIMARY KEY, summary_records INTEGER NOT NULL
);
CREATE TABLE summary_counts (
    activation_date TEXT NOT NULL, entry_age INTEGER NOT NULL,
    end_age INTEGER NOT NULL, event INTEGER NOT NULL,
    subject_count INTEGER NOT NULL,
    PRIMARY KEY (activation_date, entry_age, end_age, event)
);
-- The pre-optimisation import_batches lacked the ``deleted`` column.
CREATE TABLE import_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL,
    imported_at TEXT NOT NULL, total_rows INTEGER NOT NULL,
    inserted INTEGER NOT NULL, updated INTEGER NOT NULL,
    unchanged INTEGER NOT NULL, summary_records INTEGER NOT NULL
);
"""


def test_existing_volume_is_optimised_without_reimport(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    now = "2026-01-01T00:00:00+00:00"
    for no, activation, end, outcome in (
        ("B1", "2026-01-01", "2026-02-01", "failure"),
        ("B2", "2026-01-01", "2026-03-01", "censored"),
    ):
        conn.execute(
            "INSERT INTO battery_ledger VALUES (?, ?, ?, ?, ?, ?, ?)",
            (no, activation, activation, end, outcome, now, now),
        )
    # The legacy volume already had activation-keyed aggregates populated.
    conn.execute(
        "INSERT INTO summary_counts VALUES ('2026-01-01', 0, 31, 1, 1)"
    )
    conn.execute(
        "INSERT INTO summary_counts VALUES ('2026-01-01', 0, 59, 0, 1)"
    )
    conn.execute(
        "INSERT INTO battery_summary VALUES "
        "('B1', '2026-01-01', 0, 31, 1, 1, ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT INTO battery_summary VALUES "
        "('B2', '2026-01-01', 0, 59, 0, 1, ?, ?)",
        (now, now),
    )
    conn.commit()
    conn.close()

    database = Database(path)
    database.initialize()  # migrates + backfills once

    conn = database.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM age_counts").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM followup_end_counts").fetchone()[0] == 2
        mode, rows, total, risk = load_survival(conn, "2026-03-01")
    finally:
        conn.close()
    assert mode == "fast"
    assert total == 2
    assert risk == 1  # B2 ends exactly on the cutoff
    assert summarize_ages(rows, date(2026, 3, 1), risk_at_cutoff=risk).points[0].age_days == 31

    # Re-initialising does not backfill twice.
    database.initialize()
    assert _count(database, "age_counts") == 2


def test_existing_volume_backfill_falls_back_to_battery_summary(tmp_path):
    """Legacy volume with battery_summary but no activation-keyed aggregates."""

    path = tmp_path / "legacy2.db"
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO battery_summary VALUES ('B1', '2026-01-01', 0, 31, 1, 1, ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT INTO battery_summary VALUES ('B2', '2026-01-05', 2, 40, 0, 1, ?, ?)",
        (now, now),
    )
    conn.commit()
    conn.close()

    database = Database(path)
    database.initialize()

    conn = database.connect()
    try:
        assert _snapshot(database)["age_counts"] == [
            (0, 31, 1, 1),
            (2, 40, 0, 1),
        ]
        # followup end dates reconstructed from activation + end age:
        # B1 -> 2026-02-01, B2 -> 2026-02-14.
        ends = dict(_snapshot(database)["end_counts"])
        assert ends == {"2026-02-01": 1, "2026-02-14": 1}
        mode, rows, total, risk = load_survival(conn, "2026-02-14")
    finally:
        conn.close()
    assert mode == "fast"
    assert total == 2
    assert risk == 1  # B2 ends on the cutoff
    assert summarize_ages(rows, date(2026, 2, 14), risk_at_cutoff=risk).points[0].age_days == 31


def test_empty_database_survival_page_is_fast_and_empty(client):
    page = client.get("/survival?cutoff=2026-12-31")
    assert page.status_code == 200
    assert "尚没有发生故障" in page.get_data(as_text=True)


def test_failed_import_keeps_aggregates_and_page(database, client):
    _seed(database)
    before = _snapshot(database)
    with pytest.raises(Exception):
        import_lines(
            database,
            "B7,2026-01-01,2026-01-02,2026-02-01,failure",
            "B8,2026-01-01,not-a-date,2026-02-01,failure",
        )
    assert _snapshot(database) == before
    assert client.get("/survival?cutoff=2026-12-31").status_code == 200


# --------------------------------------------------------------------------
# Read / write concurrency and single-snapshot consistency
# --------------------------------------------------------------------------


def test_survival_read_does_not_block_concurrent_import(database):
    _seed(database)

    reader = database.connect()
    try:
        # Hold a read snapshot open, exactly like a long survival query.
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM age_counts").fetchone()

        # A different connection must still be able to import in WAL mode.
        import_lines(database, "B9,2026-05-01,2026-05-02,2026-06-01,censored")

        # The open snapshot still sees the pre-import state (consistent), and
        # the import is committed and visible to new readers.
        assert reader.execute("SELECT COUNT(*) FROM age_counts").fetchone()[0] == 4
        reader.execute("COMMIT")
    finally:
        reader.close()

    conn = database.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM age_counts").fetchone()[0] == 5
    finally:
        conn.close()


def test_load_survival_mode_and_rows_come_from_one_snapshot(database):
    _seed(database)
    # The cutoff covers all ends, so mode + totals must describe the same state
    # in which no follow-up end is greater than the cutoff.
    conn = database.connect()
    try:
        mode, _rows, total, _risk = load_survival(conn, "2026-03-01")
        max_end = conn.execute(
            "SELECT MAX(followup_end) FROM followup_end_counts"
        ).fetchone()[0]
    finally:
        conn.close()
    assert mode == "fast"
    assert max_end <= "2026-03-01"
    assert total == 4


def test_import_completes_while_survival_query_holds_snapshot(database):
    """A long survival read must not prevent another window's import."""

    import threading

    _seed(database)
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        conn = database.connect()
        try:
            conn.execute("BEGIN")
            conn.execute("SELECT COUNT(*) FROM age_counts").fetchone()  # snapshot
            started.set()
            release.wait(timeout=10)
            # The open snapshot still sees the pre-import state.
            count = conn.execute("SELECT COUNT(*) FROM age_counts").fetchone()[0]
            if count != 4:
                errors.append(AssertionError(f"snapshot saw {count} rows, expected 4"))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            conn.close()

    thread = threading.Thread(target=reader)
    thread.start()
    assert started.wait(timeout=10)
    try:
        # Import in another "window" while the read snapshot is still open.
        # It must complete promptly (timeout is a hard regression signal).
        done: list = []
        importer = threading.Thread(
            target=lambda: done.append(
                import_lines(database, "B9,2026-05-01,2026-05-02,2026-06-01,censored")
            )
        )
        importer.start()
        importer.join(timeout=10)
        assert not importer.is_alive(), "import was blocked by the open read"
        assert done and done[0].inserted == 1
    finally:
        release.set()
        thread.join(timeout=10)

    assert not errors, errors
    conn = database.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM age_counts").fetchone()[0] == 5
    finally:
        conn.close()
