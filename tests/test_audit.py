"""Tests de vigia_ee.audit: cadena de hashes, verificación y export por rango."""

from __future__ import annotations

from datetime import timedelta

import pytest

from vigia_ee import audit
from vigia_ee.db import Database, create_org, utcnow


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "audit.db")
    yield database
    database.close()


@pytest.fixture
def org(db):
    return create_org(db, "Org")


def _fill(db, org, n=5, start=None):
    start = start or utcnow()
    actor = {"user_id": "u1"}
    return [
        audit.append(
            db,
            org,
            actor,
            "policy_change",
            f"policy-{i}",
            {"alpha": i / 10},
            now=start + timedelta(minutes=i),
        )
        for i in range(n)
    ]


def test_append_links_rows_into_a_chain(db, org):
    rows = _fill(db, org, 3)
    assert rows[0]["prev_hash"] == audit.GENESIS
    assert rows[1]["prev_hash"] == rows[0]["hash"]
    assert rows[2]["prev_hash"] == rows[1]["hash"]
    assert audit.verify_chain(db, org) == audit.ChainCheck(True, 3)


def _drop_append_only_triggers(db):
    """Simula a un atacante con acceso al archivo: quita los triggers antes de alterar filas."""
    db.execute("DROP TRIGGER audit_log_no_update")
    db.execute("DROP TRIGGER audit_log_no_delete")


def test_at_p09_triggers_block_update_and_delete(db, org):
    """AT-P09: la base rechaza UPDATE y DELETE sobre audit_log."""
    import sqlite3

    import pytest

    rows = _fill(db, org)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE audit_log SET detail_json = '{}' WHERE seq = ?", (rows[0]["seq"],))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM audit_log WHERE seq = ?", (rows[0]["seq"],))


def test_at_p09_altering_a_row_breaks_verification(db, org):
    """AT-P09: modificar el detalle de una fila rompe la verificación en esa fila."""
    rows = _fill(db, org)
    _drop_append_only_triggers(db)
    db.execute("UPDATE audit_log SET detail_json = ? WHERE seq = ?", ('{"alpha":0.99}', rows[2]["seq"]))
    check = audit.verify_chain(db, org)
    assert not check.ok and check.broken_at == rows[2]["seq"] and check.checked == 2


def test_at_p09_recomputing_the_hash_of_an_altered_row_breaks_the_next_link(db, org):
    """AT-P09: si el atacante recalcula el hash de la fila alterada, se rompe el enlace siguiente."""
    rows = _fill(db, org)
    _drop_append_only_triggers(db)
    tampered = dict(db.query_one("SELECT * FROM audit_log WHERE seq = ?", (rows[1]["seq"],)))
    tampered["actor_user_id"] = "otro"
    db.execute(
        "UPDATE audit_log SET actor_user_id = ?, hash = ? WHERE seq = ?",
        ("otro", audit.row_hash(tampered), rows[1]["seq"]),
    )
    check = audit.verify_chain(db, org)
    assert not check.ok and check.broken_at == rows[2]["seq"]


def test_at_p09_deleting_a_middle_row_breaks_verification(db, org):
    """AT-P09: borrar una fila intermedia rompe la cadena."""
    rows = _fill(db, org)
    _drop_append_only_triggers(db)
    db.execute("DELETE FROM audit_log WHERE seq = ?", (rows[2]["seq"],))
    check = audit.verify_chain(db, org)
    assert not check.ok and check.broken_at == rows[3]["seq"]


def test_chains_are_independent_per_org(db, org):
    other = create_org(db, "Otra")
    _fill(db, org, 2)
    first_other = audit.append(db, other, None, "login", "u2")
    _fill(db, org, 2)
    assert first_other["prev_hash"] == audit.GENESIS
    assert audit.verify_chain(db, org).checked == 4
    assert audit.verify_chain(db, other).checked == 1
    system = audit.append(db, None, None, "startup")
    assert system["prev_hash"] == audit.GENESIS and audit.verify_chain(db, None).ok


def test_actor_from_principal_like_object(db, org):
    class P:
        user_id = None
        key_id = "kid123"

    row = audit.append(db, org, P(), "trace_ingest", "t1")
    assert row["actor_key_id"] == "kid123" and row["actor_user_id"] is None


def test_export_range_is_ordered_and_verifiable(db, org):
    start = utcnow()
    _fill(db, org, 6, start=start)
    rows = audit.export_range(db, org, start=start + timedelta(minutes=2), end=start + timedelta(minutes=5))
    assert [r["target"] for r in rows] == ["policy-2", "policy-3", "policy-4"]
    assert rows[0]["detail"] == {"alpha": 0.2}
    assert audit.verify_rows(rows).ok
    rows[1]["target"] = "alterado"
    assert not audit.verify_rows(rows).ok


def test_detail_rejects_secret_keys_and_bad_action(db, org):
    for detail in ({"password": "x"}, {"api_token": "x"}, {"meta": {"secret": "x"}}):
        with pytest.raises(ValueError):
            audit.append(db, org, None, "user_create", "u", detail)
    with pytest.raises(ValueError):
        audit.append(db, org, None, "")
    audit.append(db, org, None, "trace_ingest", "t", {"n_tokens": 3})
    assert audit.verify_chain(db, org).ok
