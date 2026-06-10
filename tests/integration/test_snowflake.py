"""Tests d'intégration — Connexion Snowflake."""

import os
import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("SNOWFLAKE_ACCOUNT"), reason="Variables Snowflake non configurées"
)
def test_snowflake_connection():
    """Connexion Snowflake établie avec succès."""
    pytest.importorskip("snowflake.connector")
    from pipeline.snowflake_io import _get_connection

    conn = _get_connection()
    assert conn is not None
    conn.close()
