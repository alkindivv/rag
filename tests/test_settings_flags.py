from src.config.settings import settings


def test_p0_feature_flags_exist_and_defaults():
    # Ensure flags exist
    assert hasattr(settings, "NEW_PG_RETRIEVAL")
    assert hasattr(settings, "USE_SQL_FUSION")
    assert hasattr(settings, "USE_RERANKER")

    # Defaults should be False (safe rollout)
    assert settings.NEW_PG_RETRIEVAL is False
    assert settings.USE_SQL_FUSION is False
    assert settings.USE_RERANKER is False
