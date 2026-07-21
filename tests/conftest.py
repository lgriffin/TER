def pytest_collection_modifyitems(config, items):
    """Skip tests that require optional embeddings when the extra is absent."""
    import importlib.util
    import pytest

    if importlib.util.find_spec("sentence_transformers") is not None:
        return
    skip = pytest.mark.skip(reason="requires the embeddings optional dependency")
    for item in items:
        nodeid = item.nodeid
        if "test_input_analysis.py" in nodeid or (
            "test_cli.py::TestAnalyzeCommand" in nodeid
        ):
            item.add_marker(skip)
