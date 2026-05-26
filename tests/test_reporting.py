import pytest
import os
import tempfile
from autoexplainml.reporting.report_builder import build_report


# ── build_report ───────────────────────────────────────────────────────────────

def test_build_report_returns_dict(trained_rf):
    model, X = trained_rf
    from autoexplainml.core.engine import explain
    from autoexplainml.intelligence.data_quality import check_data_quality
    explanation = explain(model, X)
    dq = check_data_quality(X)
    result = build_report(explanation, dq, None)
    assert isinstance(result, dict)


def test_build_report_not_empty(trained_rf):
    model, X = trained_rf
    from autoexplainml.core.engine import explain
    from autoexplainml.intelligence.data_quality import check_data_quality
    explanation = explain(model, X)
    dq = check_data_quality(X)
    result = build_report(explanation, dq, None)
    assert len(result) > 0


def test_build_report_accepts_none_metadata(trained_rf):
    model, X = trained_rf
    from autoexplainml.core.engine import explain
    from autoexplainml.intelligence.data_quality import check_data_quality
    explanation = explain(model, X)
    dq = check_data_quality(X)
    result = build_report(explanation, dq, None)
    assert result is not None


# ── html export ────────────────────────────────────────────────────────────────

try:
    from autoexplainml.reporting.export_html import export_html
    HAS_HTML = True
except ImportError:
    HAS_HTML = False


@pytest.mark.skipif(not HAS_HTML, reason="export_html not importable")
def test_export_html_creates_file(trained_rf):
    model, X = trained_rf
    from autoexplainml.core.pipeline import run_pipeline
    report = run_pipeline(model, X)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        export_html(report, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


@pytest.mark.skipif(not HAS_HTML, reason="export_html not importable")
def test_export_html_content_is_valid(trained_rf):
    model, X = trained_rf
    from autoexplainml.core.pipeline import run_pipeline
    report = run_pipeline(model, X)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        export_html(report, path)
        with open(path) as f:
            content = f.read()
        assert "<html" in content.lower() or "<!doctype" in content.lower()


# ── pdf export ─────────────────────────────────────────────────────────────────

try:
    from autoexplainml.reporting.export_pdf import export_pdf
    HAS_PDF = True
except ImportError:
    HAS_PDF = False


@pytest.mark.skipif(not HAS_PDF, reason="export_pdf not importable")
def test_export_pdf_creates_file(trained_rf):
    model, X = trained_rf
    from autoexplainml.core.pipeline import run_pipeline
    report = run_pipeline(model, X)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.pdf")
        export_pdf(report, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
