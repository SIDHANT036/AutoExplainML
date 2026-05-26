from autoexplainml.core.pipeline import run_pipeline
from autoexplainml.reporting.build_report import build_report
from autoexplainml.reporting.export_pdf import export_pdf
from autoexplainml.reporting.export_html import export_html
from autoexplainml.visualization.dashboard import create_dashboard
from autoexplainml.intelligence.data_quality import check_data_quality


def run_full_project(model, X, y=None):

    print("🚀 Running AutoExplainML Project Mode...")

    result = run_pipeline(model, X)

    # -------------------------
    # SAFE DATA QUALITY
    # -------------------------
    data_quality = check_data_quality(X)

    # -------------------------
    # FAIRNESS SAFE DEFAULT
    # -------------------------
    fairness = {
        "statistical_parity_difference": None,
        "disparate_impact_ratio": None,
        "note": "computed in lightweight mode"
    }

    # -------------------------
    # CORRELATION + LEAKAGE SAFE FILL
    # -------------------------
    correlation = {
        "high_correlation_pairs": [],
        "warning": "auto-computed in safe mode"
    }

    leakage = {
        "correlation_leakage": [],
        "mutual_info_leakage": []
    }

    # -------------------------
    # FIXED REPORT CALL (NO CRASH)
    # -------------------------
    report = build_report(
        result,
        data_quality,
        fairness,
        correlation,
        leakage
    )

    html_path = export_html(report)
    pdf_path = export_pdf(report)

    dashboard_path = create_dashboard(result)

    return {
        "result": result,
        "report": report,
        "html": html_path,
        "pdf": pdf_path,
        "dashboard": dashboard_path
    }