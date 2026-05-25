from autoexplainml.core.pipeline import run_pipeline
from autoexplainml.reporting.report_builder import build_report
from autoexplainml.reporting.export_pdf import export_pdf
from autoexplainml.reporting.export_html import export_html
from autoexplainml.visualization.dashboard import create_dashboard

def run_full_project(model, X, y=None):
    """
    STUDENT PROJECT MODE:
    One function → full ML project output
    """

    print("🚀 Running AutoExplainML Project Mode...")

    # 1. Run core pipeline
    result = run_pipeline(model, X)

    # 2. Build report
    report = build_report(result)

    # 3. Export outputs
    html_path = export_html(report)
    pdf_path = export_pdf(report)

    # 4. Dashboard (optional)
    dashboard_path = create_dashboard(result)

    return {
        "result": result,
        "report": report,
        "html": html_path,
        "pdf": pdf_path,
        "dashboard": dashboard_path
    }