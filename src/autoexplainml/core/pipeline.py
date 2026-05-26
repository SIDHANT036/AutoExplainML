from autoexplainml.core.engine import explain
from autoexplainml.core.validator import validate
from autoexplainml.intelligence.data_quality import check_data_quality
from autoexplainml.reporting.report_builder import build_report

def run_pipeline(model, X):
    validate(model, X)
    explanation = explain(model, X)
    dq = check_data_quality(X)
    report = build_report(explanation, dq, None)
    return report
