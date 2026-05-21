# reporting/report_builder.py

def build_report(explanation, data_quality, fairness):

    return {
        "summary": "AutoExplainML Report",
        "explanation": explanation,
        "data_quality": data_quality,
        "fairness": fairness
    }