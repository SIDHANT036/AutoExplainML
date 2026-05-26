from autoexplainml.visualization.shap_viz import shap_bar_chart
from autoexplainml.visualization.waterfall import waterfall_plot


def build_html_dashboard(report):

    explanation = report["explanation"]

    shap_html = shap_bar_chart(
        explanation["features"],
        explanation["importance"]
    )

    # fake waterfall fallback (first sample)
    waterfall_html = waterfall_plot(
        explanation["features"],
        explanation["importance"]
    )

    html = f"""
    <html>
    <head>
        <title>AutoExplainML Report</title>
    </head>
    <body>

        <h1>AutoExplainML Dashboard</h1>

        <h2>SHAP Feature Importance</h2>
        {shap_html}

        <h2>Prediction Explanation</h2>
        {waterfall_html}

        <h2>Data Quality</h2>
        <pre>{report['data_quality']}</pre>

        <h2>Fairness</h2>
        <pre>{report['fairness']}</pre>

        <h2>Leakage</h2>
        <pre>{report['leakage']}</pre>

    </body>
    </html>
    """

    return html