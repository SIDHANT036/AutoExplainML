def export_html(report: dict, output_path: str = "report.html"):
    html = f"""
    <html>
    <body>
        <h1>AutoExplainML Report</h1>
        <pre>{report}</pre>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html)

    return output_path