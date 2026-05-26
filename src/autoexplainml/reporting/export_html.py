# reporting/export_html.py

def export_html(report, filename="report.html"):
    """
    Export a report dict to a self-contained HTML file.
    No external dependencies required.
    """
    def _rows(data, indent=0):
        html = ""
        prefix = "  " * indent
        if isinstance(data, dict):
            html += f"{prefix}<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;margin:4px 0'>\n"
            for k, v in data.items():
                html += f"{prefix}  <tr><td><strong>{k}</strong></td><td>{_rows(v, indent+2)}</td></tr>\n"
            html += f"{prefix}</table>\n"
        elif isinstance(data, list):
            html += "<ul>" + "".join(f"<li>{_rows(i)}</li>" for i in data) + "</ul>"
        else:
            html += str(data)
        return html

    body = _rows(report)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AutoExplainML Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
    h1   {{ color: #2c3e50; }}
    table {{ font-size: 0.9em; }}
    td   {{ vertical-align: top; padding: 6px 10px; }}
  </style>
</head>
<body>
  <h1>🧠 AutoExplainML Report</h1>
  {body}
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    return filename
