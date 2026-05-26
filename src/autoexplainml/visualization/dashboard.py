# visualization/dashboard.py

def create_dashboard(result, filename="dashboard.html"):
    """
    Generate a lightweight HTML dashboard from a pipeline result dict.
    No external JS dependencies — pure HTML/CSS.
    """

    def _render(value, depth=0):
        if isinstance(value, dict):
            rows = "".join(
                f"<tr><td style='padding:4px 10px;font-weight:bold;white-space:nowrap'>{k}</td>"
                f"<td style='padding:4px 10px'>{_render(v, depth+1)}</td></tr>"
                for k, v in value.items()
            )
            return f"<table border='1' style='border-collapse:collapse;font-size:0.9em'>{rows}</table>"
        elif isinstance(value, list):
            if all(isinstance(x, (int, float)) for x in value):
                bars = ""
                if value:
                    mx = max(abs(x) for x in value) or 1
                    for x in value:
                        pct = int(abs(x) / mx * 100)
                        bars += (
                            f"<div style='display:flex;align-items:center;margin:2px 0'>"
                            f"<div style='width:{pct}%;background:#3498db;height:14px;border-radius:3px'></div>"
                            f"<span style='margin-left:6px;font-size:0.85em'>{x:.4f}</span></div>"
                        )
                return bars or "[]"
            return "<ul style='margin:0;padding-left:18px'>" + "".join(
                f"<li>{_render(i, depth+1)}</li>" for i in value
            ) + "</ul>"
        else:
            return f"<span>{value}</span>"

    feature_section = ""
    if "features" in result and "importance" in result:
        pairs = sorted(
            zip(result["features"], result["importance"]),
            key=lambda x: x[1], reverse=True
        )
        mx = pairs[0][1] if pairs else 1
        rows = ""
        for feat, imp in pairs:
            pct = int(imp / mx * 100) if mx else 0
            rows += (
                f"<tr><td style='padding:4px 12px'>{feat}</td>"
                f"<td style='padding:4px 12px;width:60%'>"
                f"<div style='width:{pct}%;background:#2ecc71;height:16px;border-radius:3px'></div></td>"
                f"<td style='padding:4px 12px'>{imp:.4f}</td></tr>"
            )
        feature_section = f"""
        <h2 style='color:#2c3e50'>📊 Feature Importance</h2>
        <table border='1' style='border-collapse:collapse;width:100%;font-size:0.9em'>
          <tr style='background:#ecf0f1'><th>Feature</th><th>Importance</th><th>Value</th></tr>
          {rows}
        </table>"""

    full_data = _render(result)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AutoExplainML Dashboard</title>
  <style>
    body   {{ font-family: Arial, sans-serif; margin: 40px; background: #f9f9f9; color: #222; }}
    h1     {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
    h2     {{ color: #2c3e50; margin-top: 32px; }}
    table  {{ background: #fff; }}
    td, th {{ vertical-align: top; }}
  </style>
</head>
<body>
  <h1>🧠 AutoExplainML Dashboard</h1>
  {feature_section}
  <h2>📋 Full Result</h2>
  {full_data}
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    return filename
