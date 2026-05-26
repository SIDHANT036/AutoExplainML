# reporting/export_pdf.py

def export_pdf(report, filename="report.pdf"):
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF export. Install it with: pip install reportlab"
        )

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    content = [Paragraph(str(report), styles["Normal"])]
    doc.build(content)
    return filename
