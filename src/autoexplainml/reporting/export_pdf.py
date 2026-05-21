# reporting/export_pdf.py

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def export_pdf(report, filename="report.pdf"):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = [Paragraph(str(report), styles["Normal"])]

    doc.build(content)

    return filename