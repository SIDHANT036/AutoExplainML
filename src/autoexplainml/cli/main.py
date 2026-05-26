import argparse
import joblib
import pandas as pd
import json
import os
from pathlib import Path

from autoexplainml.core.pipeline import run_pipeline
from autoexplainml.product.project_mode import run_full_project
from autoexplainml.reporting.export_html import export_html
from autoexplainml.reporting.export_pdf import export_pdf


def main():

    parser = argparse.ArgumentParser(
        description="AutoExplainML CLI - ML Explainability + Project Generator"
    )

    parser.add_argument("model", help="Path to trained model file (.pkl)")
    parser.add_argument("data", help="Path to dataset file (.csv)")

    parser.add_argument(
        "--mode",
        choices=["analyze", "project"],
        default="analyze",
        help="analyze = ML pipeline | project = full student project generation"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    try:
        # =========================
        # VALIDATION (FIXED ISSUE YOU HAD)
        # =========================
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return

        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            return

        # =========================
        # LOAD INPUTS
        # =========================
        model = joblib.load(model_path)
        X = pd.read_csv(data_path, low_memory=False)

        print("\n🚀 AutoExplainML Running...")
        print(f"Mode: {args.mode}\n")

        # =========================
        # MODE 1: ANALYZE
        # =========================
        if args.mode == "analyze":

            result = run_pipeline(model, X)

            print("\n📊 ANALYSIS RESULT:\n")
            print(json.dumps(result, indent=2, default=str))

        # =========================
        # MODE 2: PROJECT
        # =========================
        elif args.mode == "project":

            result = run_full_project(model, X)

            # =========================
            # OUTPUT DIRECTORY
            # =========================
            output_dir = "autoexplainml_outputs"
            os.makedirs(output_dir, exist_ok=True)

            # =========================
            # SAVE JSON
            # =========================
            json_path = os.path.join(output_dir, "result.json")
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            # =========================
            # EXPORT REPORTS (SAFE)
            # =========================
            try:
                html_path = export_html(result)
            except Exception as e:
                print(f"⚠ HTML export failed: {e}")
                html_path = None

            try:
                pdf_path = export_pdf(result)
            except Exception as e:
                print(f"⚠ PDF export failed: {e}")
                pdf_path = None

            # =========================
            # FINAL OUTPUT
            # =========================
            print("\n📦 PROJECT OUTPUT GENERATED SUCCESSFULLY\n")
            print(f"📁 JSON File  : {json_path}")
            print(f"🌐 HTML Report: {html_path}")
            print(f"📄 PDF Report : {pdf_path}")

    except FileNotFoundError:
        print("❌ Error: File not found. Check model/data path.")

    except Exception as e:
        print("❌ Unexpected error occurred:")
        print(str(e))


if __name__ == "__main__":
    main()