# src/generate_report.py
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import json
import os

print("📝 Report generation started...")

# 1. Setup PDF file
report_path = "reports/final_report.pdf"
doc = SimpleDocTemplate(report_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

# 2. Title
story.append(Paragraph("<b>Heart Disease ML Project Report</b>", styles["Title"]))
story.append(Spacer(1, 20))

# 3. Dataset Description
story.append(Paragraph("📊 <b>Dataset:</b>", styles["Heading2"]))
story.append(Paragraph(
    "The dataset contains heart disease patient records. "
    "After preprocessing and cleaning, we applied PCA to reduce dimensions "
    "from 34 features to 15 principal components.", styles["Normal"]))
story.append(Spacer(1, 12))

# 4. Steps
story.append(Paragraph("⚙️ <b>Project Steps:</b>", styles["Heading2"]))
steps_text = """
1. Data Preprocessing (handling missing values, cleaning).
2. Dimensionality Reduction using PCA.
3. Training multiple models (Logistic Regression, Random Forest, XGBoost).
4. Model selection and saving the best model.
5. Evaluation on test data (classification report, accuracy, confusion matrix).
6. Visualization of results.
"""
story.append(Paragraph(steps_text.replace("\n", "<br/>"), styles["Normal"]))
story.append(Spacer(1, 12))

# 5. Results
story.append(Paragraph("🏆 <b>Model Results:</b>", styles["Heading2"]))
with open("reports/model_results.json", "r") as f:
    results = json.load(f)

best_model = max(results, key=results.get)
best_acc = results[best_model]

results_text = "<br/>".join([f"{model}: {acc:.4f}" for model, acc in results.items()])
story.append(Paragraph(results_text, styles["Normal"]))
story.append(Spacer(1, 12))

story.append(Paragraph(
    f"<b>Best Model:</b> {best_model} with accuracy {best_acc:.4f}", styles["Normal"]))
story.append(Spacer(1, 20))

# 6. Add visuals if exist
visuals = [
    "reports/confusion_matrix.png",
    "reports/model_accuracies.png",
    "reports/roc_curve.png"
]

for vis in visuals:
    if os.path.exists(vis):
        story.append(Image(vis, width=400, height=300))
        story.append(Spacer(1, 20))

# 7. Build PDF
doc.build(story)
print(f"✅ Report saved successfully at {report_path}")