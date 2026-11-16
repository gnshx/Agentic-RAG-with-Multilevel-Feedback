import os
import sys
import subprocess
from markdown_it import MarkdownIt

def build_pdf():
    print("=== Generating REPORT.pdf from REPORT.md ===")
    md_path = "REPORT.md"
    assert os.path.exists(md_path), f"Missing {md_path}"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    md = MarkdownIt("commonmark", {"breaks": True, "html": True})
    html_content = md.render(md_text)

    # Insert Pareto frontier plot if available and not embedded
    if os.path.exists("results/plots/pareto_frontier.png"):
        img_abs = os.path.abspath("results/plots/pareto_frontier.png")
        plot_html = f'''
        <div style="text-align: center; margin: 25px 0;">
            <img src="file://{img_abs}" style="max-width: 90%; height: auto; border: 1px solid #e2e8f0; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);" alt="Pareto Frontier" />
            <p style="font-size: 9pt; color: #64748b; margin-top: 8px;"><b>Figure 1:</b> Cost-Accuracy Pareto Frontier comparing FeedbackRAG against static baselines with 95% bootstrap confidence intervals.</p>
        </div>
        '''
        if "pareto_frontier.png" not in html_content:
            # Insert before Section 5
            html_content = html_content.replace("<h2>5. Feedback Signal Ablation Study</h2>", plot_html + "\n<h2>5. Feedback Signal Ablation Study</h2>")

    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FeedbackRAG Report</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {{delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}}
            ]}});"></script>
    <style>
        @page {{
            size: A4;
            margin: 20mm 18mm 20mm 18mm;
            @bottom-right {{
                content: counter(page);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 8pt;
                color: #64748b;
            }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.55;
            color: #1e293b;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            font-size: 19pt;
            font-weight: 700;
            color: #0f172a;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 6px;
            margin-top: 0;
            margin-bottom: 12px;
            line-height: 1.25;
        }}
        h2 {{
            font-size: 13.5pt;
            font-weight: 600;
            color: #1e3a8a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
            margin-top: 22px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 11pt;
            font-weight: 600;
            color: #2563eb;
            margin-top: 14px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        p, li {{
            text-align: justify;
            margin-top: 4px;
            margin-bottom: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 14px 0;
            font-size: 8.5pt;
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 5px 8px;
            text-align: left;
        }}
        th {{
            background-color: #f1f5f9;
            color: #0f172a;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        code {{
            font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
            font-size: 8.5pt;
            background-color: #f1f5f9;
            padding: 1px 4px;
            border-radius: 3px;
            color: #0f172a;
        }}
        pre {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 5px;
            padding: 10px;
            font-size: 8pt;
            overflow-x: auto;
            page-break-inside: avoid;
            margin: 10px 0;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        blockquote {{
            margin: 10px 0;
            padding: 6px 14px;
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            color: #1e3a8a;
            font-style: italic;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 18px 0;
        }}
        .author-box {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 18px;
        }}
        .author-box p {{
            margin: 2px 0;
            font-size: 9.5pt;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>'''

    temp_html = "report_temp.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_template)

    pdf_out = "REPORT.pdf"
    cmd = [
        "google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_out}",
        temp_html
    ]

    print("Running command:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(pdf_out):
        size_kb = os.path.getsize(pdf_out) / 1024
        print(f"[SUCCESS] {pdf_out} generated successfully ({size_kb:.1f} KB).")
        if os.path.exists(temp_html):
            os.remove(temp_html)
        return 0
    else:
        print("[ERROR] Chrome PDF generation failed:")
        print("stdout:", res.stdout)
        print("stderr:", res.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(build_pdf())
