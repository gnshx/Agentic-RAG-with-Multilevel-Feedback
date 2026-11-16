import os
import sys
import subprocess
import re
from markdown_it import MarkdownIt

def build_pdf():
    print("=== Generating Clean, Publication-Grade REPORT.pdf ===")
    md_path = "REPORT.md"
    assert os.path.exists(md_path), f"Missing {md_path}"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Read the vector architecture SVG
    svg_path = "results/plots/mlfq_architecture.svg"
    assert os.path.exists(svg_path), f"Missing {svg_path}"
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    # Wrap the SVG in an elegant, responsive container
    svg_html_block = f'''
    <div class="architecture-figure">
        {svg_content}
        <p class="caption"><b>Figure 1:</b> FeedbackRAG Multilevel Feedback Queue (MLFQ) operational state machine. Every query enters at Level 0 and escalates only when verified runtime signals ($S_1 \\dots S_4$) indicate deficiency.</p>
    </div>
    '''

    # Replace both markdown image syntax and old ASCII box with our clean, responsive SVG
    md_text = re.sub(
        r'!\[FeedbackRAG MLFQ Architecture\]\(.*?\)(?:\s*\*Figure 1:[^\*]+\*)?',
        '<!-- ARCHITECTURE_SVG_PLACEHOLDER -->',
        md_text,
        flags=re.DOTALL
    )
    ascii_pattern = re.compile(r'```(?:\n|\r\n)\+[-+]+\+.*?\+[-+]+\+(?:\n|\r\n)```', re.DOTALL)
    if ascii_pattern.search(md_text):
        md_text = ascii_pattern.sub("<!-- ARCHITECTURE_SVG_PLACEHOLDER -->", md_text)
    md_text = re.sub(r'```.*?FeedbackRAG MLFQ Control Flow.*?```', '<!-- ARCHITECTURE_SVG_PLACEHOLDER -->', md_text, flags=re.DOTALL)

    md = MarkdownIt("commonmark", {"breaks": True, "html": True})
    html_content = md.render(md_text)

    # Insert the SVG diagram at placeholder
    html_content = html_content.replace("<!-- ARCHITECTURE_SVG_PLACEHOLDER -->", svg_html_block)

    # Embed Pareto frontier plot before Section 7
    if os.path.exists("results/plots/pareto_frontier.png"):
        img_abs = os.path.abspath("results/plots/pareto_frontier.png")
        plot_html = f'''
        <div class="figure-container">
            <img src="file://{img_abs}" class="pareto-img" alt="Cost-Accuracy Pareto Frontier" />
            <p class="caption"><b>Figure 2:</b> Empirical Cost-Accuracy Pareto Frontier comparing FeedbackRAG against static and agentic baselines with 95% bootstrap confidence intervals.</p>
        </div>
        '''
        if "pareto_frontier.png" not in html_content:
            html_content = html_content.replace("<h2>7. Feedback Signal Ablation Study</h2>", plot_html + "\n<h2>7. Feedback Signal Ablation Study</h2>")

    # Strategic Page Breaks for Flawless 10-Page Print Layout
    page_breaks = [
        "<h2>2. System Architecture: The MLFQ Paradigm</h2>",
        "<h2>3. Detailed Specifications of the 4 Processing Tiers</h2>",
        "<h2>4. The 4 Decoupled Feedback Verification Signals</h2>",
        "<h2>6. Comprehensive Empirical Evaluation</h2>",
        "<h3>6.2 Regime B: Live Neural Model Validation (Local GPU Inference)</h3>",
        "<h2>7. Feedback Signal Ablation Study</h2>",
        "<h2>9. Production Engineering &amp; Architectural Guarantees</h2>",
        "<h2>11. Conclusion &amp; Future Work</h2>"
    ]
    for target in page_breaks:
        if target in html_content:
            html_content = html_content.replace(target, f'<div class="page-break"></div>\n{target}')
        else:
            print(f"[PAGE BREAK WARNING] Target not found in html: {target}")

    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FeedbackRAG Research Report</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {{delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}}
            ]}});"></script>
    <style>
        @page {{
            size: A4 portrait;
            margin: 16mm 15mm 18mm 15mm;
            @bottom-right {{
                content: "Page " counter(page);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 8pt;
                color: #64748b;
            }}
            @bottom-left {{
                content: "FeedbackRAG: Cost-Optimal Agentic RAG | IIT Bhilai";
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 8pt;
                color: #94a3b8;
            }}
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 9.5pt;
            line-height: 1.5;
            color: #1e293b;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            font-size: 18pt;
            font-weight: 700;
            color: #0f172a;
            border-bottom: 2.5px solid #2563eb;
            padding-bottom: 6px;
            margin-top: 0;
            margin-bottom: 10px;
            line-height: 1.25;
            letter-spacing: -0.3px;
        }}
        h2 {{
            font-size: 12.5pt;
            font-weight: 600;
            color: #1e3a8a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 3px;
            margin-top: 20px;
            margin-bottom: 8px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 10.5pt;
            font-weight: 600;
            color: #2563eb;
            margin-top: 12px;
            margin-bottom: 5px;
            page-break-after: avoid;
        }}
        p, li {{
            text-align: justify;
            margin-top: 3px;
            margin-bottom: 5px;
        }}
        ul, ol {{
            margin-top: 3px;
            margin-bottom: 6px;
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 3px;
        }}
        .page-break {{
            page-break-before: always !important;
            break-before: page !important;
            height: 0;
            margin: 0;
            padding: 0;
        }}
        table {{
            width: 100% !important;
            max-width: 100% !important;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 8pt;
            page-break-inside: avoid !important;
            break-inside: avoid !important;
            table-layout: auto;
        }}
        tr {{
            page-break-inside: avoid !important;
            break-inside: avoid !important;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 4px 6px;
            text-align: left;
            vertical-align: middle;
            word-wrap: break-word;
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
            font-family: "JetBrains Mono", "SFMono-Regular", Consolas, "Courier New", monospace;
            font-size: 8pt;
            background-color: #f1f5f9;
            padding: 1px 3px;
            border-radius: 3px;
            color: #0f172a;
        }}
        pre {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 5px;
            padding: 8px 10px;
            font-size: 7.5pt;
            white-space: pre-wrap;
            word-break: break-word;
            page-break-inside: avoid;
            margin: 8px 0;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            white-space: pre-wrap;
        }}
        blockquote {{
            margin: 8px 0;
            padding: 5px 12px;
            background-color: #eff6ff;
            border-left: 3.5px solid #3b82f6;
            color: #1e3a8a;
            font-style: italic;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 14px 0;
        }}
        .architecture-figure {{
            width: 100%;
            margin: 10px 0;
            text-align: center;
            page-break-inside: avoid !important;
            break-inside: avoid !important;
        }}
        .architecture-figure svg {{
            width: 100% !important;
            height: auto !important;
            max-height: 480px;
            display: block;
            margin: 0 auto;
        }}
        .figure-container {{
            width: 100%;
            text-align: center;
            margin: 10px 0;
            page-break-inside: avoid !important;
            break-inside: avoid !important;
        }}
        .pareto-img {{
            max-width: 88%;
            height: auto;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .caption {{
            font-size: 8pt;
            color: #475569;
            margin-top: 5px;
            text-align: center;
        }}
        .author-box {{
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 12px;
            font-size: 9pt;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>'''

    temp_html = "report_clean.html"
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
        print(f"[SUCCESS] Clean {pdf_out} generated successfully ({size_kb:.1f} KB).")
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
