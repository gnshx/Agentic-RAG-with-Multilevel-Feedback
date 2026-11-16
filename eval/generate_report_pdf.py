import os
import sys
import subprocess
import re
from markdown_it import MarkdownIt

def build_pdf():
    print("=== Generating Clean, Publication-Grade 10-Page REPORT.pdf ===")
    md_path = "REPORT.md"
    assert os.path.exists(md_path), f"Missing {md_path}"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Step 1: Protect all currency dollar signs ($0.000149, \$0.000149, etc.)
    # We replace currency with a dedicated unique token so KaTeX will not parse them as math delimiters.
    md_text = md_text.replace(r'\$', 'XXCURRENCYSIGNXX')
    md_text = re.sub(r'(\|\s*)-?\$(\d)', r'\1XXCURRENCYSIGNXX\2', md_text)
    md_text = md_text.replace('($)', '(XXCURRENCYSIGNXX)')
    md_text = md_text.replace(r'-\$', '-XXCURRENCYSIGNXX')

    # Step 2: Extract display math $$...$$ to protect from markdown-it formatting
    display_math_blocks = []
    def save_display_math(match):
        idx = len(display_math_blocks)
        display_math_blocks.append(match.group(1).strip())
        return f"\n\nXXDISPLAYMATH{idx}XX\n\n"

    md_text = re.sub(r'\$\$(.*?)\$\$', save_display_math, md_text, flags=re.DOTALL)

    # Step 3: Extract inline math $...$ to protect from markdown-it underscore emphasis
    inline_math_blocks = []
    def save_inline_math(match):
        idx = len(inline_math_blocks)
        inline_math_blocks.append(match.group(1).strip())
        return f"XXINLINEMATH{idx}XX"

    md_text = re.sub(r'\$([^\$\n]+?)\$', save_inline_math, md_text)

    # Step 4: Replace architecture image syntax with clean responsive image container
    abs_svg_path = os.path.abspath("results/plots/mlfq_architecture.svg")
    assert os.path.exists(abs_svg_path), f"Missing {abs_svg_path}"

    architecture_html = f'''
    <div class="architecture-figure">
        <img src="file://{abs_svg_path}" class="architecture-img" alt="FeedbackRAG MLFQ Architecture Diagram" />
        <p class="caption"><b>Figure 1:</b> FeedbackRAG Multilevel Feedback Queue (MLFQ) operational state machine. Every query enters at Level 0 and escalates only when verified runtime signals ($S_1 \\dots S_4$) indicate deficiency.</p>
    </div>
    '''
    md_text = re.sub(
        r'!\[FeedbackRAG MLFQ Architecture\]\(.*?\)(?:\s*\*Figure 1:[^\*]+\*)?',
        '<!-- ARCHITECTURE_FIGURE_PLACEHOLDER -->',
        md_text,
        flags=re.DOTALL
    )

    # Step 5: Render Markdown to HTML via MarkdownIt
    md = MarkdownIt('commonmark', {'breaks': True, 'html': True}).enable('table')
    html_content = md.render(md_text)

    # Step 6: Restore architecture figure placeholder
    html_content = html_content.replace('<!-- ARCHITECTURE_FIGURE_PLACEHOLDER -->', architecture_html)

    # Step 7: Restore display math and inline math
    for idx, math_content in enumerate(display_math_blocks):
        html_content = html_content.replace(f'XXDISPLAYMATH{idx}XX', f'$${math_content}$$')
    for idx, math_content in enumerate(inline_math_blocks):
        html_content = html_content.replace(f'XXINLINEMATH{idx}XX', f'${math_content}$')

    # Step 8: Restore currency tokens with safe HTML spans
    html_content = html_content.replace('XXCURRENCYSIGNXX', '<span class="currency">&#36;</span>')

    # Step 9: Convert relative plot image sources to absolute file:// URLs
    abs_plots_dir = os.path.abspath("results/plots")
    html_content = html_content.replace('src="results/plots/', f'src="file://{abs_plots_dir}/')

    # Step 10: Precise section page-breaks for a perfectly proportioned 10-page academic report
    page_break_targets = [
        "<h2>2. System Architecture: The MLFQ Paradigm</h2>",
        "<h2>3. Detailed Specifications of the 4 Processing Tiers</h2>",
        "<h2>5. Learned Escalation Policy &amp; Budget Dial Optimization</h2>",
        "<h3>5.3 Risk-Controlled Escalation via Split Conformal Prediction</h3>",
        "<h3>6.2 Regime B: Live Neural Model Validation",
        "<h2>7. Feedback Signal Ablation Study &amp; Cost-Saving Heuristics</h2>",
        "<h2>8. Safety, Reliability, Adversarial Probes &amp; Human Validation</h2>",
        "<h3>8.3 Adversarial Attack Probing",
        "<h2>10. Limitations &amp; Scientific Scope</h2>",
    ]
    for target in page_break_targets:
        for heading in ["<h2>", "<h3>"]:
            if target.startswith(heading) and target in html_content:
                repl = target.replace(heading, f'{heading[:-1]} class="break-page">')
                html_content = html_content.replace(target, repl)
                break

    # Step 11: Build full HTML document with KaTeX and print-perfect CSS
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FeedbackRAG Research Report</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {{
                delimiters: [
                    {{left: '$$', right: '$$', display: true}},
                    {{left: '$', right: '$', display: false}}
                ],
                ignoredClasses: ['katex-ignore', 'currency']
            }});"></script>
    <style>
        @page {{
            size: A4 portrait;
            margin: 8.5mm 9.5mm 9.5mm 9.5mm;
            @bottom-right {{
                content: "Page " counter(page);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 7.8pt;
                color: #64748b;
            }}
            @bottom-left {{
                content: "FeedbackRAG: Cost-Optimal Agentic RAG | IIT Bhilai";
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 7.8pt;
                color: #94a3b8;
            }}
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 8.2pt;
            line-height: 1.30;
            color: #1e293b;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            font-size: 15pt;
            font-weight: 700;
            color: #0f172a;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 3px;
            margin-top: 0;
            margin-bottom: 5px;
            line-height: 1.2;
            letter-spacing: -0.3px;
        }}
        h2 {{
            font-size: 10.8pt;
            font-weight: 600;
            color: #1e3a8a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 2px;
            margin-top: 8px;
            margin-bottom: 4px;
            page-break-after: avoid;
            break-after: avoid;
        }}
        h2.break-page, h3.break-page {{
            break-before: page !important;
            page-break-before: always !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }}
        h3 {{
            font-size: 9.3pt;
            font-weight: 600;
            color: #2563eb;
            margin-top: 6px;
            margin-bottom: 2px;
            page-break-after: avoid;
            break-after: avoid;
        }}
        h4 {{
            font-size: 8.6pt;
            font-weight: 600;
            color: #334155;
            margin-top: 4px;
            margin-bottom: 2px;
            page-break-after: avoid;
            break-after: avoid;
        }}
        p, li {{
            text-align: justify;
            margin-top: 1px;
            margin-bottom: 1.5px;
        }}
        ul, ol {{
            margin-top: 1px;
            margin-bottom: 1.5px;
            padding-left: 14px;
        }}
        li {{
            margin-bottom: 1px;
        }}
        table {{
            width: 100% !important;
            max-width: 100% !important;
            border-collapse: collapse;
            margin: 4px 0;
            font-size: 7.2pt;
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
            padding: 2.2px 4px;
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
            font-size: 7.1pt;
            background-color: #f1f5f9;
            padding: 1px 3px;
            border-radius: 3px;
            color: #0f172a;
        }}
        pre {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            padding: 3px 5px;
            font-size: 6.8pt;
            white-space: pre-wrap;
            word-break: break-word;
            page-break-inside: avoid;
            break-inside: avoid;
            margin: 3px 0;
        }}
        blockquote {{
            margin: 3px 0;
            padding: 2px 6px;
            background-color: #eff6ff;
            border-left: 3px solid #3b82f6;
            color: #1e3a8a;
            font-size: 7.8pt;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 5px 0;
        }}
        .architecture-figure {{
            width: 100%;
            margin: 6px 0;
            text-align: center;
            page-break-inside: avoid !important;
            break-inside: avoid !important;
        }}
        .architecture-img {{
            width: 98% !important;
            max-width: 680px;
            height: auto !important;
            max-height: 480px;
            display: block;
            margin: 0 auto;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
        }}
        .figure-container {{
            width: 100%;
            text-align: center;
            margin: 4px 0;
            page-break-inside: avoid !important;
            break-inside: avoid !important;
        }}
        .pareto-img {{
            max-width: 70%;
            max-height: 190px;
            width: auto;
            height: auto;
            border: 1px solid #e2e8f0;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .caption {{
            font-size: 7pt;
            color: #475569;
            margin-top: 2px;
            text-align: center;
        }}
        .protocol-card {{
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            border-left: 3.5px solid #2563eb;
            border-radius: 4px;
            padding: 3px 6px;
            margin: 4px 0;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        .protocol-title {{
            font-size: 7.3pt;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 1px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .protocol-body {{
            font-size: 6.9pt;
            color: #334155;
        }}
        .katex-display {{
            margin: 2px 0 !important;
        }}
        .currency {{
            font-weight: normal;
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
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--virtual-time-budget=2000",
        f"--print-to-pdf={pdf_out}",
        temp_html
    ]

    print("Running command:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(pdf_out):
        size_kb = os.path.getsize(pdf_out) / 1024
        print(f"[SUCCESS] Clean {pdf_out} generated successfully ({size_kb:.1f} KB).")
        return 0
    else:
        print("[ERROR] Chrome PDF generation failed:")
        print("stdout:", res.stdout)
        print("stderr:", res.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(build_pdf())
