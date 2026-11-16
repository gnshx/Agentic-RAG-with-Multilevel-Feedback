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

    # Convert all escaped currency dollar signs (\$0.001) to HTML entity &#36; BEFORE markdown parsing
    md_text = md_text.replace(r'\$', '&#36;')

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

    # Replace markdown image syntax for architecture diagram with our clean, responsive SVG
    md_text = re.sub(
        r'!\[FeedbackRAG MLFQ Architecture\]\(.*?\)(?:\s*\*Figure 1:[^\*]+\*)?',
        '<!-- ARCHITECTURE_SVG_PLACEHOLDER -->',
        md_text,
        flags=re.DOTALL
    )

    md = MarkdownIt('commonmark', {'breaks': True, 'html': True}).enable('table')
    html_content = md.render(md_text)

    # Insert the SVG diagram at placeholder
    html_content = html_content.replace("<!-- ARCHITECTURE_SVG_PLACEHOLDER -->", svg_html_block)

    # Convert currency dollar signs to katex-ignored elements so KaTeX won't misinterpret them as math delimiters
    html_content = re.sub(r'(?:&amp;#36;|&#36;|\\\$|\$)(?=\d)', '<span class="katex-ignore">&#36;</span>', html_content)
    html_content = html_content.replace(r'\$', '<span class="katex-ignore">&#36;</span>')

    # Convert relative plot image sources to absolute file:// URLs
    abs_plots_dir = os.path.abspath("results/plots")
    html_content = html_content.replace('src="results/plots/', f'src="file://{abs_plots_dir}/')

    # Natural page flow with zero empty double pages
    sections_to_break = []
    for target in sections_to_break:
        if target in html_content:
            tag_name = "h3" if target.startswith("<h3>") else "h2"
            target_with_class = target.replace(f"<{tag_name}>", f'<{tag_name} class="break-page">')
            html_content = html_content.replace(target, target_with_class)

    # Keep Section 8.4 Failure Taxonomy block intact without splitting across pages
    html_content = html_content.replace(
        "<h3>8.4 Failure Taxonomy Breakdown (30 Case Studies)</h3>",
        '<div style="break-inside: avoid; page-break-inside: avoid;"><h3>8.4 Failure Taxonomy Breakdown (30 Case Studies)</h3>'
    )
    html_content = html_content.replace(
        "<h2>9. Production Engineering &amp; Architectural Guarantees</h2>",
        '</div>\n<h2>9. Production Engineering &amp; Architectural Guarantees</h2>'
    )

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
                ignoredClasses: ['katex-ignore']
            }});"></script>
    <style>
        @page {{
            size: A4 portrait;
            margin: 12mm 12mm 14mm 12mm;
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
            font-size: 8.8pt;
            line-height: 1.42;
            color: #1e293b;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            font-size: 16.5pt;
            font-weight: 700;
            color: #0f172a;
            border-bottom: 2.5px solid #2563eb;
            padding-bottom: 4px;
            margin-top: 0;
            margin-bottom: 8px;
            line-height: 1.2;
            letter-spacing: -0.3px;
        }}
        h2 {{
            font-size: 11.5pt;
            font-weight: 600;
            color: #1e3a8a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 2px;
            margin-top: 14px;
            margin-bottom: 6px;
            page-break-after: avoid;
            break-after: avoid;
        }}
        h2.break-page, h3.break-page {{
            break-before: page !important;
            page-break-before: always !important;
            margin-top: 0;
        }}
        h3 {{
            font-size: 9.8pt;
            font-weight: 600;
            color: #2563eb;
            margin-top: 9px;
            margin-bottom: 3px;
            page-break-after: avoid;
            break-after: avoid;
        }}
        h4 {{
            font-size: 9pt;
            font-weight: 600;
            color: #334155;
            margin-top: 7px;
            margin-bottom: 2px;
            page-break-after: avoid;
            break-after: avoid;
        }}
        p, li {{
            text-align: justify;
            margin-top: 2px;
            margin-bottom: 3px;
        }}
        ul, ol {{
            margin-top: 2px;
            margin-bottom: 4px;
            padding-left: 16px;
        }}
        li {{
            margin-bottom: 2px;
        }}
        table {{
            width: 100% !important;
            max-width: 100% !important;
            border-collapse: collapse;
            margin: 6px 0;
            font-size: 7.5pt;
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
            padding: 3px 4px;
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
            font-size: 7.5pt;
            background-color: #f1f5f9;
            padding: 1px 3px;
            border-radius: 3px;
            color: #0f172a;
        }}
        pre {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            padding: 5px 8px;
            font-size: 6.8pt;
            white-space: pre-wrap;
            word-break: break-word;
            page-break-inside: avoid;
            break-inside: avoid;
            margin: 5px 0;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            white-space: pre-wrap;
        }}
        blockquote {{
            margin: 5px 0;
            padding: 3px 8px;
            background-color: #eff6ff;
            border-left: 3px solid #3b82f6;
            color: #1e3a8a;
            font-style: italic;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 10px 0;
        }}
        .architecture-figure {{
            width: 100%;
            margin: 6px 0;
            text-align: center;
            page-break-inside: avoid !important;
            break-inside: avoid !important;
        }}
        .architecture-figure svg {{
            width: 85% !important;
            height: auto !important;
            max-height: 310px;
            display: block;
            margin: 0 auto;
        }}
        .figure-container {{
            width: 100%;
            text-align: center;
            margin: 6px 0;
            page-break-inside: avoid !important;
            break-inside: avoid !important;
        }}
        .pareto-img {{
            max-width: 76%;
            max-height: 220px;
            width: auto;
            height: auto;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .caption {{
            font-size: 7.5pt;
            color: #475569;
            margin-top: 3px;
            text-align: center;
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
        # if os.path.exists(temp_html):
        #     os.remove(temp_html)
        return 0
    else:
        print("[ERROR] Chrome PDF generation failed:")
        print("stdout:", res.stdout)
        print("stderr:", res.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(build_pdf())
