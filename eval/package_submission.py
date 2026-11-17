import os
import shutil
import zipfile
import sys

def package():
    print("=== Creating FeedbackRAG Final Submission Package ===")
    dest_dir = "FeedbackRAG_Submission"
    zip_name = "FeedbackRAG_Submission.zip"

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    if os.path.exists(zip_name):
        os.remove(zip_name)

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "source"), exist_ok=True)

    # 1. Root documents & configs
    root_files = [
        "README.md",
        "REPORT.pdf",
        "REPORT.md",
        "requirements.txt",
        "pyproject.toml",
        ".env.example",
        "config.yaml",
        "AUDIT_REPORT.md"
    ]
    for rf in root_files:
        if os.path.exists(rf):
            shutil.copy2(rf, os.path.join(dest_dir, rf))
            print(f"  [+] Copied {rf}")
        else:
            print(f"  [!] Missing expected file: {rf}")

    # 2. Source directories into source/ AND root (for immediate execution in either path)
    source_dirs = ["src", "app", "api", "core", "retrieval", "feedback", "graph", "eval", "tests", "scripts"]
    for sd in source_dirs:
        if os.path.exists(sd):
            # Copy into source/
            shutil.copytree(
                sd,
                os.path.join(dest_dir, "source", sd),
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")
            )
            # Also copy to root of package so pytest and eval run directly without path confusion
            shutil.copytree(
                sd,
                os.path.join(dest_dir, sd),
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")
            )
            print(f"  [+] Copied module: {sd}")

    # 3. Data, Config, Docs, Results directories
    meta_dirs = ["data", "config", "docs", "results"]
    for md in meta_dirs:
        if os.path.exists(md):
            shutil.copytree(
                md,
                os.path.join(dest_dir, md),
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp", "*.log", "feedback_store.db", "*cache.json")
            )
            print(f"  [+] Copied metadata directory: {md}")

    # 4. Verify no secret or cache leakage
    leaks = []
    for root, dirs, files in os.walk(dest_dir):
        for f in files:
            if f.endswith(".pyc") or f == ".env" or f == "credentials.json" or f.endswith(".key") or f.endswith(".pem"):
                leaks.append(os.path.join(root, f))
    if leaks:
        print("[ERROR] Found leak candidates:", leaks)
        for lf in leaks:
            os.remove(lf)

    # 5. Create ZIP archive
    print(f"\nCompressing into {zip_name}...")
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dest_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=os.path.dirname(dest_dir))
                zipf.write(abs_path, rel_path)

    zip_size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"[SUCCESS] Submission ZIP created: {zip_name} ({zip_size_mb:.2f} MB)")

    # 6. Verify ZIP content summary
    with zipfile.ZipFile(zip_name, "r") as z:
        namelist = z.namelist()
        print(f"Total files in ZIP: {len(namelist)}")
        has_pdf = any("REPORT.pdf" in n for n in namelist)
        has_readme = any("README.md" in n for n in namelist)
        assert has_pdf, "Missing REPORT.pdf in zip!"
        assert has_readme, "Missing README.md in zip!"

    return 0

if __name__ == "__main__":
    sys.exit(package())

