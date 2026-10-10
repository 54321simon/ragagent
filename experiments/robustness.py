"""Live cache/concurrency and bounded-input acceptance checks."""

import sys, os, json, time, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.react_loop import react_loop
from retriever.loader import load_any, load_pdf
from rag.cache import SemanticCache


def main():
    records = []
    os.environ["CACHE_ENABLED"] = "1"
    SemanticCache().clear()
    question = "MAE 的作者和年份是什么？"
    for name, q in [
        ("cache_cold", question),
        ("cache_exact", question),
        ("cache_semantic", "请问 MAE 的作者和年份是什么？"),
    ]:
        result = react_loop(q, doc_ids=["MAE"], session_id="cache-benchmark")
        records.append(
            dict(
                test=name,
                metrics=result["metrics"],
                success=result["success"],
                answer=result["answer"],
            )
        )
        print(name, result["metrics"], flush=True)
    os.environ["CACHE_ENABLED"] = "0"
    with ThreadPoolExecutor(max_workers=2) as ex:

        def user(task):
            doc, q, session = task
            res = react_loop(q, doc_ids=[doc], session_id=session)
            return dict(
                test="concurrent_" + doc,
                session=session,
                metrics=res["metrics"],
                answer=res["answer"],
                success=res["success"],
                isolated=("【" + doc + ".pdf-第" in res["answer"])
                and not (
                    "【" + ("Swin" if doc == "MAE" else "MAE") + ".pdf-第"
                    in res["answer"]
                ),
            )

        records += list(
            ex.map(
                user,
                [
                    ("MAE", "MAE 的编码器只看哪些 patches？", "user-A"),
                    ("Swin", "Swin shifted windows 的作用是什么？", "user-B"),
                ],
            )
        )
    test_dir = ROOT / "data/test_fixtures"
    test_dir.mkdir(exist_ok=True)
    import pymupdf

    pdf = pymupdf.open()
    for i in range(301):
        pdf.new_page()
    path = test_dir / "oversized_pages.pdf"
    pdf.save(path)
    pdf.close()
    try:
        load_pdf(str(path))
        ok = False
    except ValueError:
        ok = True
    records.append(dict(test="301_page_limit", passed=ok))
    from docx import Document

    doc = Document()
    doc.add_paragraph("文档加载测试 Mixed formats scientific research")
    doc.save(test_dir / "loader.docx")
    (test_dir / "loader.txt").write_text(
        "文本加载测试 scientific research", encoding="utf-8"
    )
    (test_dir / "loader.md").write_text("# Markdown\n知识库加载测试", encoding="utf-8")
    for ext in ["docx", "txt", "md"]:
        records.append(
            dict(
                test="load_" + ext,
                passed=bool(load_any(str(test_dir / f"loader.{ext}"))),
            )
        )
    env = {
        **os.environ,
        "INDEX_DIR": "D:/NJAU_RAG_Test_Data/empty_index",
        "PYTHONIOENCODING": "utf-8",
    }
    cmd = [
        sys.executable,
        "-c",
        "from retriever.api import retrieve_best; assert retrieve_best('空知识库问题') == []; print('passed')",
    ]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    records.append(
        dict(
            test="empty_knowledge_base",
            passed=result.returncode == 0,
            details=result.stdout + result.stderr,
        )
    )
    (ROOT / "experiments/results/robustness.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(records, flush=True)


if __name__ == "__main__":
    main()
