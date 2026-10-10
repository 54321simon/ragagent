"""Representative end-to-end tests with actual Ollama counters and answers."""

import os, sys, json, time, statistics, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CACHE_ENABLED"] = "0"
from agent.react_loop import react_loop
from dataclasses import asdict


def main():
    dataset = json.loads(
        (ROOT / "experiments/eval_set.json").read_text(encoding="utf-8")
    )
    ids = [
        "F01",
        "F07",
        "F10",
        "F22",
        "C01",
        "C03",
        "C04",
        "C09",
        "S01",
        "S02",
        "S04",
        "S05",
        "R01",
        "R02",
        "R04",
        "R06",
        "U01",
        "U03",
        "U04",
        "U07",
    ]
    samples = [x for x in dataset if x["id"] in ids]
    samples += [
        dict(
            id="T01",
            type="工具路由",
            question="计算 3.14*2.56",
            expected_tools=["calculator"],
            reference_answer="8.0384",
            doc_ids=[],
        ),
        dict(
            id="T02",
            type="工具路由",
            question="当前时间",
            expected_tools=["current_time"],
            reference_answer="北京时间 UTC+08:00",
            doc_ids=[],
        ),
        dict(
            id="T03",
            type="工具路由",
            question="有哪些论文",
            expected_tools=["list_documents"],
            reference_answer="知识库的12篇论文",
            doc_ids=[],
        ),
        dict(
            id="T04",
            type="工具路由",
            question="请提取 MAE 论文的作者、年份、摘要和 DOI。",
            expected_tools=["paper_meta"],
            reference_answer="Kaiming He 等，arXiv首次提交2021年，10.48550/arXiv.2111.06377。",
            doc_ids=["MAE"],
        ),
    ]
    output = []
    for item in samples:
        try:
            result = react_loop(
                item["question"],
                doc_ids=item.get("doc_ids") or None,
                session_id="eval-" + item["id"],
            )
            chosen = [
                s.action
                for s in result["trace"]
                if s.action not in ("Final", "PARSE_ERROR")
            ]
            row = dict(
                id=item["id"],
                type=item["type"],
                question=item["question"],
                reference_answer=item["reference_answer"],
                expected_tools=item["expected_tools"],
                chosen_tools=chosen,
                routing_correct=bool(set(chosen) & set(item["expected_tools"])),
                answer=result["answer"],
                success=result["success"],
                metrics=result["metrics"],
                trace=[asdict(s) for s in result["trace"]],
            )
        except Exception as e:
            row = dict(
                id=item["id"],
                type=item["type"],
                question=item["question"],
                error=str(e),
                success=False,
                routing_correct=False,
            )
        output.append(row)
        path = ROOT / "experiments/results/agent_evaluation.json"
        path.write_text(
            json.dumps({"rows": output}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(item["id"], row.get("chosen_tools"), row.get("metrics"), flush=True)
    rows = [r for r in output if "metrics" in r]
    summary = dict(
        n=len(output),
        success_rate=statistics.mean(int(r["success"]) for r in output),
        tool_routing_accuracy=statistics.mean(
            int(r["routing_correct"]) for r in output
        ),
        avg_iterations=statistics.mean(r["metrics"]["iterations"] for r in rows),
        avg_latency_ms=statistics.mean(r["metrics"]["elapsed_ms"] for r in rows),
        avg_tokens=statistics.mean(r["metrics"]["total_tokens"] for r in rows),
        invalid_citations=sum(r["metrics"]["invalid_citations"] for r in rows),
        definition="Routing correct: at least one expected tool invoked. Success requires tool/model completion and citation guard; neither measures factual correctness.",
        manual_quality_review="pending independent human scoring",
    )
    path.write_text(
        json.dumps(dict(summary=summary, rows=output), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (ROOT / "experiments/results/manual_review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ID",
                "类型",
                "问题",
                "参考答案",
                "系统答案",
                "正确性0至5",
                "完整性0至5",
                "引用准确性0至5",
                "评审人",
                "评审备注",
            ]
        )
        for item, row in zip(samples, output):
            writer.writerow(
                [
                    item["id"],
                    item["type"],
                    item["question"],
                    item["reference_answer"],
                    row.get("answer", ""),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
