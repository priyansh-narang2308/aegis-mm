"""
Aegis-MM Automated Evaluation Harness
Runs multi-concurrency stress benchmarks and generates comprehensive SLA reports.
"""
import asyncio
from typing import Dict, Any, List
from src.benchmarks.load_tester import run_concurrency_load_test


async def generate_benchmark_report() -> Dict[str, Any]:
    concurrency_levels = [1, 4, 8, 16]
    report_data = []
    
    print("\nExecuting Aegis-MM Automated Concurrency Benchmark Suite...")
    for n in concurrency_levels:
        res = await run_concurrency_load_test(num_sessions=n, requests_per_session=10)
        report_data.append(res)
        
    return {"concurrency_suite": report_data}


def format_benchmark_report_md(report: Dict[str, Any]) -> str:
    md = "\n# ⚡ Aegis-MM Production Benchmark Report\n\n"
    md += "### High-Concurrency SLA Performance (End-to-End Multimodal Late Fusion)\n\n"
    md += "| Concurrent Sessions | Throughput (Req/sec) | Mean Latency | p50 SLA | p95 SLA | p99 SLA |\n"
    md += "| :---: | :---: | :---: | :---: | :---: | :---: |\n"
    
    for item in report["concurrency_suite"]:
        n = item["concurrent_sessions"]
        thr = item["throughput_req_per_sec"]
        l = item["latency_ms"]
        md += f"| **{n} Sessions** | **{thr} req/s** | {l['mean']} ms | **{l['p50']} ms** | `{l['p95']} ms` | {l['p99']} ms |\n"
        
    return md


if __name__ == "__main__":
    rep = asyncio.run(generate_benchmark_report())
    print(format_benchmark_report_md(rep))
