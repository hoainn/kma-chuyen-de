#!/usr/bin/env python3
"""
Re-translate DeSFAM LaTeX sections from English source using Claude Batch API.

Strategy:
  - System prompt (cached): Vietnamese glossary + full English paper text
  - Per-request: current Vietnamese draft of each section
  - Claude produces accurate translation by comparing English source with the draft
  - Batch API: 50% cheaper, all 9 sections submitted in parallel
  - Prompt caching: system prompt (~35K tokens) cached across all 9 requests
"""

import os
import time
from pathlib import Path
import anthropic

SECTIONS_DIR = Path(__file__).parent / "latex" / "sections"
PAPER_TXT    = Path(__file__).parent / "paper.txt"

SECTION_FILES = [
    "00_abstract.tex",
    "01_introduction.tex",
    "02_background.tex",
    "03_related_work.tex",
    "04_desfam.tex",
    "05_evaluation.tex",
    "06_discussion.tex",
    "07_conclusion.tex",
    "appendix_a.tex",
]

GLOSSARY = """\
# Bảng thuật ngữ chuẩn (dùng nhất quán trong toàn bài)

## Giữ nguyên — KHÔNG dịch
eBPF, BPF, container, Docker, Kubernetes, seccomp, CVE, MITRE ATT&CK,
VAE, KDE, LSM, AppArmor, SELinux, cgroup, namespace, rootkit, exploit,
payload, API, GPU, DeSFAM, SyscallAD, iForest, PrefixSpan, DongTing

## Dịch sang tiếng Việt (dùng chính xác mẫu dưới đây)
- system call / syscall           → lời gọi hệ thống (syscall)
- kernel                          → nhân (Linux)
- anomaly detection               → phát hiện bất thường
- real-time                       → thời gian thực
- framework                       → framework
- hook (LSM hook, eBPF hook)      → móc (e.g., móc LSM)
- filter / filtering              → bộ lọc / lọc
- privilege escalation            → leo thang đặc quyền
- attack surface                  → bề mặt tấn công
- threat                          → mối đe dọa
- vulnerability                   → lỗ hổng bảo mật
- capability (Linux)              → khả năng (Linux capability)
- whitelist / allowlist           → danh sách cho phép
- blocklist / denylist            → danh sách chặn
- false positive                  → dương tính giả
- false negative                  → âm tính giả
- overhead                        → chi phí hoạt động
- benchmark                       → điểm chuẩn
- Variational Autoencoder         → Bộ tự mã hóa biến phân (VAE)
- Isolation Forest                → Rừng cô lập
- anomaly score                   → điểm bất thường
- microservice                    → dịch vụ vi mô
- orchestration                   → điều phối
- cloud-native                    → gốc đám mây (cloud-native)
- workload                        → khối lượng công việc
- tracing / trace                 → theo vết
- profiling / profile             → hồ sơ hoạt động / tạo hồ sơ
- adaptive                        → thích nghi
- enforcement / enforce           → thực thi
- policy                          → chính sách
- principle of least privilege    → nguyên tắc đặc quyền tối thiểu
- lateral movement                → di chuyển ngang
- data exfiltration               → rò rỉ dữ liệu
- intrusion detection             → phát hiện xâm nhập
- runtime                         → thời điểm chạy
- baseline                        → đường cơ sở
- threshold                       → ngưỡng
- precision                       → độ chính xác
- recall                          → độ phủ
- ablation study                  → nghiên cứu loại bỏ thành phần
- latency                         → độ trễ
- throughput                      → thông lượng
- container escape                → thoát container
- syscall injection               → tiêm lời gọi hệ thống
- access list                     → danh sách truy cập
- sliding window                  → cửa sổ trượt
- risk score                      → điểm rủi ro
- tiered response                 → phản hồi phân tầng

## Quy tắc ngữ pháp
- Chủ ngữ tập thể: "chúng tôi" (không dùng "chúng ta" hay "tác giả")
- Dùng "bài báo này" hoặc "nghiên cứu này", không dùng "paper này"
- Câu học thuật: rõ ràng, trang trọng, không dài dòng
- Giữ nguyên ký hiệu toán học, số công thức, nhãn hình/bảng
- Giữ nguyên tên biến, lệnh trong \\texttt{} và lstlisting
- KHÔNG dịch nội dung bên trong môi trường lstlisting
- KHÔNG thay đổi bất kỳ lệnh LaTeX nào
"""


def build_system_prompt() -> str:
    paper_text = PAPER_TXT.read_text(encoding="utf-8")
    return f"""\
Bạn là chuyên gia dịch thuật học thuật chuyên về bảo mật thông tin và hệ điều hành Linux.
Nhiệm vụ: dịch một section của bài báo khoa học IEEE "DeSFAM" sang tiếng Việt chuẩn xác, \
dựa trên bản tiếng Anh gốc bên dưới và bảng thuật ngữ chuẩn.

{GLOSSARY}

---
# BẢN GỐC TIẾNG ANH (toàn bài — dùng làm nguồn tham chiếu)

{paper_text}
---

## Quy tắc đầu ra
1. Trả về DUY NHẤT nội dung LaTeX đã dịch, không có giải thích hay bình luận.
2. Giữ nguyên toàn bộ lệnh LaTeX (\\section, \\label, \\cite, \\begin, \\end, v.v.)
3. Giữ nguyên môi trường toán học ($...$, \\[ \\], equation, align).
4. Giữ nguyên nội dung bên trong lstlisting (code không dịch).
5. Áp dụng bảng thuật ngữ nhất quán.
6. Nếu bản dịch hiện tại đã đúng một phần, giữ lại cấu trúc và chỉ sửa những chỗ sai/không tự nhiên.
"""


def load_section(filename: str) -> str:
    return (SECTIONS_DIR / filename).read_text(encoding="utf-8")


def save_section(filename: str, content: str) -> None:
    path = SECTIONS_DIR / filename
    backup = SECTIONS_DIR / (filename + ".bak")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    print(f"  Written: {filename}")


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```latex"):
        text = text[len("```latex"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def create_batch_requests(system_prompt: str) -> list:
    requests = []
    for filename in SECTION_FILES:
        current_vi = load_section(filename)
        user_msg = (
            f"Section file: {filename}\n\n"
            f"Dưới đây là bản dịch tiếng Việt hiện tại (cần kiểm tra và cải thiện dựa trên bản gốc tiếng Anh):\n\n"
            f"```latex\n{current_vi}\n```\n\n"
            f"Hãy tạo ra bản dịch tiếng Việt tốt hơn cho section này, "
            f"đối chiếu với bản gốc tiếng Anh và áp dụng bảng thuật ngữ chuẩn."
        )
        requests.append(
            anthropic.types.message_create_params.Request(
                custom_id=filename,
                params={
                    "model": "claude-opus-4-7",
                    "max_tokens": 16000,
                    "thinking": {"type": "adaptive"},
                    "system": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "messages": [{"role": "user", "content": user_msg}],
                },
            )
        )
    return requests


def process_results(client: anthropic.Anthropic, batch_id: str):
    succeeded, failed = 0, []
    for result in client.messages.batches.results(batch_id):
        filename = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next(
                (b.text for b in msg.content if b.type == "text"), None
            )
            if text:
                save_section(filename, strip_fences(text))
                succeeded += 1
            else:
                print(f"  WARNING: empty response for {filename}")
                failed.append(filename)
        else:
            print(f"  ERROR [{result.result.type}]: {filename}")
            failed.append(filename)
    return succeeded, failed


def poll_until_done(client: anthropic.Anthropic, batch_id: str):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        done = c.succeeded + c.errored + c.canceled + c.expired
        total = done + c.processing
        print(f"    {batch.processing_status} | {done}/{total} done "
              f"(ok={c.succeeded} err={c.errored})")
        if batch.processing_status == "ended":
            break
        time.sleep(30)


def run_batch_translation():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    print("==> Building system prompt (English source + glossary)...")
    system_prompt = build_system_prompt()
    prompt_kb = len(system_prompt.encode()) / 1024
    print(f"    System prompt: {prompt_kb:.1f} KB (will be cached after first request)")

    print(f"==> Submitting batch ({len(SECTION_FILES)} sections)...")
    requests = create_batch_requests(system_prompt)
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"    Batch ID: {batch_id}")

    state_file = Path(__file__).parent / ".translate_batch_id"
    state_file.write_text(batch_id)

    print("\n==> Polling (Batch API typically completes in 5-30 min)...")
    poll_until_done(client, batch_id)

    print("\n==> Writing results...")
    succeeded, failed = process_results(client, batch_id)

    print(f"\n==> Done: {succeeded}/{len(SECTION_FILES)} sections translated")
    if failed:
        print(f"    Failed: {failed}")
    state_file.unlink(missing_ok=True)
    return succeeded, failed


def resume_batch(batch_id: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    print(f"==> Resuming batch {batch_id}...")
    poll_until_done(client, batch_id)

    print("\n==> Writing results...")
    succeeded, failed = process_results(client, batch_id)
    print(f"\n==> Done: {succeeded}/{len(SECTION_FILES)} sections translated")
    if failed:
        print(f"    Failed: {failed}")
    return succeeded, failed


if __name__ == "__main__":
    import sys

    state_file = Path(__file__).parent / ".translate_batch_id"

    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        bid = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else state_file.read_text().strip()
        resume_batch(bid)
    elif state_file.exists() and "--new" not in sys.argv:
        bid = state_file.read_text().strip()
        print(f"Found in-progress batch: {bid}")
        print("Resuming... (pass --new to start a fresh batch instead)")
        resume_batch(bid)
    else:
        state_file.unlink(missing_ok=True)
        run_batch_translation()
