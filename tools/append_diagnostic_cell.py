"""Append the diagnostic batch to the user's notebook; preserve other cells."""
import json
from pathlib import Path
import uuid


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/DataMate_Colab_Inference.ipynb"
    original = path.read_bytes()
    notebook = json.loads(original)
    backups = root / "outputs/diagnosis-2026-11-02"
    backups.mkdir(parents=True, exist_ok=True)
    backup = backups / ("inference-before-diagnostic-" + uuid.uuid4().hex[:8] + ".ipynb")
    backup.write_bytes(original)
    code_id, markdown_id = "datamate-train-diagnostic-120", "datamate-train-diagnostic-help"
    existing = next((c for c in notebook["cells"] if c.get("id") == code_id), None)
    if existing is None:
        notebook["cells"].append({"cell_type": "markdown", "id": markdown_id, "metadata": {}, "source": []})
        existing = {"cell_type": "code", "id": code_id, "metadata": {}, "execution_count": None, "outputs": []}
        notebook["cells"].append(existing)
    help_cell = next(c for c in notebook["cells"] if c.get("id") == markdown_id)
    help_cell["source"] = [
            "## Chẩn đoán batch: 72 câu train, 3 câu mỗi doanh nghiệp\n",
            "Chạy ô bên dưới trong cùng kernel sau khi đã load `agent`. Dùng lại model hiện có, giữ nguyên prompt và decoding; không fine-tune.\n",
            "Đúng 3 câu mỗi development business, phủ câu không dấu, viết tắt và tín hiệu độ khó; các nhóm có thể chồng lấp. Chọn từ câu train nguyên bản, không tự sửa lời câu hỏi. Giữ riêng 6 holdout; PA không đại diện toàn train.\n",
            "Mặc định trần mềm 90 phút (khoảng 15 Unit ở mức 10 Unit/giờ), nếu chưa đủ 72 câu sẽ xuất báo cáo `partial`. Log lưu từng câu, có thể tiếp tục nếu cùng cấu hình.\n",
            "Đầu ô có các biến cấu hình. `AUTO_DISCONNECT=False`: ngắt GPU thủ công sau dòng REPORT. Nếu đổi True, code thử đồng bộ Drive rồi kết thúc runtime sau khi ghi ZIP; hỗ trợ tùy môi trường Colab.\n",
            "Xem `summary.json` (có điểm theo nhóm câu), `review.csv`, `review.jsonl`, `submission.jsonl` và `logs/` trong ZIP. Tải ZIP về ổ E: để phân tích.\n",
        ]
    existing["source"] = (root / "tools/diagnose_train.py").read_text(encoding="utf-8").splitlines(keepends=True)
    # Editing code clears only this diagnostic cell's stale outputs.
    existing["execution_count"], existing["outputs"] = None, []
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Backup:", backup)
    print("Updated:", path)


if __name__ == "__main__":
    main()
