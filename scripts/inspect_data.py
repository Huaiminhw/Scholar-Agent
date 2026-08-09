"""
数据探索工具

递归扫描 data/ 目录，统计 jsonl 行数、字段结构、样例。
输出 Markdown + JSON 报告。
"""
import json
import zipfile
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
from scholar_agent.config import settings
from scholar_agent.utils.logger import setup_logger, logger

DATA_DIR = settings.data_dir
OUT_MD = settings.project_root / 'data_schema_report.md'
OUT_JSON = settings.project_root / 'data_schema_report.json'


def get_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        if len(value) == 0:
            return "list[empty]"
        inner_types = sorted({get_type_name(x) for x in value[:20]})
        return "list[" + "|".join(inner_types) + "]"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def short_value(value: Any, max_len: int = 300) -> Any:
    if isinstance(value, str):
        return value[:max_len] + ("..." if len(value) > max_len else "")
    if isinstance(value, list):
        return [short_value(x, max_len=120) for x in value[:3]]
    if isinstance(value, dict):
        return {k: short_value(v, max_len=120) for k, v in list(value.items())[:8]}
    return value


def read_jsonl_sample(path: Path, max_rows: int = 2000) -> dict[str, Any]:
    total_lines = 0
    valid_lines = 0
    bad_lines = 0

    key_counter: Counter[str] = Counter()
    type_counter: dict[str, Counter[str]] = defaultdict(Counter)
    first_obj = None
    examples_by_key: dict[str, Any] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            valid_lines += 1
            if first_obj is None:
                first_obj = obj

            if isinstance(obj, dict):
                for k, v in obj.items():
                    key_counter[k] += 1
                    type_counter[k][get_type_name(v)] += 1
                    if k not in examples_by_key:
                        examples_by_key[k] = short_value(v)
            else:
                type_counter["__root__"][get_type_name(obj)] += 1

            if valid_lines >= max_rows:
                for _ in f:
                    total_lines += 1
                break

    fields = []
    for key, cnt in key_counter.most_common():
        fields.append({
            "field": key,
            "seen_in_sample": cnt,
            "types": dict(type_counter[key]),
            "example": examples_by_key.get(key),
        })

    return {
        "file": str(path),
        "kind": "jsonl",
        "total_lines": total_lines,
        "valid_sample_lines": valid_lines,
        "bad_sample_lines": bad_lines,
        "sample_limit": max_rows,
        "fields": fields,
        "first_obj": short_value(first_obj),
    }


def read_json_file(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "file": str(path),
        "kind": "json",
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        info["error"] = repr(e)
        return info

    info["root_type"] = get_type_name(obj)
    if isinstance(obj, dict):
        info["top_level_keys_count"] = len(obj)
        info["top_level_keys_sample"] = list(obj.keys())[:10]
        if obj:
            first_key = next(iter(obj.keys()))
            first_value = obj[first_key]
            info["first_key"] = first_key
            info["first_value_type"] = get_type_name(first_value)
            info["first_value_sample"] = short_value(first_value)
            if isinstance(first_value, dict):
                info["paper_fields"] = {k: get_type_name(v) for k, v in first_value.items()}
    elif isinstance(obj, list):
        info["list_len"] = len(obj)
        if obj:
            info["first_item_type"] = get_type_name(obj[0])
            info["first_item_sample"] = short_value(obj[0])
            if isinstance(obj[0], dict):
                info["item_fields"] = {k: get_type_name(v) for k, v in obj[0].items()}
    return info


def inspect_zip(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "file": str(path),
        "kind": "zip",
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            info["inner_file_count"] = len(names)
            info["inner_files_sample"] = names[:30]
            total_uncompressed = sum(item.file_size for item in zf.infolist())
            info["uncompressed_size_mb"] = round(total_uncompressed / 1024 / 1024, 2)
    except Exception as e:
        info["error"] = repr(e)
    return info


def write_markdown(results: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# PaSa Data Schema Report\n")
    lines.append(f"数据目录：`{DATA_DIR}`\n")
    lines.append(f"文件数量：{len(results)}\n")

    for item in results:
        path = Path(item["file"])
        try:
            rel = path.relative_to(DATA_DIR)
        except ValueError:
            rel = path

        lines.append("\n---\n")
        lines.append(f"## `{rel}`\n")
        lines.append(f"- 类型：`{item.get('kind')}`\n")

        if item.get("kind") == "jsonl":
            lines.append(f"- 总行数：{item.get('total_lines')}\n")
            lines.append(f"- 抽样有效行数：{item.get('valid_sample_lines')}\n")
            lines.append(f"- 抽样坏行数：{item.get('bad_sample_lines')}\n")
            lines.append("\n### 字段结构\n")
            lines.append("| 字段 | 抽样出现次数 | 类型 | 示例 |\n")
            lines.append("|---|---:|---|---|\n")
            for f in item.get("fields", []):
                example = json.dumps(f.get("example"), ensure_ascii=False)
                types = json.dumps(f.get("types"), ensure_ascii=False)
                lines.append(f"| `{f.get('field')}` | {f.get('seen_in_sample')} | `{types}` | `{example}` |\n")
            lines.append("\n### 第一条样例\n")
            lines.append("```json\n")
            lines.append(json.dumps(item.get("first_obj"), ensure_ascii=False, indent=2))
            lines.append("\n```\n")

        elif item.get("kind") == "json":
            lines.append(f"- 文件大小：{item.get('size_mb')} MB\n")
            if "error" in item:
                lines.append(f"- 解析错误：`{item['error']}`\n")
            else:
                lines.append(f"- 顶层类型：`{item.get('root_type')}`\n")
                if "top_level_keys_count" in item:
                    lines.append(f"- 顶层 key 数量：{item.get('top_level_keys_count')}\n")
                    lines.append(f"- 顶层 key 示例：`{item.get('top_level_keys_sample')}`\n")
                if "paper_fields" in item:
                    lines.append("\n### 单篇论文字段\n")
                    lines.append("| 字段 | 类型 |\n")
                    lines.append("|---|---|\n")
                    for k, v in item["paper_fields"].items():
                        lines.append(f"| `{k}` | `{v}` |\n")
                lines.append("\n### 样例\n")
                lines.append("```json\n")
                sample = {k: item[k] for k in item if k not in {"file", "kind"}}
                lines.append(json.dumps(sample, ensure_ascii=False, indent=2))
                lines.append("\n```\n")

        elif item.get("kind") == "zip":
            lines.append(f"- 压缩包大小：{item.get('size_mb')} MB\n")
            if "error" in item:
                lines.append(f"- 读取错误：`{item['error']}`\n")
            else:
                lines.append(f"- 内部文件数：{item.get('inner_file_count')}\n")
                lines.append(f"- 解压后估计大小：{item.get('uncompressed_size_mb')} MB\n")
                lines.append("\n### 内部文件示例\n")
                lines.append("```text\n")
                lines.append("\n".join(item.get("inner_files_sample", [])))
                lines.append("\n```\n")

    OUT_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    setup_logger(settings.output_dir)
    if not DATA_DIR.exists():
        logger.error(f"Data directory does not exist: {DATA_DIR}")
        return

    results: list[dict[str, Any]] = []
    files = sorted([p for p in DATA_DIR.rglob("*") if p.is_file()])
    logger.info(f"Found {len(files)} files, scanning {DATA_DIR}")

    for path in files:
        suffix = path.suffix.lower()
        try:
            rel = path.relative_to(DATA_DIR)
        except ValueError:
            rel = path
        logger.info(f"Parsing: {rel}")

        if suffix == ".jsonl":
            item = read_jsonl_sample(path)
        elif suffix == ".json":
            item = read_json_file(path)
        elif suffix == ".zip":
            item = inspect_zip(path)
        else:
            item = {"file": str(path), "kind": "other", "size_mb": round(path.stat().st_size / 1024 / 1024, 2)}
        results.append(item)

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(results)
    logger.info(f"Done. Report: {OUT_MD}")


if __name__ == "__main__":
    main()
