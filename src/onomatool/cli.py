import argparse
import csv
import fnmatch
import glob
import json
import difflib
import os
import shutil
import shlex
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import toml

from onomatool.config import DEFAULT_CONFIG, get_config
from onomatool.conflict_resolver import resolve_conflict
from onomatool.file_collector import collect_files
from onomatool.file_dispatcher import FileDispatcher
from onomatool.llm_integration import get_suggestions, list_ollama_models
from onomatool.utils.cover_utils import cover_workflow, extract_isbn
from onomatool.utils.titlecase_utils import apply_titlecase, evaluate_title
from onomatool.utils.image_utils import convert_svg_to_png

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _merge_overrides(base: dict, overrides: dict) -> dict:
    merged = base.copy()
    merged_markitdown = base.get("markitdown", {}).copy()
    if isinstance(overrides.get("markitdown"), dict):
        merged_markitdown.update(overrides["markitdown"])
    merged.update(overrides)
    merged["markitdown"] = merged_markitdown
    return merged


def apply_batch_rules(base_config: dict, file_path: str) -> dict:
    rules = base_config.get("batch_rules") or []
    config = base_config
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        pattern = rule.get("pattern") or rule.get("path") or rule.get("match")
        if not pattern:
            continue
        if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(
            os.path.basename(file_path), pattern
        ):
            overrides = rule.get("overrides") or rule.get("config") or {}
            if isinstance(overrides, dict):
                config = _merge_overrides(config, overrides)
    return config


def _provider_model(config: dict) -> tuple[str, str]:
    provider = config.get("default_provider", "openai")
    if provider == "ollama":
        model = config.get("ollama_model") or config.get("llm_model", "")
    elif provider == "anthropic":
        model = config.get("anthropic_model") or config.get("llm_model", "")
    elif provider == "openai":
        model = config.get("openai_model") or config.get("llm_model", "")
    elif provider == "google":
        model = "gemini-pro"
    else:
        model = config.get("llm_model", "")
    return provider, model


def _build_report_path(config: dict, report_format: str, report_path: str | None) -> str:
    if report_path:
        return os.path.expanduser(report_path)
    report_dir = config.get("report_dir") or os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(report_dir, f"onoma-report-{timestamp}.{report_format}")


def _cleanup_junk_files(pattern: str) -> None:
    # Delete .DS_Store and ._* files under the base path for the pattern
    base = os.path.expanduser(pattern)
    magic_chars = ["*", "?", "[", "{"]
    first_magic = min([base.find(ch) for ch in magic_chars if base.find(ch) != -1], default=-1)
    if first_magic != -1:
        base = base[:first_magic]
    base = base.rstrip(os.sep)
    if not base:
        base = "."
    if not os.path.isdir(base):
        base = os.path.dirname(base) or "."
    for root, _, files in os.walk(base):
        for name in files:
            if name == ".DS_Store" or name.startswith("._"):
                try:
                    os.remove(os.path.join(root, name))
                except Exception:
                    pass


def _write_report(entries: list[dict], path: str, report_format: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if report_format == "csv":
        fieldnames = [
            "timestamp",
            "status",
            "original_path",
            "final_path",
            "suggestion",
            "provider",
            "model",
            "score",
            "flags",
            "error",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                writer.writerow({k: entry.get(k, "") for k in fieldnames})
    elif report_format == "html":
        rows = []
        for e in entries:
            before = os.path.basename(e.get("original_path", ""))
            after = os.path.basename(e.get("final_path", "")) if e.get("final_path") else ""
            diff = "".join(
                (
                    f"<ins>{d[2:]}</ins>" if d.startswith("+ ")
                    else f"<del>{d[2:]}</del>" if d.startswith("- ")
                    else d[2:]
                )
                for d in difflib.ndiff(before, after)
            )
            rows.append(
                f"<tr><td>{e.get('timestamp','')}</td><td>{e.get('status','')}</td>"
                f"<td>{e.get('original_path','')}</td><td>{e.get('final_path','')}</td>"
                f"<td>{e.get('suggestion','')}</td><td>{e.get('provider','')}</td>"
                f"<td>{e.get('model','')}</td><td>{e.get('score','')}</td>"
                f"<td>{e.get('flags','')}</td><td>{e.get('error','')}</td>"
                f"<td>{diff}</td></tr>"
            )
        html = (
            "<html><head><meta charset='utf-8'/>"
            "<style>body{font-family:sans-serif}table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #ddd;padding:6px}ins{background:#c7f5c7;text-decoration:none}"
            "del{background:#f7c3c3}</style></head><body>"
            "<h2>Onoma Report</h2><table><tr>"
            "<th>Timestamp</th><th>Status</th><th>Original</th><th>Final</th>"
            "<th>Suggestion</th><th>Provider</th><th>Model</th><th>Score</th><th>Flags</th>"
            "<th>Error</th><th>Diff</th>"
            "</tr>"
            + "".join(rows)
            + "</table></body></html>"
        )
        with open(path, "w") as f:
            f.write(html)
    else:
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    if report_format in {"jsonl", "csv", "html"}:
        # also write undo script
        _write_undo_script(entries, path + ".undo.sh")


def _write_undo_script(entries: list[dict], path: str) -> None:
    lines = ["#!/bin/bash", "set -euo pipefail"]
    for e in entries:
        if e.get("status") not in {"renamed", "duplicate"}:
            continue
        src = e.get("final_path") or ""
        dst = e.get("original_path") or ""
        if not src or not dst:
            continue
        lines.append(f"mv {shlex.quote(src)} {shlex.quote(dst)}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)


def _load_report_entries(path: str) -> list[dict]:
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _find_latest_report(config: dict) -> str | None:
    report_dir = config.get("report_dir") or os.getcwd()
    candidates = []
    for ext in ("jsonl", "csv"):
        candidates.extend(glob.glob(os.path.join(report_dir, f"onoma-report-*.{ext}")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _plan_rename(file_path: str, new_name: str) -> tuple[str, str]:
    directory = os.path.dirname(file_path) or "."
    _, ext = os.path.splitext(file_path)
    base_new_name, _ = os.path.splitext(new_name)
    new_name_with_ext = base_new_name + ext
    existing_files = os.listdir(directory)
    final_name = resolve_conflict(new_name_with_ext, existing_files)
    final_path = os.path.join(directory, final_name)
    return final_name, final_path


def _hash_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _build_rename_plan(items: list[dict], config: dict) -> list[dict]:
    duplicates_dir_name = config.get("duplicates_dir", "duplicates")
    fuzzy_threshold = float(config.get("fuzzy_duplicate_threshold", 0.9))
    fuzzy_candidates: list[tuple[str, str]] = []
    # Build desired targets
    desired = []
    for item in items:
        orig = item["original_path"]
        suggestion = item["suggestion"]
        directory = os.path.dirname(orig) or "."
        _, ext = os.path.splitext(orig)
        base_new, _ = os.path.splitext(suggestion)
        desired_name = base_new + ext
        desired.append(
            {
                **item,
                "directory": directory,
                "desired_name": desired_name,
                "desired_key": os.path.join(directory, desired_name).lower(),
            }
        )

    # Collision groups (case-insensitive)
    collision_groups: dict[str, list[dict]] = {}
    for d in desired:
        collision_groups.setdefault(d["desired_key"], []).append(d)

    duplicates: set[str] = set()
    for group in collision_groups.values():
        if len(group) < 2:
            continue
        # hash-based duplicate detection
        hash_map: dict[str, list[dict]] = {}
        for entry in group:
            try:
                h = _hash_file(entry["original_path"])
            except Exception:
                h = ""
            hash_map.setdefault(h, []).append(entry)
        for h, entries in hash_map.items():
            if h and len(entries) > 1:
                # keep first, mark rest as duplicates
                for dup in entries[1:]:
                    duplicates.add(dup["original_path"])

    # Resolve final names
    plan: list[dict] = []
    per_dir_existing: dict[str, list[str]] = {}
    per_dir_lower: dict[str, set[str]] = {}

    for entry in desired:
        orig = entry["original_path"]
        directory = entry["directory"]
        desired_name = entry["desired_name"]
        provider = entry.get("provider", "")
        model_name = entry.get("model", "")

        if directory not in per_dir_existing:
            per_dir_existing[directory] = os.listdir(directory)
            per_dir_lower[directory] = {n.lower() for n in per_dir_existing[directory]}

        if orig in duplicates:
            dup_dir = os.path.join(directory, duplicates_dir_name)
            os.makedirs(dup_dir, exist_ok=True)
            dup_existing = os.listdir(dup_dir)
            dup_name = resolve_conflict(os.path.basename(orig), dup_existing)
            target_path = os.path.join(dup_dir, dup_name)
            plan.append(
                {
                    "action": "duplicate",
                    "original_path": orig,
                    "target_path": target_path,
                    "suggestion": entry["suggestion"],
                    "provider": provider,
                    "model": model_name,
                    "isbn": entry.get("isbn"),
                }
            )
            continue

        if desired_name.lower() in per_dir_lower[directory]:
            final_name = resolve_conflict(desired_name, per_dir_existing[directory])
        else:
            final_name = desired_name
        per_dir_existing[directory].append(final_name)
        per_dir_lower[directory].add(final_name.lower())
        target_path = os.path.join(directory, final_name)

        if os.path.abspath(orig) == os.path.abspath(target_path):
            plan.append(
                {
                    "action": "skip",
                    "original_path": orig,
                    "target_path": target_path,
                    "suggestion": entry["suggestion"],
                    "provider": provider,
                    "model": model_name,
                    "isbn": entry.get("isbn"),
                }
            )
        else:
            plan.append(
                {
                    "action": "rename",
                    "original_path": orig,
                    "target_path": target_path,
                    "suggestion": entry["suggestion"],
                    "provider": provider,
                    "model": model_name,
                    "isbn": entry.get("isbn"),
                }
            )

    # Fuzzy duplicate detection (advisory)
    try:
        from rapidfuzz import fuzz

        titles = [(p["original_path"], os.path.basename(p["target_path"])) for p in plan if p["action"] == "rename"]
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                score = fuzz.ratio(titles[i][1], titles[j][1]) / 100.0
                if score >= fuzzy_threshold:
                    fuzzy_candidates.append((titles[i][0], titles[j][0]))
    except Exception:
        pass

    if fuzzy_candidates:
        for a, b in fuzzy_candidates:
            plan.append(
                {
                    "action": "fuzzy_duplicate",
                    "original_path": a,
                    "target_path": b,
                    "suggestion": "",
                    "provider": "",
                    "model": "",
                }
            )

    return plan


def _rename_folders(file_paths: list[str], config: dict, add_report_entry) -> None:
    if not file_paths:
        return
    # Build unique directories, deepest first
    dirs = {os.path.dirname(p) for p in file_paths}
    # remove empty
    dirs = {d for d in dirs if d}
    for directory in sorted(dirs, key=lambda d: d.count(os.sep), reverse=True):
        if not os.path.isdir(directory):
            continue
        parent = os.path.dirname(directory) or "."
        base = os.path.basename(directory)
        new_base = apply_titlecase(base, config=config, original_ext=None)
        if new_base == base:
            add_report_entry("folder_skipped", directory, directory)
            continue
        existing = os.listdir(parent)
        final_name = resolve_conflict(new_base, existing)
        target = os.path.join(parent, final_name)
        if os.path.abspath(directory) == os.path.abspath(target):
            add_report_entry("folder_skipped", directory, target)
            continue
        try:
            shutil.move(directory, target)
            add_report_entry("folder_renamed", directory, target)
        except Exception as err:
            add_report_entry("folder_error", directory, target, error=str(err))

def _undo_from_report(report_path: str) -> list[dict]:
    entries = _load_report_entries(report_path)
    renames = [e for e in entries if e.get("status") == "renamed"]
    results: list[dict] = []
    for entry in reversed(renames):
        original_path = entry.get("original_path") or ""
        final_path = entry.get("final_path") or entry.get("new_path") or ""
        if not original_path or not final_path:
            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "status": "undo_error",
                    "original_path": original_path,
                    "final_path": final_path,
                    "suggestion": "",
                    "provider": "",
                    "model": "",
                    "error": "Missing original_path or final_path in report entry",
                }
            )
            continue
        if not os.path.exists(final_path):
            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "status": "undo_error",
                    "original_path": original_path,
                    "final_path": final_path,
                    "suggestion": "",
                    "provider": "",
                    "model": "",
                    "error": "Final path does not exist",
                }
            )
            continue
        target_dir = os.path.dirname(original_path) or "."
        target_name = os.path.basename(original_path)
        existing_files = os.listdir(target_dir)
        safe_name = resolve_conflict(target_name, existing_files)
        target_path = os.path.join(target_dir, safe_name)
        try:
            shutil.move(final_path, target_path)
            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "status": "undo",
                    "original_path": original_path,
                    "final_path": target_path,
                    "suggestion": "",
                    "provider": "",
                    "model": "",
                    "error": "",
                }
            )
        except Exception as err:
            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "status": "undo_error",
                    "original_path": original_path,
                    "final_path": final_path,
                    "suggestion": "",
                    "provider": "",
                    "model": "",
                    "error": str(err),
                }
            )
    return results


def main(args=None):
    try:
        if args is None:
            args = sys.argv[1:]
        parser = argparse.ArgumentParser(
            description="Onoma - AI-powered file renaming tool",
            epilog="Configuration is loaded from ~/.onomarc (TOML format)",
            add_help=False,
        )
        parser.add_argument(
            "-h", "--help", action="help", help="Show this help message and exit"
        )
        parser.add_argument(
            "pattern",
            nargs="?",
            help="Glob pattern to match files (e.g., '*.pdf', 'docs/**/*.md')",
        )
        parser.add_argument(
            "-f",
            "--format",
            help="Force specific file format processing (optional)",
            choices=["text", "markdown", "pdf", "docx", "image"],
            metavar="FORMAT",
        )
        parser.add_argument(
            "-s",
            "--save-config",
            action="store_true",
            help="Save default configuration to ~/.onomarc and exit",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Print basic LLM debugging information",
        )
        parser.add_argument(
            "-vv",
            "--very-verbose",
            action="store_true",
            help="Print detailed LLM request and response for debugging",
        )
        parser.add_argument(
            "-d",
            "--dry-run",
            action="store_true",
            help="Show intended renames but do not modify any files",
        )
        parser.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            help=(
                "With --dry-run, prompt for confirmation and then perform the renames "
                "using the dry-run suggestions"
            ),
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help=(
                "Do not delete temporary files created for SVG, PDF, or PPTX processing; "
                "print their paths."
            ),
        )
        parser.add_argument(
            "--config",
            help="Specify a configuration file to use",
        )
        parser.add_argument(
            "--ollama-list",
            action="store_true",
            help="List installed Ollama models and exit",
        )
        parser.add_argument(
            "--ollama-select",
            action="store_true",
            help="Prompt to select an installed Ollama model for this run",
        )
        parser.add_argument(
            "--ollama-model",
            help="Use a specific Ollama model for this run (implies provider=ollama)",
        )
        parser.add_argument(
            "--forget-last",
            action="store_true",
            help="Clear remembered last selections (provider/model/pattern)",
        )
        parser.add_argument(
            "--forget-last-provider",
            action="store_true",
            help="Clear remembered last provider selection",
        )
        parser.add_argument(
            "--forget-last-pattern",
            action="store_true",
            help="Clear remembered last path/pattern",
        )
        parser.add_argument(
            "--forget-last-ollama-model",
            action="store_true",
            help="Clear remembered last Ollama model",
        )
        parser.add_argument(
            "--report",
            help="Write a report to a specific path",
        )
        parser.add_argument(
            "--report-format",
            choices=["jsonl", "csv", "html"],
            help="Report format (jsonl, csv, or html)",
        )
        parser.add_argument(
            "--no-report",
            action="store_true",
            help="Disable report output for this run",
        )
        parser.add_argument(
            "--undo",
            action="store_true",
            help="Undo the last rename run (uses last report by default)",
        )
        parser.add_argument(
            "--undo-report",
            help="Undo using a specific report file",
        )
        parser.add_argument(
            "--rename-folders",
            action="store_true",
            help="Rename folders (bottom-up) using title casing",
        )
        parser.add_argument(
            "--cover",
            action="store_true",
            help="Enable cover extraction and saving",
        )
        args = parser.parse_args(args)

        if args.save_config:
            save_default_config()
            print("Default configuration saved to ~/.onomarc")
            return 0

        if args.ollama_list:
            config = get_config(args.config)
            models = list_ollama_models(config)
            if not models:
                print("No Ollama models found.")
                return 0
            print("Installed Ollama models:")
            for name in models:
                print(f"- {name}")
            return 0

        if args.forget_last or args.forget_last_provider or args.forget_last_pattern or args.forget_last_ollama_model:
            config = get_config(args.config)
            if args.forget_last or args.forget_last_provider:
                config.pop("last_provider_choice", None)
            if args.forget_last or args.forget_last_pattern:
                config.pop("last_pattern", None)
            if args.forget_last or args.forget_last_ollama_model:
                config.pop("ollama_model", None)
            _save_config(config, args.config)
            print("Cleared remembered selections.")
            return 0

        if args.undo:
            config = get_config(args.config)
            report_path = (
                args.undo_report
                or config.get("last_report_path")
                or _find_latest_report(config)
            )
            if not report_path:
                print("No report found to undo.")
                return 1
            report_path = os.path.expanduser(report_path)
            if not os.path.exists(report_path):
                print(f"Report not found: {report_path}")
                return 1
            undo_entries = _undo_from_report(report_path)
            if not undo_entries:
                print("No renames found in report.")
                return 1
            report_enabled = config.get("report_enabled", True) and not args.no_report
            report_format = args.report_format or config.get("report_format", "jsonl")
            if report_enabled:
                out_path = _build_report_path(config, report_format, args.report)
                _write_report(undo_entries, out_path, report_format)
                config["last_report_path"] = out_path
                _save_config(config, args.config)
                print(f"Undo report written to: {out_path}")
            ok = sum(1 for r in undo_entries if r.get("status") == "undo")
            err = sum(1 for r in undo_entries if r.get("status") == "undo_error")
            print(f"Undo complete: {ok} succeeded, {err} failed.")
            return 0 if err == 0 else 1

        def prompt_choice(prompt: str, options: list[str]) -> int:
            print(prompt)
            for idx, label in enumerate(options, start=1):
                print(f"{idx}. {label}")
            choice = input("Enter number: ").strip()
            if not choice.isdigit() or not (1 <= int(choice) <= len(options)):
                return -1
            return int(choice) - 1

        def interactive_run() -> tuple[dict, str] | None:
            config = get_config(args.config)
            provider_options = [
                "Anthropic → OpenAI (fallback)",
                "Ollama → Anthropic → OpenAI (fallback)",
                "Anthropic only",
                "OpenAI only",
                "Ollama only",
            ]
            last_choice = config.get("last_provider_choice")
            idx = -1
            if isinstance(last_choice, int) and 0 <= last_choice < len(provider_options):
                keep = input(
                    f"Use last selection ({provider_options[last_choice]})? [Y/n]: "
                ).strip().lower()
                if keep in {"", "y", "yes"}:
                    idx = last_choice
            if idx < 0:
                idx = prompt_choice(
                    "Select model/provider mode for this run:", provider_options
                )
                if idx < 0:
                    print("Invalid selection.")
                    return None

            if idx == 0:
                config["default_provider"] = "anthropic"
            elif idx == 1:
                config["default_provider"] = "ollama"
            elif idx == 2:
                config["default_provider"] = "anthropic"
                config["disable_fallback"] = True
            elif idx == 3:
                config["default_provider"] = "openai"
                config["disable_fallback"] = True
            elif idx == 4:
                config["default_provider"] = "ollama"
                config["disable_fallback"] = True

            if config.get("default_provider") == "ollama":
                models = list_ollama_models(config)
                if not models:
                    print("No Ollama models found.")
                    return None
                last_ollama_model = config.get("ollama_model")
                if isinstance(last_ollama_model, str) and last_ollama_model in models:
                    keep_model = input(
                        f"Use last Ollama model ({last_ollama_model})? [Y/n]: "
                    ).strip().lower()
                    if keep_model in {"", "y", "yes"}:
                        config["ollama_model"] = last_ollama_model
                    else:
                        m_idx = prompt_choice("Select an Ollama model:", models)
                        if m_idx < 0:
                            print("Invalid selection.")
                            return None
                        config["ollama_model"] = models[m_idx]
                else:
                    m_idx = prompt_choice("Select an Ollama model:", models)
                    if m_idx < 0:
                        print("Invalid selection.")
                        return None
                    config["ollama_model"] = models[m_idx]

            last_pattern = config.get("last_pattern")
            if isinstance(last_pattern, str) and last_pattern:
                keep_pattern = input(
                    f"Use last path/pattern ({last_pattern})? [Y/n]: "
                ).strip().lower()
                if keep_pattern in {"", "y", "yes"}:
                    pattern = last_pattern
                else:
                    pattern = input("Enter file path or glob pattern: ").strip()
            else:
                pattern = input("Enter file path or glob pattern: ").strip()
            if not pattern:
                print("No path provided.")
                return None
            config["last_provider_choice"] = idx
            config["last_pattern"] = pattern
            _save_config(config, args.config)
            return config, pattern

        if args.interactive and not args.dry_run:
            parser.error("--interactive must be used with --dry-run")

        # Handle verbosity levels
        if args.very_verbose:
            verbose_level = 2  # Very verbose
        elif args.verbose:
            verbose_level = 1  # Basic verbose
        else:
            verbose_level = 0  # No verbose output

        if not args.pattern:
            interactive = interactive_run()
            if not interactive:
                return 1
            config, pattern = interactive
        else:
            config = get_config(args.config)
            pattern = args.pattern

        if config.get("cleanup_junk_files", True):
            _cleanup_junk_files(pattern)

        if args.ollama_model:
            config["ollama_model"] = args.ollama_model
            config["default_provider"] = "ollama"

        if args.ollama_select:
            models = list_ollama_models(config)
            if not models:
                print("No Ollama models found.")
                return 1
            print("Select an Ollama model:")
            for idx, name in enumerate(models, start=1):
                print(f"{idx}. {name}")
            choice = input("Enter number: ").strip()
            if not choice.isdigit() or not (1 <= int(choice) <= len(models)):
                print("Invalid selection.")
                return 1
            config["ollama_model"] = models[int(choice) - 1]
            config["default_provider"] = "ollama"
        if args.cover:
            config["cover_enabled"] = True
        files = collect_files(pattern)
        planned_renames: list[dict] = []
        report_entries: list[dict] = []

        def add_report_entry(
            status: str,
            original_path: str,
            final_path: str = "",
            suggestion: str = "",
            error: str = "",
            provider: str = "",
            model_name: str = "",
        ) -> None:
            title_for_score = os.path.basename(final_path) if final_path else suggestion
            score, flags = evaluate_title(title_for_score, config) if title_for_score else (0.0, [])
            report_entries.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "status": status,
                    "original_path": os.path.abspath(original_path),
                    "final_path": os.path.abspath(final_path) if final_path else "",
                    "suggestion": suggestion,
                    "provider": provider,
                    "model": model_name,
                    "score": score,
                    "flags": ",".join(flags),
                    "error": error,
                }
            )

        # List to hold tempdir references in debug mode to prevent garbage collection
        debug_tempdirs = []

        for file_path in files:
            print(f"Processing file: {file_path}")
            file_config = apply_batch_rules(config, file_path)
            dispatcher = FileDispatcher(file_config, debug=args.debug)
            provider, model_name = _provider_model(file_config)
            _, ext = os.path.splitext(file_path)
            is_svg = ext.lower() == ".svg"
            tempdir = None
            png_path = None
            if is_svg:
                if args.debug:
                    # Create a regular temp directory that won't auto-cleanup
                    tempdir_path = tempfile.mkdtemp(prefix="onoma_svg_")
                    tempdir = type(
                        "TempDir", (), {"name": tempdir_path, "cleanup": lambda: None}
                    )()
                    print(f"[DEBUG] Created tempdir for SVG: {tempdir.name}")
                else:
                    tempdir = tempfile.TemporaryDirectory()
                try:
                    png_path = convert_svg_to_png(file_path, tempdir.name)
                    if args.debug:
                        print(f"[DEBUG] Created PNG: {png_path}")
                except Exception as e:
                    print(f"[SVG ERROR] Could not convert {file_path} to PNG: {e}")
                    if tempdir is not None and not args.debug:
                        tempdir.cleanup()
                    continue
            result = dispatcher.process(file_path)
            if not result:
                add_report_entry(
                    "skipped",
                    file_path,
                    provider=provider,
                    model_name=model_name,
                )
                if tempdir is not None and not args.debug:
                    tempdir.cleanup()
                continue
            try:
                if is_svg and png_path:
                    # Always use PNG for all LLM input for SVGs
                    all_image_suggestions = []
                    img_suggestions = get_suggestions(
                        "",
                        verbose_level=verbose_level,
                        file_path=png_path,
                        config=file_config,
                    )
                    if img_suggestions:
                        all_image_suggestions.append(img_suggestions)
                    flat_image_suggestions = [
                        s for sublist in all_image_suggestions for s in sublist
                    ]
                    md_suggestions = get_suggestions(
                        result
                        if isinstance(result, str)
                        else result.get("markdown", ""),
                        verbose_level=verbose_level,
                        file_path=png_path,
                        config=file_config,
                    )
                    guidance = "\n".join(flat_image_suggestions)
                    final_prompt = (
                        "You have previously suggested the following file names for each "
                        "page/slide/image of the file:\n"
                        f"{guidance}\n"
                        "Now, based on the full document content (markdown below) and the "
                        "above suggestions, generate 3 final file name suggestions that best "
                        "represent the entire file.\n"
                        f"MARKDOWN:\n{result if isinstance(result, str) else result.get('markdown', '')}"
                    )
                    final_suggestions = get_suggestions(
                        final_prompt,
                        verbose_level=verbose_level,
                        file_path=png_path,
                        config=file_config,
                    )
                    suggestions = (
                        final_suggestions or md_suggestions or flat_image_suggestions
                    )
                    if suggestions:
                        new_name = suggestions[0]
                        isbn = extract_isbn(
                            result if isinstance(result, str) else result.get("markdown", "")
                        )
                        planned_renames.append(
                            {
                                "original_path": file_path,
                                "suggestion": new_name,
                                "provider": provider,
                                "model": model_name,
                                "isbn": isbn,
                            }
                        )
                    else:
                        add_report_entry(
                            "skipped",
                            file_path,
                            provider=provider,
                            model_name=model_name,
                        )
                else:
                    # Non-SVG logic unchanged
                    if (
                        isinstance(result, dict)
                        and "markdown" in result
                        and "images" in result
                    ):
                        images = result["images"]
                        pdf_tempdir = result.get("tempdir")
                        if pdf_tempdir is not None and args.debug:
                            debug_tempdirs.append(
                                pdf_tempdir
                            )  # Prevent GC in debug mode
                            print(
                                f"[DEBUG] Created tempdir for PDF/PPTX: {pdf_tempdir.name}"
                            )
                            for img_path in images:
                                print(f"[DEBUG] Created image: {img_path}")
                            # Check if markdown file was created
                            markdown_path = os.path.join(
                                pdf_tempdir.name, "extracted_content.md"
                            )
                            if os.path.exists(markdown_path):
                                print(f"[DEBUG] Created markdown: {markdown_path}")
                        all_image_suggestions = []
                        for img_path in images:
                            img_suggestions = get_suggestions(
                                "",
                                verbose_level=verbose_level,
                                file_path=img_path,
                                config=file_config,
                            )
                            if img_suggestions:
                                all_image_suggestions.append(img_suggestions)
                        flat_image_suggestions = [
                            s for sublist in all_image_suggestions for s in sublist
                        ]
                        md_file_path = images[0] if len(images) > 0 else file_path
                        md_suggestions = get_suggestions(
                            result["markdown"],
                            verbose_level=verbose_level,
                            file_path=md_file_path,
                            config=file_config,
                        )
                        guidance = "\n".join(flat_image_suggestions)
                        final_prompt = (
                            "You have previously suggested the following file names for each "
                            "page/slide/image of the file:\n"
                            f"{guidance}\n"
                            "Now, based on the full document content (markdown below) and the "
                            "above suggestions, generate 3 final file name suggestions that best "
                            "represent the entire file.\n"
                            f"MARKDOWN:\n{result['markdown']}"
                        )
                        final_suggestions = get_suggestions(
                            final_prompt,
                            verbose_level=verbose_level,
                            file_path=md_file_path,
                            config=file_config,
                        )
                        suggestions = (
                            final_suggestions
                            or md_suggestions
                            or flat_image_suggestions
                        )
                        if suggestions:
                            new_name = suggestions[0]
                            isbn = extract_isbn(result.get("markdown", ""))
                            planned_renames.append(
                                {
                                    "original_path": file_path,
                                    "suggestion": new_name,
                                    "provider": provider,
                                    "model": model_name,
                                    "isbn": isbn,
                                }
                            )
                        else:
                            add_report_entry(
                                "skipped",
                                file_path,
                                provider=provider,
                                model_name=model_name,
                            )
                    elif isinstance(result, dict) and "tempdir" in result:
                        # Handle files with tempdir but no images (text files, Word docs, etc. in debug mode)
                        file_tempdir = result.get("tempdir")
                        if file_tempdir is not None and args.debug:
                            debug_tempdirs.append(
                                file_tempdir
                            )  # Prevent GC in debug mode
                            file_type = (
                                os.path.splitext(file_path)[1].upper().lstrip(".")
                            )
                            print(
                                f"[DEBUG] Created tempdir for {file_type}: {file_tempdir.name}"
                            )
                            # Check if markdown file was created
                            markdown_path = os.path.join(
                                file_tempdir.name, "extracted_content.md"
                            )
                            if os.path.exists(markdown_path):
                                print(f"[DEBUG] Created markdown: {markdown_path}")

                        content = result.get("markdown", "")
                        suggestions = get_suggestions(
                            content,
                            verbose_level=verbose_level,
                            file_path=file_path,
                            config=file_config,
                        )
                        if suggestions:
                            new_name = suggestions[0]
                            isbn = extract_isbn(result.get("markdown", ""))
                            planned_renames.append(
                                {
                                    "original_path": file_path,
                                    "suggestion": new_name,
                                    "provider": provider,
                                    "model": model_name,
                                    "isbn": isbn,
                                }
                            )
                    else:
                        content = result
                        suggestions = get_suggestions(
                            content,
                            verbose_level=verbose_level,
                            file_path=file_path,
                            config=file_config,
                        )
                        if suggestions:
                            new_name = suggestions[0]  # Use first suggestion in Phase 1
                            isbn = extract_isbn(content)
                            planned_renames.append(
                                {
                                    "original_path": file_path,
                                    "suggestion": new_name,
                                    "provider": provider,
                                    "model": model_name,
                                    "isbn": isbn,
                                }
                            )
                        else:
                            add_report_entry(
                                "skipped",
                                file_path,
                                provider=provider,
                                model_name=model_name,
                            )
            finally:
                # Clean up SVG tempdir if not in debug mode
                if tempdir is not None:
                    if args.debug:
                        print(f"[DEBUG] Preserving SVG tempdir: {tempdir.name}")
                    else:
                        tempdir.cleanup()

                # Clean up PDF tempdir if not in debug mode (if it exists in this scope)
                if "pdf_tempdir" in locals() and pdf_tempdir is not None:
                    if args.debug:
                        print(f"[DEBUG] Preserving PDF tempdir: {pdf_tempdir.name}")
                    else:
                        pdf_tempdir.cleanup()

                # Clean up file tempdir if not in debug mode (if it exists in this scope)
                if "file_tempdir" in locals() and file_tempdir is not None:
                    if args.debug:
                        file_type = os.path.splitext(file_path)[1].upper().lstrip(".")
                        print(
                            f"[DEBUG] Preserving {file_type} tempdir: {file_tempdir.name}"
                        )
                    else:
                        file_tempdir.cleanup()

        plan = _build_rename_plan(planned_renames, config)

        if args.dry_run:
            for entry in plan:
                action = entry["action"]
                src = entry["original_path"]
                dst = entry["target_path"]
                if action == "rename":
                    print(f"{os.path.basename(src)} --dry-run-> {os.path.basename(dst)}")
                    add_report_entry(
                        "dry_run",
                        src,
                        dst,
                        entry.get("suggestion", ""),
                        provider=entry.get("provider", ""),
                        model_name=entry.get("model", ""),
                    )
                    if config.get("cover_enabled", False):
                        add_report_entry(
                            "cover_pending",
                            src,
                            dst,
                            entry.get("suggestion", ""),
                            provider=entry.get("provider", ""),
                            model_name=entry.get("model", ""),
                        )
                elif action == "duplicate":
                    print(f"{os.path.basename(src)} --dry-run-> DUPLICATE {os.path.basename(dst)}")
                    add_report_entry(
                        "dry_run_duplicate",
                        src,
                        dst,
                        entry.get("suggestion", ""),
                        provider=entry.get("provider", ""),
                        model_name=entry.get("model", ""),
                    )
                elif action == "fuzzy_duplicate":
                    add_report_entry(
                        "fuzzy_duplicate",
                        src,
                        dst,
                        entry.get("suggestion", ""),
                        provider=entry.get("provider", ""),
                        model_name=entry.get("model", ""),
                    )
                else:
                    add_report_entry(
                        "skipped",
                        src,
                        dst,
                        entry.get("suggestion", ""),
                        provider=entry.get("provider", ""),
                        model_name=entry.get("model", ""),
                    )

            if args.interactive and plan:
                confirm = input("\nProceed with these renames? [y/N]: ").strip().lower()
                if confirm == "y":
                    for entry in plan:
                        action = entry["action"]
                        src = entry["original_path"]
                        dst = entry["target_path"]
                        if action == "rename":
                            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                            shutil.move(src, dst)
                            add_report_entry(
                                "renamed",
                                src,
                                dst,
                                entry.get("suggestion", ""),
                                provider=entry.get("provider", ""),
                                model_name=entry.get("model", ""),
                            )
                            if config.get("cover_enabled", False):
                                cover = cover_workflow(dst, dst, entry.get("isbn"), config)
                                if cover:
                                    add_report_entry(cover.get("status", "cover"), src, dst)
                        elif action == "duplicate":
                            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                            shutil.move(src, dst)
                            add_report_entry(
                                "duplicate",
                                src,
                                dst,
                                entry.get("suggestion", ""),
                                provider=entry.get("provider", ""),
                                model_name=entry.get("model", ""),
                            )
                        elif action == "fuzzy_duplicate":
                            add_report_entry(
                                "fuzzy_duplicate",
                                src,
                                dst,
                                entry.get("suggestion", ""),
                                provider=entry.get("provider", ""),
                                model_name=entry.get("model", ""),
                            )
                        else:
                            add_report_entry(
                                "skipped",
                                src,
                                dst,
                                entry.get("suggestion", ""),
                                provider=entry.get("provider", ""),
                                model_name=entry.get("model", ""),
                            )
                else:
                    print("Aborted. No files were renamed.")
        else:
            for entry in plan:
                action = entry["action"]
                src = entry["original_path"]
                dst = entry["target_path"]
                if action == "rename":
                    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                    shutil.move(src, dst)
                    add_report_entry(
                        "renamed",
                        src,
                        dst,
                        entry.get("suggestion", ""),
                        provider=entry.get("provider", ""),
                        model_name=entry.get("model", ""),
                    )
                    if config.get("cover_enabled", False):
                        cover = cover_workflow(dst, dst, entry.get("isbn"), config)
                        if cover:
                            add_report_entry(cover.get("status", "cover"), src, dst)
                elif action == "duplicate":
                    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                    shutil.move(src, dst)
                    add_report_entry(
                        "duplicate",
                        src,
                        dst,
                        entry.get("suggestion", ""),
                        provider=entry.get("provider", ""),
                        model_name=entry.get("model", ""),
                    )
                elif action == "fuzzy_duplicate":
                    add_report_entry(
                        "fuzzy_duplicate",
                        src,
                        dst,
                        entry.get("suggestion", ""),
                        provider=entry.get("provider", ""),
                        model_name=entry.get("model", ""),
                    )
                else:
                    add_report_entry(
                        "skipped",
                        src,
                        dst,
                        entry.get("suggestion", ""),
                        provider=entry.get("provider", ""),
                        model_name=entry.get("model", ""),
                    )

        if args.rename_folders or config.get("rename_folders", False):
            _rename_folders(files, config, add_report_entry)

        report_enabled = config.get("report_enabled", True) and not args.no_report
        if report_enabled and report_entries:
            report_format = args.report_format or config.get("report_format", "jsonl")
            report_path = _build_report_path(config, report_format, args.report)
            _write_report(report_entries, report_path, report_format)
            config["last_report_path"] = report_path
            _save_config(config, args.config)
            print(f"Report written to: {report_path}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user (Ctrl+C). Exiting gracefully.")
        return 130
    except SystemExit:
        # Allow normal sys.exit() and argparse exits without stack trace
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    return 0


def save_default_config():
    """Save default configuration to ~/.onomarc"""
    config_path = os.path.expanduser("~/.onomarc")
    # Ensure llm_model is in the main section and not in markitdown
    config = DEFAULT_CONFIG.copy()
    if "markitdown" in config and "llm_model" in config["markitdown"]:
        del config["markitdown"]["llm_model"]
    config["llm_model"] = config.get("llm_model", "gpt-4o")
    config["image_prompt"] = config.get("image_prompt", "")
    config["min_filename_words"] = config.get("min_filename_words", 5)
    config["max_filename_words"] = config.get("max_filename_words", 15)
    try:
        with open(config_path, "w") as f:
            toml.dump(config, f)
    except Exception as e:
        print(f"Error saving default config: {e}")
        sys.exit(1)


def _save_config(config: dict, config_path: str | None) -> None:
    """Persist config to the default path or a custom one."""
    path = os.path.expanduser(config_path) if config_path else os.path.expanduser("~/.onomarc")
    try:
        with open(path, "w") as f:
            toml.dump(config, f)
    except Exception as e:
        print(f"Error saving config: {e}")
        sys.exit(1)


def console_script():
    """Entry point for console_scripts."""
    sys.exit(main())


if __name__ == "__main__":
    console_script()
