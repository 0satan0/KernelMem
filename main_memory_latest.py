# main.py
from __future__ import annotations
import argparse
import re
import random
import time
import json
import csv
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
# Auto-detect GPU platform and use appropriate profiler
from gpu_platform import is_amd_gpu, is_nvidia_gpu

if is_amd_gpu():
    # Use ROCm profiler for AMD GPUs
    from run_rocm_profiler import profile_bench, load_rocm_metrics as load_ncu_metrics, metrics_to_prompt
    from run_rocm_profiler import profile_bench as nsys_profile_bench, load_nsys_stats
    print("[Info] Using ROCm profiler for AMD GPU")
elif is_nvidia_gpu():
    # Use NVIDIA profiler for NVIDIA GPUs
    from run_ncu_memory import profile_bench, load_ncu_metrics, metrics_to_prompt
    from run_nsys import profile_bench as nsys_profile_bench, load_nsys_stats
    print("[Info] Using NVIDIA profiler for NVIDIA GPU")
else:
    # Fallback - try NVIDIA (will fail gracefully if not available)
    try:
        from run_ncu_memory import profile_bench, load_ncu_metrics, metrics_to_prompt
        from run_nsys import profile_bench as nsys_profile_bench, load_nsys_stats
    except ImportError as e:
        print(f"[Warning] No profiler available: {e}")
        # Create dummy functions
        profile_bench = None
        load_ncu_metrics = None
        metrics_to_prompt = None
        nsys_profile_bench = None
        load_nsys_stats = None
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from agents.query_server import query_server
from prompts.generate_custom_cuda_memory import build_seed_prompt, default_system_prompt
from prompts.judger_compilation_timeout import build_compilation_timeout_prompts
from utils.compile_and_run import compare_and_bench
from utils.kernel_io import extract_code_block, save_kernel_code, extract_json, extract_cuda_kernel_names
from scripts.individual import KernelIndividual  # adjust path if needed
from prompts.error_memory import build_error_prompt
from prompts.optimization_memory_latest import build_optimization_prompt
from prompts.judger_repair_memory import build_correctness_prompts
from prompts.judger_optimization_memory_latest import build_judger_optimization_prompts
_INVOCATION_SPLITTER = "Invoked with:"

def _sanitize_error_message(exc: Exception) -> str:
    """Strip pybind's large‑tensor printouts and keep only the key error text."""
    msg = str(exc)
    if _INVOCATION_SPLITTER in msg:
        msg = msg.split(_INVOCATION_SPLITTER, 1)[0].rstrip()
    return msg

# ------------------------- CLI -------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Single-LLM self-iterative kernel generation/optimization")
    p.add_argument(
        "arch_py",
        type=Path,
        help="Path to a single task .py file OR a directory containing many tasks (.py)",
    )
    # p.add_argument("--gpu", default="Quadro RTX 6000", help="GPU name in prompt spec")
    p.add_argument("--gpu", default="A100-80GB", help="GPU name in prompt spec")
    p.add_argument("--server_type", default="openai", help="LLM provider (local, openai, deepseek, vllm, etc.)")
    p.add_argument("--server_address", default="localhost", help="LLM server address (for vllm/sglang)")
    p.add_argument("--server_port", type=int, default=8000, help="LLM server port (for vllm/sglang)")
    p.add_argument("--model_name", default="gpt-5.1-chat", help="LLM model")
    p.add_argument("--round", "-G", type=int, default=10, help="Number of generations per task")
    p.add_argument("--work_dir", type=Path, default=Path("run"), help="Output root directory")
    p.add_argument("--device", type=int, default=0, help="CUDA device index for benchmarking")
    p.add_argument("--warmup", type=int, default=25, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=100, help="Timed iterations per benchmark")
    p.add_argument("--tol", type=float, default=1e-2, help="Max |err| tolerated")
    p.add_argument("--max_tokens", type=int, default=16384, help="LLM max new tokens")
    p.add_argument("--temperature", type=float, default=1, help="LLM temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="LLM top_p")
    # multi-task controls
    p.add_argument("--first_n", type=int, default=0,
                   help="When arch_py is a directory, take the first N tasks (sorted)")
    p.add_argument(
        "--start_from",
        type=int,
        default=1,
        help="1-based index in the sorted task list to start from (only applies when using --first_n)",
    )
    p.add_argument("--num_tasks", type=int, default=1,
                   help="When sampling, how many tasks to pick (if >0 and first_n=0)")
    p.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for sampling (0 = time)")
    p.add_argument("--filter_from_summary", type=Path, default=None,
                   help="Path to summary.json file. If provided, only tasks with best_runnable=false will be selected from this summary.")
    
    p.add_argument("--subproc_id", type=int, default=0, help="Identifier for sub-process (e.g., when running multiple in parallel)")
    
    return p


# ---------------------- naming helpers -----------------
def _slugify_tag(text: str, max_len: int = 80) -> str:
    """Collapse a string into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if max_len > 0:
        slug = slug[:max_len]
    return slug or "unknown"


def _build_run_tag(server_type: str, model_name: str) -> str:
    server_tag = _slugify_tag(server_type)
    model_tag = _slugify_tag(model_name)
    return f"{server_tag}_{model_tag}"


# ---------------------- small utils --------------------
def _last_n_lines(text: str, n: int = 150) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_full_cuda_source(text: str) -> str:
    """Extract CUDA source from a Python or markdown-like file.

    Order:
      1) ```cuda ... ``` fenced code
      2) source = \"\"\" ... \"\"\"
      3) fallback: raw text
    """
    m = re.search(r"```cuda\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"source\s*=\s*([\"']{3})(.*?)(?:\1)", text, flags=re.DOTALL)
    if m:
        return m.group(2).strip()
    return text.strip()


def _build_history_block(code_dir: Path, keep_last: int = 10) -> str:
    """Collect the CUDA `source` of the most recent *keep_last* kernel files from code_dir."""
    if not code_dir.exists():
        return "## Existing kernels\n(None yet)\n"

    files: List[Path] = sorted(
        list(code_dir.glob("*.py")) + list(code_dir.glob("*.cu")),
        key=lambda p: p.stat().st_mtime,
    )[-keep_last:]

    if not files:
        return "## Existing kernels\n(None yet)\n"

    snippets: List[str] = []
    for idx, p in enumerate(files, 1):
        try:
            cuda_src = _extract_full_cuda_source(_read_text(p))
        except Exception:
            cuda_src = "(failed to read/extract)"
        snippets.append(f"### Kernel {idx} · {p.name}\n```cuda\n{cuda_src}\n```")

    return "## Existing kernels\n" + "\n\n".join(snippets) + "\n"


# ------------------- LLM & eval steps ------------------
def _make_llm_caller(args):

    def call_llm(
        prompt: str,
        sys_prompt: Optional[str] = None,
        log_path: Optional[Path] = None,
        call_type: str = "unknown",
        round_idx: int = -1,
    ) -> str:
        sp = default_system_prompt if sys_prompt is None else sys_prompt
        res = query_server(
            prompt=prompt,
            system_prompt=sp,
            server_type=args.server_type,
            model_name=args.model_name,
        max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            server_address=args.server_address,
            server_port=args.server_port,
            log_path=str(log_path) if log_path else None,
            call_type=call_type,
            round_idx=round_idx,
        )
        if isinstance(res, list):
            return res[0] if res else ""
        return str(res)
    return call_llm


def _extract_kernel_from_optimization_reply(raw: str) -> str:
    """Extract kernel code from optimization reply that contains mapping + kernel sections.
    
    The optimization reply format is:
    - Section A: Checklist evidence (plan-to-code mapping)
    - Delimiter: === KERNEL CODE STARTS BELOW ===
    - Section B: Kernel code block (```python ... ```)
    
    Returns the kernel code block only.
    """
    delimiter = "=== KERNEL CODE STARTS BELOW ==="
    delimiter_idx = raw.find(delimiter)
    
    if delimiter_idx != -1:
        # Extract everything after the delimiter
        kernel_section = raw[delimiter_idx + len(delimiter):].strip()
        # Extract the first code block from the kernel section
        code = extract_code_block(kernel_section)
        return code
    else:
        # Fallback: if no delimiter found, try to extract the last code block (assuming mapping doesn't have code blocks)
        # This handles cases where LLM didn't follow the format exactly
        code = extract_code_block(raw)
        return code

def _llm_to_kernel(
    prompt: str,
    code_dir: Path,
    call_llm,
    io_dir: Path,
    round_idx,
    sys_prompt: Optional[str] = None,   # New: optional system prompt
    log_path: Optional[Path] = None,
    call_type: str = "unknown",
) -> KernelIndividual:
    """LLM → code → save → KernelIndividual (no evaluation)."""
    raw = call_llm(
        prompt,
        sys_prompt=sys_prompt,
        log_path=log_path,
        call_type=call_type,
        round_idx=round_idx,
    )
    # Ensure io_dir exists before writing
    io_dir.mkdir(parents=True, exist_ok=True)
    reply_file = io_dir / f"{round_idx}_raw_reply.txt"
    reply_file.write_text(raw, encoding="utf-8")
    
    # For optimization calls, extract kernel code using delimiter-aware extraction
    if call_type == "optimization":
        code = _extract_kernel_from_optimization_reply(raw)
    else:
        # For other call types (seed, repair, etc.), use standard extraction
        code = extract_code_block(raw) or raw  # fallback
    
    path = save_kernel_code(code, code_dir)
    ind = KernelIndividual(code)
    ind.code_path = path  # type: ignore[attr-defined]
    return ind

# ================== Top-level worker: MUST live at module top level, not inside another function ==================
def _bench_worker_entry(test_py: str,
                        ref_py: str,
                        device_idx: int,
                        warmup: int,
                        repeat: int,
                        tol: float,
                        conn) -> None:
    """
    Subprocess entry: set GPU, call compare_and_bench, and send result or error
    back to the parent via a Pipe. Note: we pass string paths here to avoid
    non-picklable objects.
    """
    import torch
    from pathlib import Path
    from utils.compile_and_run import CompilationError, CompilationTimeoutError, AccuracyError

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_idx)

        res = compare_and_bench(
            ref_py=Path(ref_py),
            test_py=Path(test_py),
            device_idx=device_idx,
            warmup=warmup,
            repeat=repeat,
            tol=tol,
        )
        conn.send(("ok", res))
    except Exception as e:
        # Clean the error message if helper is available; otherwise fall back to str(e)
        try:
            cleaned = _sanitize_error_message(e)
            msg = _last_n_lines(cleaned)
        except Exception:
            msg = str(e)

        if isinstance(e, CompilationTimeoutError):
            err_type = "CompilationTimeoutError"
        elif isinstance(e, CompilationError):
            err_type = "CompilationError"
        elif isinstance(e, AccuracyError):
            err_type = "AccuracyError"
        else:
            err_type = e.__class__.__name__

        conn.send(("err", {"type": err_type, "message": msg}))
    finally:
        # Try to sync at the end so errors surface within this round
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device_idx)
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


# ================== Top-level worker for preloading kernel (must be at module level for pickling) ==================
def _preload_worker(test_kernel_path: str, conn) -> None:
    """
    Subprocess entry: preload kernel to ensure .so is cached.
    This MUST be at module level to be picklable by multiprocessing.
    """
    try:
        import sys as _sys
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "preload_test_kernel_temp", 
            test_kernel_path
        )
        if spec and spec.loader:
            preload_mod = importlib.util.module_from_spec(spec)
            _sys.modules[spec.name] = preload_mod
            spec.loader.exec_module(preload_mod)
            conn.send(("ok", "loaded"))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        try:
            conn.close()
        except:
            pass


# ================== Keep original behavior: _bench_and_score (uses spawn + top-level worker) ==================
def _bench_and_score(
    ind: KernelIndividual,
    *,
    ref_py: Path,
    device_idx: int,
    warmup: int,
    repeat: int,
    tol: float,
    phase: str = "seed",
    metrics_dir: Path | None = None,
) -> None:
    """
    Benchmark and update the individual's metrics/score; on exception, fill in
    failure info and save metrics (if a directory is provided).
    Same functionality as the original version, but runs compare_and_bench in a
    **spawned subprocess** to isolate the CUDA context.
    """
    import torch
    from multiprocessing import get_context

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    # Only pass picklable arguments (e.g., string paths)
    p = ctx.Process(
        target=_bench_worker_entry,
        args=(
            str(ind.code_path),  # type: ignore[attr-defined]
            str(ref_py),
            device_idx,
            warmup,
            repeat,
            tol,
            child_conn,
        ),
    )
    p.start()
    # Parent does not use the child end
    try:
        child_conn.close()
    except Exception:
        pass

    # ========== 添加超时保护：20 分钟（10 分钟编译 + 10 分钟测试）==========
    # Wait for child with timeout
    timeout_occurred = False
    p.join(timeout=1200)  # 20 minutes
    
    # Check if process is still alive after timeout
    if p.is_alive():
        print(f"[{phase}] WARNING: Subprocess timed out after 20 minutes, terminating...", flush=True)
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            print(f"[{phase}] WARNING: Subprocess did not terminate, killing...", flush=True)
            p.kill()
            p.join()
        timeout_occurred = True
    
    payload = None
    try:
        if timeout_occurred:
            # Don't try to receive from a terminated process
            payload = ("err", {"type": "TimeoutError", "message": "Compilation or execution exceeded 20 minute timeout. This may indicate:\n1. GPU runtime errors (e.g., illegal memory access, out-of-bounds indexing) causing the process to hang\n2. Extremely poor performance due to low GPU occupancy or resource conflicts\n3. Infinite loops or deadlocks in the kernel code\n4. Compilation taking too long due to complex template metaprogramming\n\nPlease investigate:\n- Check array indexing and boundary conditions\n- Verify memory access patterns are valid\n- Reduce per-thread memory usage to improve occupancy"})
        elif parent_conn.poll():
            payload = parent_conn.recv()
    except EOFError:
        # 子进程可能在发送错误信息前崩溃（如 CUDA 上下文损坏）
        # 这种情况下 payload 保持为 None，后续会处理为错误
        pass
    except Exception as e:
        # 其他连接错误也捕获，避免程序崩溃
        print(f"Warning: Failed to receive payload from child process: {e}")
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass

    # —— Update metrics/score based on child payload (same logic as before) ——
    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] in ("ok", "err"):
        tag, data = payload
        if tag == "ok":
            metrics = data
            metrics["runnable"] = True
            metrics["phase"] = phase
            speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
            metrics["score"] = speedup

            ind.metrics = metrics
            ind.score = speedup
            print(f"[{phase}] score={speedup:.4f}", flush=True)

            # # === Optional: on successful compile+run, copy code to root/test_kernel.py ===
            # try:
            #     from pathlib import Path as _Path
            #     import shutil as _shutil
            #     root_dir = _Path(__file__).resolve().parent
            #     dst = root_dir / "test_kernel.py"
            #     src = _Path(ind.code_path)  # type: ignore[arg-type]
            #     if src.exists():
            #         _shutil.copy2(src, dst)
            #         print(f"[{phase}] saved successful kernel to: {dst}")
            #     else:
            #         print(f"[{phase}] WARNING: source code file not found: {src}")
            # except Exception as _copy_exc:
            #     print(f"[{phase}] WARNING: failed to save test_kernel.py: {_copy_exc}")

        else:
            err_type = "RuntimeError"
            message = data
            if isinstance(data, dict):
                err_type = data.get("type", err_type) or err_type
                message = data.get("message", message)

            if not isinstance(message, str):
                message = str(message)

            print(f"\033[91mTest Error ({err_type}):\033[0m {message}", flush=True)
            ind.metrics = {
                "runnable": False,
                "phase": phase,
                "error_type": err_type,
                "message": message,
            }
            ind.score = float("-inf")
            print(f"[{phase}] failed. See metrics.message for details.", flush=True)
    else:
        # Subprocess exited unexpectedly with no payload
        ind.metrics = {
            "runnable": False,
            "phase": phase,
            "error_type": "SubprocessCrashed",
            "message": "subprocess exited unexpectedly (no payload received)",
        }
        ind.score = float("-inf")
        print(f"[{phase}] failed. Subprocess crashed.", flush=True)

    # —— As before: try to save metrics regardless of success/failure —— 
    if metrics_dir is not None:
        try:
            saved = ind.save_metrics(metrics_dir)
            print(f"[{phase}] metrics saved to: {saved}", flush=True)
        except Exception as save_exc:
            print(f"[{phase}] WARNING: failed to save metrics: {save_exc}", flush=True)

    # Light cleanup in parent
    # NOTE: 不在父进程中执行 CUDA 操作，避免子进程的 GPU 错误传播到父进程
    # 子进程的 CUDA 上下文是隔离的，父进程不需要也不应该清理
    if torch.cuda.is_available():
        try:
            # 仅清理内存，不做同步操作（避免触发子进程遗留的 CUDA 错误）
            torch.cuda.empty_cache()
        except Exception:
            pass



# ---------------------- task helpers -------------------
def _collect_tasks(maybe_dir: Path) -> List[Path]:
    """If a directory, return all .py files (sorted); if a file, return [file]."""
    if maybe_dir.is_file():
        return [maybe_dir]
    if maybe_dir.is_dir():
        return sorted([p for p in maybe_dir.rglob("*.py") if p.is_file()])
    raise FileNotFoundError(f"{maybe_dir} not found")


def _filter_tasks_from_summary(all_tasks: List[Path], summary_path: Path) -> List[Path]:
    """Filter tasks based on summary.json, keeping only tasks with best_runnable=false.
    
    Args:
        all_tasks: List of all available task paths
        summary_path: Path to summary.json file
        
    Returns:
        Filtered list of task paths that match best_runnable=false tasks in summary.json
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    # Load summary.json
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    
    # Extract task paths with best_runnable=false
    failed_tasks = []
    for task_info in summary_data.get("tasks", []):
        if task_info.get("best_runnable") is False:
            task_path_str = task_info.get("task", "")
            if task_path_str:
                failed_tasks.append(task_path_str)
    
    print(f"[Filter] Found {len(failed_tasks)} tasks with best_runnable=false in summary.json")
    
    # Match failed tasks to actual file paths
    # Convert task paths from summary (e.g., "KernelBench/level2/100_ConvTranspose3d_Clamp_Min_Divide.py" or "19_ReLU")
    # to actual Path objects by matching against all_tasks
    matched_tasks = []
    for failed_task in failed_tasks:
        # Extract the base name (filename without extension)
        # Handle both formats: "KernelBench/level1/19_ReLU.py" and "19_ReLU"
        task_path_obj = Path(failed_task)
        task_base_name = task_path_obj.stem  # filename without extension (e.g., "19_ReLU")
        task_filename_with_ext = task_path_obj.name  # filename with extension if present
        
        # Try multiple matching strategies
        matched = False
        for task_path in all_tasks:
            # Strategy 1: Exact match with extension (handles "KernelBench/level1/19_ReLU.py" -> "19_ReLU.py")
            if task_path.name == task_filename_with_ext:
                matched_tasks.append(task_path)
                matched = True
                break
            # Strategy 2: Match by base name (without extension)
            # This handles cases where summary has "19_ReLU" but file is "19_ReLU.py"
            if task_path.stem == task_base_name:
                matched_tasks.append(task_path)
                matched = True
                break
        
        if not matched:
            print(f"[Filter] WARNING: Could not find task file for '{failed_task}' (searched for base name: '{task_base_name}')")
    
    print(f"[Filter] Matched {len(matched_tasks)} tasks from summary to available task files")
    return sorted(matched_tasks)


def _pick_first_n(tasks: List[Path], n: int) -> List[Path]:
    n = max(1, min(max(n, 0), len(tasks)))
    return tasks[:n]


def _sample_tasks(all_tasks: List[Path], k: int, seed: int | None) -> List[Path]:
    if not all_tasks:
        raise RuntimeError("No .py tasks found.")
    k = max(1, min(k, len(all_tasks)))
    if seed is None or seed == 0:
        seed = int(time.time())
    rng = random.Random(seed)
    return rng.sample(all_tasks, k)


def _plot_scores(save_path: Path, scores: List[float], err_flags: List[bool], title: str):
    """Plot per-round score curve.
    
    - Green circles (o): runnable kernels (err_flags=False)
    - Red squares (s): non-runnable kernels (err_flags=True)
    """
    xs = list(range(len(scores)))
    plt.figure()
    
    # Separate runnable and non-runnable points
    runnable_xs = []
    runnable_ys = []
    non_runnable_xs = []
    non_runnable_ys = []
    
    for x, y, is_error in zip(xs, scores, err_flags):
        if is_error:
            # Non-runnable: red square
            non_runnable_xs.append(x)
            non_runnable_ys.append(y)
        else:
            # Runnable: green circle
            runnable_xs.append(x)
            runnable_ys.append(y)
    
    # Plot runnable kernels as green circles
    if runnable_xs:
        plt.scatter(runnable_xs, runnable_ys, marker="o", color="green", 
                   s=40, alpha=0.7, label="Runnable", zorder=3)
    
    # Plot non-runnable kernels as red squares
    if non_runnable_xs:
        plt.scatter(non_runnable_xs, non_runnable_ys, marker="s", color="red", 
                   s=40, alpha=0.7, label="Non-runnable", zorder=3)
    
    # Draw connecting line for visualization
    plt.plot(xs, scores, linestyle="-", color="gray", alpha=0.3, linewidth=1, zorder=1)
    
    plt.xlabel("Round")
    plt.ylabel("Speedup (ref/test)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _append_usage_totals(log_path: Path) -> Dict[str, int]:
    """Append a totals row to usage.csv and return the summed token counts."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not log_path.exists():
        return totals

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames or not rows:
        return totals

    for row in rows:
        if row.get("call_type") == "sum" or row.get("timestamp") == "Total":
            continue
        for key in totals:
            try:
                totals[key] += int(row.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue

    total_row = {fn: "" for fn in fieldnames}
    for key, value in totals.items():
        if key in total_row:
            total_row[key] = str(value)

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(total_row)

    return totals


# --------------------- single-task run -----------------
def _run_single_task(task_path: Path, args, batch_dir: Path) -> Dict[str, Any]:
    # --- per-task directories under the SAME batch_dir
    task_root = (batch_dir / task_path.stem).resolve()
    
    # Check if this is a level3 task
    is_level3 = "level3" in str(task_path) or "level3" in str(task_path.parent)
    code_dir = task_root / "code"
    eval_dir = task_root / "evaluation"
    fig_dir = task_root / "figures"
    io_dir = eval_dir / "llm_io"
    profile_dir = task_root / "profile"

    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_root / "usage.csv"

    # === Write the contents of task_path into root/ref.py ===
    root_dir = Path(__file__).resolve().parent
    ref_py = root_dir / f"ref_{args.subproc_id}.py"
    test_kernel = root_dir / f"test_kernel_{args.subproc_id}.py"
    bench_py = root_dir / f"bench_ref_inputs_{args.subproc_id}.py"
    content = task_path.read_text(encoding="utf-8")  # read source from task_path
    with open(ref_py, "w", encoding="utf-8") as f:
        f.write(content)
    
    # === Create bench_ref_inputs_{subproc_id}.py from template ===
    if not bench_py.exists():
        template_path = root_dir / "bench_ref_inputs_0.py"
        if template_path.exists():
            template_content = template_path.read_text(encoding="utf-8")
            # Replace hardcoded ref_0.py and test_kernel_0.py with subproc_id versions
            bench_content = template_content.replace("ref_0.py", f"ref_{args.subproc_id}.py")
            bench_content = bench_content.replace("test_kernel_0.py", f"test_kernel_{args.subproc_id}.py")
            with open(bench_py, "w", encoding="utf-8") as f:
                f.write(bench_content)
        else:
            raise FileNotFoundError(f"Template file {template_path} not found. Cannot create {bench_py}")

    call_llm = _make_llm_caller(args)

    current_kernel: Optional[KernelIndividual] = None
    base_kernel: Optional[KernelIndividual] = None  # Base kernel for optimization (updated with strict conditions)
    base_score: float = float("-inf")
    best_kernel: Optional[KernelIndividual] = None  # Best kernel for statistics (updated unconditionally when score is higher)
    best_score: float = float("-inf")
    # Track optimization history: map from round_idx to opt_history_file path
    # This tracks which round's opt history should be updated after repair
    opt_history_files: Dict[int, Path] = {}
    
    # Repair chain tracking: track the first kernel in a repair chain
    # A repair chain starts when a kernel from opt phase fails, and continues until repair succeeds
    # All repair history for kernels in the same chain should be saved in the same folder
    repair_chain_kernel: Optional[KernelIndividual] = None
    
    # Optimization tree: track kernel genealogy
    # Structure: {kernel_name: {parent, speedup, ncu_passed, strategy, phase, round, ...}}
    optimization_tree: Dict[str, Dict[str, Any]] = {}

    scores: List[float] = []
    err_flags: List[bool] = []
    last_score_for_curve = 0.0  # default baseline for plotting on early failures

    for round_idx in range(args.round):
        print(f"[{task_path.name}] Round {round_idx}")

        if round_idx == 0:
            print("[Seed] Generating the initial kernel ...")
            seed_prompt = build_seed_prompt(arch_path=task_path, gpu_name=args.gpu)
            prompt_file = io_dir / f"round{round_idx:03d}_seed_prompt.txt"
            prompt_file.write_text(seed_prompt, encoding="utf-8")
            ind = _llm_to_kernel(seed_prompt, code_dir, call_llm, io_dir,
                                 round_idx, log_path=log_path, call_type="seed")
            _bench_and_score(
                ind,
                ref_py=task_path,
                device_idx=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                tol=args.tol,
                phase="seed",
                metrics_dir=eval_dir,
            )
            
            # Record seed kernel in optimization tree
            if ind and hasattr(ind, 'code_path') and ind.code_path:
                kernel_name = ind.code_path.stem
                runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                speedup = ind.score if (ind.score is not None and runnable) else None
                optimization_tree[kernel_name] = {
                    "parent": None,  # Seed is root
                    "kernel_name": kernel_name,
                    "kernel_path": str(ind.code_path),
                    "speedup": float(speedup) if speedup is not None else None,
                    "runnable": runnable,
                    "ncu_passed": False,  # Seed doesn't go through ncu
                    "phase": "seed",
                    "round": round_idx,
                    "strategy": None,  # Seed has no strategy
                    "method_matched": False,  # Seed doesn't have optimization method matching
                    "timestamp": datetime.now().isoformat(),
                }

        else:
            is_runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False)) if current_kernel else False

            if not is_runnable:
                print("[Repair] start repairing")
                # Check if we need to update opt history after repair
                # If current_kernel was generated in opt phase of this round, we should update opt history
                opt_history_file_to_update = opt_history_files.get(round_idx)
                
                # ========== Create repair history folder and file ==========
                # Repair chain logic: All repairs for kernels in the same chain should be saved in the same folder
                # A repair chain starts when a kernel from opt phase fails, and continues until repair succeeds
                # If there's no active repair chain, start a new one with current_kernel
                if repair_chain_kernel is None:
                    # Start a new repair chain with current_kernel (the first problematic kernel)
                    repair_chain_kernel = current_kernel
                    print(f"[repair] Starting new repair chain with kernel: {repair_chain_kernel.code_path.stem if (repair_chain_kernel and hasattr(repair_chain_kernel, 'code_path') and repair_chain_kernel.code_path) else 'unknown'}")
                
                # Use repair_chain_kernel (the first kernel in the chain) to create repair history folder
                # This ensures all repairs in the same chain are saved in the same folder
                kernel_to_repair_name = None
                kernel_to_repair_path = None
                repair_history_dir = None
                repair_history_file = None
                repair_round_num = 1
                
                if repair_chain_kernel and hasattr(repair_chain_kernel, 'code_path') and repair_chain_kernel.code_path:
                    kernel_to_repair_path = repair_chain_kernel.code_path
                    kernel_to_repair_name = kernel_to_repair_path.stem  # e.g., "kernel_20251225_185242"
                    repair_history_dir = code_dir / kernel_to_repair_name
                    repair_history_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Count existing repair history files to determine repair round number
                    if repair_history_dir.exists():
                        existing_repair_files = sorted(repair_history_dir.glob("repair_round_*.json"),
                                                       key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0)
                        if existing_repair_files:
                            # Get the highest round number and increment
                            last_repair_file = existing_repair_files[-1]
                            try:
                                last_round_str = last_repair_file.stem.split("_")[-1]
                                if last_round_str.isdigit():
                                    repair_round_num = int(last_round_str) + 1
                            except Exception:
                                repair_round_num = len(existing_repair_files) + 1
                    
                    repair_history_file = repair_history_dir / f"repair_round_{repair_round_num:03d}.json"
                    # Debug: Print repair chain info
                    current_kernel_name = current_kernel.code_path.stem if (current_kernel and hasattr(current_kernel, 'code_path') and current_kernel.code_path) else "None"
                    base_kernel_name = base_kernel.code_path.stem if (base_kernel and hasattr(base_kernel, 'code_path') and base_kernel.code_path) else "None"
                    best_kernel_name = best_kernel.code_path.stem if (best_kernel and hasattr(best_kernel, 'code_path') and best_kernel.code_path) else "None"
                    print(f"[repair] Creating repair history for repair chain: {kernel_to_repair_name} (repair round {repair_round_num})")
                    print(f"[repair]   - Repair chain kernel (first in chain): {kernel_to_repair_name}")
                    print(f"[repair]   - Current kernel being repaired: {current_kernel_name}")
                    print(f"[repair]   - Base kernel (for optimization): {base_kernel_name}")
                    print(f"[repair]   - Best kernel (statistics): {best_kernel_name}")
                    if kernel_to_repair_name != current_kernel_name:
                        print(f"[repair]   - ✓ Using repair chain kernel for folder (continuing existing chain)")
                    else:
                        print(f"[repair]   - ✓ Using repair chain kernel for folder (new chain)")
                
                error_log = _last_n_lines(getattr(current_kernel, "metrics", {}).get(
                    "message", "")) if current_kernel else ""

                # ========== Load repair history for the repair chain ==========
                repair_history = []
                if repair_history_dir and repair_history_dir.exists():
                    try:
                        # Read all existing repair history files (excluding the current one being created)
                        existing_repair_files = sorted(repair_history_dir.glob("repair_round_*.json"),
                                                       key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0)
                        for hist_file in existing_repair_files:
                            # Skip the current repair round file (not yet completed)
                            if hist_file == repair_history_file:
                                continue
                            try:
                                hist_data = json.loads(hist_file.read_text(encoding="utf-8"))
                                # Only include completed attempts (with test results)
                                if "test_timestamp" in hist_data or "runnable" in hist_data or "speedup" in hist_data:
                                    repair_history.append(hist_data)
                            except Exception as e:
                                print(f"[repair] Warning: Failed to read repair history from {hist_file}: {e}")
                        
                        if repair_history:
                            print(f"[repair] Loaded {len(repair_history)} previous repair attempts from {repair_history_dir}")
                    except Exception as e:
                        print(f"[repair] Warning: Failed to load repair history: {e}")

                problem_system_prompt, problem_prompt = build_correctness_prompts(error_log=error_log,
                                                                                  arch_path=task_path,
                                                                                  cuda_code=current_kernel.code,
                                                                                  repair_history=repair_history if repair_history else None)
                prompt_file = io_dir / f"round{round_idx:03d}_problem_identify_prompt.txt"
                prompt_file.write_text(problem_prompt, encoding="utf-8")
                raw = call_llm(problem_prompt, problem_system_prompt, log_path=log_path,
                               call_type="problem_identify", round_idx=round_idx)
                io_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                reply_file = io_dir / f"{round_idx}_raw_problem_identify_reply.txt"
                reply_file.write_text(raw, encoding="utf-8")
                problem_json = extract_json(raw)

                repair_prompt = build_error_prompt(
                    old_code=current_kernel.code,
                    error_log=error_log,
                    problem=problem_json,
                    gpu_name=args.gpu,
                )
                prompt_file = io_dir / f"round{round_idx:03d}_repair_prompt.txt"
                prompt_file.write_text(repair_prompt, encoding="utf-8")
                
                # ========== Save repair history before repair attempt ==========
                if repair_history_file:
                    try:
                        repair_history_data = {
                            "round": round_idx,
                            "repair_round": repair_round_num,
                            "kernel_to_repair": str(kernel_to_repair_path) if kernel_to_repair_path else None,
                            "kernel_to_repair_name": kernel_to_repair_name,
                            "error_log": error_log[:1000] if error_log else None,  # Truncate long error logs
                            "problem_identification": problem_json if problem_json else None,
                            "repair_strategy": problem_json.get("repair_strategy") if (problem_json and isinstance(problem_json, dict)) else None,
                            "timestamp": datetime.now().isoformat(),
                            "runnable": None,  # Will be updated after testing
                            "speedup": None,  # Will be updated after testing
                            "test_passed": None,  # Will be updated after testing
                            "repaired_kernel": None,  # Will be updated after testing
                            "test_timestamp": None,  # Will be updated after testing
                        }
                        repair_history_file.write_text(json.dumps(repair_history_data, indent=2, ensure_ascii=False), encoding="utf-8")
                        print(f"[repair] Saved repair history to: {repair_history_file}")
                    except Exception as e:
                        print(f"[repair] Warning: Failed to save repair history: {e}")
                
                ind = _llm_to_kernel(repair_prompt, code_dir, call_llm, io_dir,
                                     round_idx, log_path=log_path, call_type="repair")
                _bench_and_score(
                    ind,
                    ref_py=task_path,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="repair",
                    metrics_dir=eval_dir,
                )
                
                # Record repaired kernel in optimization tree
                if ind and hasattr(ind, 'code_path') and ind.code_path:
                    kernel_name = ind.code_path.stem
                    parent_name = current_kernel.code_path.stem if (current_kernel and hasattr(current_kernel, 'code_path') and current_kernel.code_path) else None
                    runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                    speedup = ind.score if (ind.score is not None and runnable) else None
                    optimization_tree[kernel_name] = {
                        "parent": parent_name,
                        "kernel_name": kernel_name,
                        "kernel_path": str(ind.code_path),
                        "speedup": float(speedup) if speedup is not None else None,
                        "runnable": runnable,
                        "ncu_passed": False,  # Repaired kernels don't go through ncu immediately
                        "phase": "repair",
                        "round": round_idx,
                        "strategy": problem_json if problem_json else None,
                        "method_matched": False,  # Repair doesn't have optimization method matching
                        "timestamp": datetime.now().isoformat(),
                    }
                
                # ========== Update repair history after testing ==========
                if repair_history_file and repair_history_file.exists():
                    try:
                        runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                        speedup = ind.score if (ind.score is not None and runnable) else None
                        repair_history_data = json.loads(repair_history_file.read_text(encoding="utf-8"))
                        repair_history_data["runnable"] = runnable
                        repair_history_data["speedup"] = float(speedup) if speedup is not None else None
                        repair_history_data["test_passed"] = runnable and speedup is not None
                        repair_history_data["repaired_kernel"] = str(getattr(ind, "code_path", None)) if hasattr(ind, "code_path") else None
                        repair_history_data["test_timestamp"] = datetime.now().isoformat()
                        repair_history_file.write_text(json.dumps(repair_history_data, indent=2, ensure_ascii=False), encoding="utf-8")
                        if runnable and speedup is not None:
                            print(f"[repair] Updated repair history: speedup={speedup:.4f}")
                            # Repair chain succeeded, clear it so next problematic kernel starts a new chain
                            repair_chain_kernel = None
                            print(f"[repair] Repair chain completed successfully, cleared repair_chain_kernel")
                        else:
                            print(f"[repair] Updated repair history: repair failed (runnable={runnable}), repair chain continues")
                    except Exception as e:
                        print(f"[repair] Warning: Failed to update repair history: {e}")
                
                # Update opt history after repair if this kernel was from opt phase
                if opt_history_file_to_update and opt_history_file_to_update.exists():
                    try:
                        runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                        speedup = ind.score if (ind.score is not None and runnable) else None
                        if runnable and speedup is not None:
                            opt_history = json.loads(opt_history_file_to_update.read_text(encoding="utf-8"))
                            opt_history["runnable"] = runnable
                            opt_history["speedup"] = float(speedup)
                            opt_history["test_passed"] = True
                            opt_history["test_kernel"] = str(getattr(ind, "code_path", None)) if hasattr(ind, "code_path") else None
                            opt_history["test_timestamp"] = datetime.now().isoformat()
                            opt_history["repaired"] = True  # Mark that this was repaired
                            opt_history_file_to_update.write_text(json.dumps(opt_history, indent=2, ensure_ascii=False), encoding="utf-8")
                            print(f"[repair] Updated opt history after repair: speedup={speedup:.4f}")
                    except Exception as e:
                        print(f"[repair] Warning: Failed to update opt history after repair: {e}")
            else:
                print("Optimizing start")
                # ========== 确定要优化的kernel：应该是base_kernel（满足更新条件的基准kernel）==========
                # 优化阶段应该一直基于base_kernel进行迭代，而不是current_kernel
                # 因为current_kernel可能是上一轮生成的，但不如base_kernel好
                # parent_kernel是base_kernel_temp，只有通过了ncu profiling才认为是真正的base_kernel
                parent_kernel = base_kernel if base_kernel is not None else current_kernel
                
                # 确保test_kernel文件包含parent_kernel（best_kernel_temp）的代码，用于ncu profiling
                if parent_kernel and hasattr(parent_kernel, 'code'):
                    with open(test_kernel, "w", encoding="utf-8") as f:
                        f.write(parent_kernel.code)
                    print(f"[opt] Updated test_kernel with {'base_kernel (temp, needs profiling)' if base_kernel else 'current_kernel'} for ncu profiling")
                
                # 从test_kernel文件提取kernel名称（现在应该是base_kernel_temp）
                kernel_names = extract_cuda_kernel_names(test_kernel)
                print("=============================================================")
                print(f"Detected kernel names: {kernel_names} (from {'base_kernel (temp)' if base_kernel else 'current_kernel'})")
                
                # ========== Helper function to handle compilation timeout repair ==========
                def _handle_compilation_timeout(error_stage: str, error_detail: str, kernel_to_repair: Optional[KernelIndividual]):
                    """Handle compilation timeout by calling repair LLM.
                    
                    Args:
                        error_stage: Stage where timeout occurred (e.g., "Pre-compile", "ncu")
                        error_detail: Detailed error message
                        kernel_to_repair: The kernel that needs to be repaired (parent_kernel in opt phase, current_kernel in repair phase)
                    
                    Returns:
                        KernelCode: The repaired kernel (or failed kernel if repair also fails)
                    """
                    # Determine which kernel to repair
                    # In opt phase: repair parent_kernel (best_kernel_temp)
                    # In repair phase: repair current_kernel (the failed kernel)
                    kernel_being_repaired = kernel_to_repair if kernel_to_repair is not None else current_kernel
                    
                    print(f"\n[{error_stage}] ⚠️  COMPILATION TIMEOUT DETECTED!")
                    print(f"[{error_stage}] Attempting to repair kernel: {kernel_being_repaired.code_path if hasattr(kernel_being_repaired, 'code_path') else 'unknown'}")
                    print(f"[{error_stage}] Timeout suggests code issues (e.g., infinite template expansion).")
                    print(f"[{error_stage}] Initiating compilation timeout repair in current round...\n")
                    
                    # Construct error message
                    error_log = (
                        f"[{error_stage} COMPILATION TIMEOUT]\n"
                        f"Compilation exceeded 10 minute timeout.\n"
                        f"Details: {error_detail}\n\n"
                        "IMPORTANT: This kernel previously compiled successfully, but now times out during recompilation.\n"
                        "This indicates the code has characteristics that cause exponential compile-time behavior.\n\n"
                        "Such as:\n"
                        "1. Infinite template recursion or excessive template instantiation\n"
                        "2. Exponential template expansion with nested templates\n"
                        "3. Excessive constexpr evaluation or compile-time computations\n"
                        "4. Large loop unrolling (#pragma unroll with huge iteration counts)\n"
                        "5. Massive inline expansion (very large __forceinline__ functions)\n"
                        "6. Compiler bugs triggered by specific code patterns\n\n"
                        "etc.\n\n"
                        "Required action: Fix the kernel to reduce compilation complexity."
                    )
                    
                    # Use current_kernel to track the kernel being repaired (avoid confusion with best/parent/test)
                    # Update current_kernel to the kernel being repaired before repair process
                    repair_target_code = kernel_being_repaired.code if kernel_being_repaired and hasattr(kernel_being_repaired, 'code') else ""
                    
                    # Call Judger with SPECIALIZED compilation timeout prompt
                    problem_system_prompt, problem_prompt = build_compilation_timeout_prompts(
                        error_log=error_log,
                        cuda_code=repair_target_code
                    )
                    prompt_file = io_dir / f"round{round_idx:03d}_compilation_timeout_{error_stage.lower()}_analysis.txt"
                    prompt_file.write_text(problem_prompt, encoding="utf-8")
                    
                    raw = call_llm(problem_prompt, problem_system_prompt, log_path=log_path,
                                   call_type=f"compilation_timeout_{error_stage.lower()}_analysis", round_idx=round_idx)
                    io_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                    reply_file = io_dir / f"round{round_idx:03d}_compilation_timeout_{error_stage.lower()}_analysis_reply.txt"
                    reply_file.write_text(raw, encoding="utf-8")
                    problem_json = extract_json(raw)
                    
                    # Call Repair LLM to generate fix
                    repair_prompt = build_error_prompt(
                        old_code=repair_target_code,
                        error_log=error_log,
                        problem=problem_json,
                        gpu_name=args.gpu,
                    )
                    prompt_file = io_dir / f"round{round_idx:03d}_compilation_timeout_{error_stage.lower()}_repair.txt"
                    prompt_file.write_text(repair_prompt, encoding="utf-8")
                    
                    repaired_kernel = _llm_to_kernel(repair_prompt, code_dir, call_llm, io_dir,
                                                     round_idx, log_path=log_path, 
                                                     call_type=f"compilation_timeout_{error_stage.lower()}_repair")
                    
                    # Test the repaired kernel
                    print(f"[{error_stage}] Testing repaired kernel after compilation timeout fix...")
                    _bench_and_score(
                        repaired_kernel,
                        ref_py=task_path,
                        device_idx=args.device,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        tol=args.tol,
                        phase=f"compilation_timeout_{error_stage.lower()}_repair",
                        metrics_dir=eval_dir,
                    )
                    
                    # Record repaired kernel in optimization tree
                    if repaired_kernel and hasattr(repaired_kernel, 'code_path') and repaired_kernel.code_path:
                        kernel_name = repaired_kernel.code_path.stem
                        parent_name = kernel_to_repair.code_path.stem if (kernel_to_repair and hasattr(kernel_to_repair, 'code_path') and kernel_to_repair.code_path) else None
                        runnable = bool(getattr(repaired_kernel, "metrics", {}).get("runnable", False))
                        speedup = repaired_kernel.score if (repaired_kernel.score is not None and runnable) else None
                        optimization_tree[kernel_name] = {
                            "parent": parent_name,
                            "kernel_name": kernel_name,
                            "kernel_path": str(repaired_kernel.code_path),
                            "speedup": float(speedup) if speedup is not None else None,
                            "runnable": runnable,
                            "ncu_passed": False,  # Compilation timeout repairs don't go through ncu
                            "phase": f"compilation_timeout_{error_stage.lower()}_repair",
                            "round": round_idx,
                            "strategy": problem_json,
                            "method_matched": False,  # Compilation timeout repair doesn't have optimization method matching
                            "timestamp": datetime.now().isoformat(),
                        }
                    
                    # Note: repaired_kernel will be assigned to current_kernel at the end of the round
                    # Only update best_kernel if the repaired kernel's score exceeds best_score
                    return repaired_kernel
                
                # ========== 预加载 kernel，确保编译缓存存在，避免 ncu 环境下重新编译 ==========
                precompile_timeout = False
                precompile_error_detail = ""
                
                print("[Pre-compile] Loading kernel to ensure .so is cached before ncu profiling...")
                from multiprocessing import get_context
                
                try:
                    ctx = get_context("spawn")
                    parent_conn, child_conn = ctx.Pipe(duplex=False)
                    p = ctx.Process(target=_preload_worker, args=(str(test_kernel), child_conn))
                    p.start()
                    child_conn.close()
                    
                    # 10 分钟超时
                    p.join(timeout=600)
                    
                    if p.is_alive():
                        print(f"[Pre-compile] Timeout after 10 minutes, terminating preload process...")
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            p.kill()
                            p.join()
                        
                        precompile_timeout = True
                        precompile_error_detail = f"Preload process exceeded 10 minute timeout for kernel: {test_kernel}"
                    else:
                        # 检查结果
                        if parent_conn.poll():
                            status, msg = parent_conn.recv()
                            if status == "ok":
                                print("[Pre-compile] Kernel loaded successfully, .so is now cached")
                            else:
                                print(f"[Pre-compile] Warning: Failed to preload kernel: {msg}")
                                precompile_timeout = True
                                precompile_error_detail = f"Preload failed with error: {msg}"
                        else:
                            print("[Pre-compile] Warning: Preload process exited without sending result")
                    
                    parent_conn.close()
                except Exception as e:
                    print(f"[Pre-compile] Warning: Failed to preload kernel: {e}")
                    precompile_timeout = True
                    precompile_error_detail = f"Preload exception: {e}"
                
                # Handle pre-compile timeout by calling repair
                if precompile_timeout:
                    # Repair parent_kernel (best_kernel_temp) that failed to pre-compile
                    ind = _handle_compilation_timeout("Pre-compile", precompile_error_detail, kernel_to_repair=parent_kernel)
                    # The repaired kernel has been tested by _bench_and_score
                    # It will be assigned to current_kernel at the end of the round
                    # IMPORTANT: If parent_kernel == base_kernel and the repaired kernel passed testing,
                    # we should update base_kernel even if score doesn't exceed base_score,
                    # because the original base_kernel cannot pass pre-compile and is "invalid"
                    if parent_kernel == base_kernel:
                        runnable_repaired = bool(getattr(ind, "metrics", {}).get("runnable", False))
                        score_repaired = ind.score if (ind.score is not None and runnable_repaired) else None
                        if score_repaired is not None:
                            # The repaired kernel passed testing, so it should replace the unprofilable base_kernel
                            print(f"[Pre-compile] Repaired kernel (score={score_repaired:.4f}) passed testing, updating base_kernel even though score < base_score ({base_score:.4f})", flush=True)
                            base_score = score_repaired
                            base_kernel = ind
                            # Also update best_kernel unconditionally if score is higher
                            if score_repaired > best_score:
                                best_score = score_repaired
                                best_kernel = ind
                            with open(test_kernel, "w") as f:
                                f.write(base_kernel.code)
                    # Otherwise, only update base_kernel if the repaired kernel's score exceeds base_score
                    # (handled at the end of the round)
                    # Continue to score tracking and next iteration
                    
                else:
                    # ========== ncu profiling with timeout handling ==========
                    ncu_timeout = False  # Flag to indicate if ncu profiling timed out
                    ncu_error_detail = ""
                    
                    try:
                        # For level3 tasks: use repeat=5 and timeout=30 minutes (1800 seconds)
                        ncu_repeat = 3 if is_level3 else args.repeat
                        ncu_timeout_seconds = 1800 if is_level3 else None  # 30 minutes for level3, None for default
                        
                        if is_level3:
                            print(f"[ncu] Level3 task detected: using repeat={ncu_repeat}, timeout={ncu_timeout_seconds//60} minutes", flush=True)
                        
                        # 明确指定要 profile 的 kernel 文件（parent_kernel 的代码）
                        kernel_file_to_profile = test_kernel  # test_kernel 已经包含了 parent_kernel.code
                        print(f"[ncu] Profiling kernel from file: {kernel_file_to_profile} (parent_kernel: {parent_kernel.code_path if parent_kernel and hasattr(parent_kernel, 'code_path') else 'N/A'})", flush=True)
                        
                        csv_path_str = f"ncu_temp_{args.subproc_id}.csv"
                        csv_path_result = profile_bench(
                            bench_py=f"bench_ref_inputs_{args.subproc_id}.py",
                            kernel_names=kernel_names,  # 传递 kernel 名称，只监控指定的 kernel
                            kernel_file=kernel_file_to_profile,  # 明确指定要 profile 的 kernel 文件
                            out_csv=csv_path_str,
                            device_idx=args.device,
                            repeat=ncu_repeat,
                            timeout_override=ncu_timeout_seconds,
                        )
                        # profile_bench returns the CSV path (already resolved), ensure it's a Path object
                        csv_path = Path(csv_path_result) if csv_path_result else Path(csv_path_str).resolve()
                        # Store csv_path for error handling
                        csv_path_for_errors = csv_path.resolve()
                        
                        # Save ncu profiling results to profile folder (always save, even if parsing fails)
                        # Use parent_kernel's filename to name the ncu csv file
                        ncu_profile_path = None
                        if parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                            import shutil
                            kernel_name = parent_kernel.code_path.stem  # e.g., "kernel_20251229_141824"
                            ncu_profile_path = profile_dir / f"{kernel_name}_ncu.csv"
                            if csv_path.exists() and csv_path.stat().st_size > 0:
                                try:
                                    shutil.copy2(csv_path, ncu_profile_path)
                                    print(f"[ncu] Saved profiling results to: {ncu_profile_path}")
                                except Exception as save_err:
                                    print(f"[ncu] Warning: Failed to save profiling CSV: {save_err}")
                        
                        metrics_df, sections_dict = load_ncu_metrics(csv_path, extra_keep=("Kernel Name", "Block Size", "Grid Size"),
                                                                      name_list=kernel_names, select="last")
                        metrics_block = metrics_to_prompt(metrics_df, sections_dict=sections_dict)
                        
                        # ========== Run nsys profiling to get kernel launch counts ==========
                        nsys_rep_path = None
                        nsys_csv_path = None
                        try:
                            print(f"[nsys] Starting nsys profiling after ncu...", flush=True)
                            nsys_rep_path = nsys_profile_bench(
                                bench_py=f"bench_ref_inputs_{args.subproc_id}.py",
                                kernel_names=kernel_names,
                                kernel_file=kernel_file_to_profile,
                                out_rep=f"nsys_temp_{args.subproc_id}.nsys-rep",
                                device_idx=args.device,
                                timeout=300,  # 5 minutes timeout
                            )
                            # Extract and save launch counts
                            nsys_csv_path = Path(f"nsys_temp_{args.subproc_id}.csv")
                            nsys_df = load_nsys_stats(
                                rep_path=nsys_rep_path,
                                kernel_names=kernel_names,
                                out_csv=nsys_csv_path,
                            )
                            print(f"[nsys] Successfully extracted kernel launch counts", flush=True)
                            
                            # Save nsys results to profile folder
                            if parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                                import shutil
                                kernel_name = parent_kernel.code_path.stem
                                nsys_profile_rep_path = profile_dir / f"{kernel_name}_nsys.nsys-rep"
                                nsys_profile_csv_path = profile_dir / f"{kernel_name}_nsys.csv"
                                if nsys_rep_path.exists():
                                    shutil.copy2(nsys_rep_path, nsys_profile_rep_path)
                                    print(f"[nsys] Saved .nsys-rep to: {nsys_profile_rep_path}")
                                if nsys_csv_path.exists():
                                    shutil.copy2(nsys_csv_path, nsys_profile_csv_path)
                                    print(f"[nsys] Saved .csv to: {nsys_profile_csv_path}")
                        except Exception as nsys_error:
                            print(f"[nsys] Warning: nsys profiling failed: {nsys_error}", flush=True)
                            # Continue without nsys data - kernel_launch_count will fall back to len(rows)
                            nsys_csv_path = None
                        
                        # Update optimization tree: mark this kernel as having passed ncu
                        if parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                            kernel_name = parent_kernel.code_path.stem
                            if kernel_name in optimization_tree:
                                optimization_tree[kernel_name]["ncu_passed"] = True
                                if ncu_profile_path and ncu_profile_path.exists():
                                    optimization_tree[kernel_name]["ncu_profile_path"] = str(ncu_profile_path)
                    except RuntimeError as ncu_error:
                        # Check if it's a timeout error
                        if "timed out" in str(ncu_error).lower():
                            ncu_timeout = True
                            ncu_error_detail = str(ncu_error)
                        else:
                            # Other ncu errors - still save the CSV if it exists (partial results)
                            print(f"[ncu] ERROR: Profiling failed: {ncu_error}")
                            # Try to save partial CSV results if available
                            if parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                                import shutil
                                kernel_name = parent_kernel.code_path.stem
                                # Use the csv_path from profile_bench if available, otherwise try to find it
                                csv_path_temp = csv_path_for_errors if 'csv_path_for_errors' in locals() else Path(f"ncu_temp_{args.subproc_id}.csv").resolve()
                                if csv_path_temp.exists() and csv_path_temp.stat().st_size > 0:
                                    ncu_profile_path = profile_dir / f"{kernel_name}_ncu_error.csv"
                                    try:
                                        shutil.copy2(csv_path_temp, ncu_profile_path)
                                        print(f"[ncu] Saved partial profiling results (error) to: {ncu_profile_path}")
                                    except Exception as save_err:
                                        print(f"[ncu] Warning: Failed to save error CSV: {save_err}")
                                # Mark parent_kernel as not passing ncu
                                if kernel_name in optimization_tree:
                                    optimization_tree[kernel_name]["ncu_passed"] = False
                            print(f"[{task_path.name}] Using previous kernel and continuing")
                            ind = current_kernel
                            scores.append(last_score_for_curve)
                            err_flags.append(True)
                            continue
                    except Exception as ncu_error:
                        print(f"[ncu] ERROR: Unexpected profiling error: {ncu_error}")
                        # Try to save partial CSV results if available
                        if parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                            import shutil
                            kernel_name = parent_kernel.code_path.stem
                            # Use the csv_path from profile_bench if available, otherwise try to find it
                            csv_path_temp = csv_path_for_errors if 'csv_path_for_errors' in locals() else Path(f"ncu_temp_{args.subproc_id}.csv").resolve()
                            if csv_path_temp.exists() and csv_path_temp.stat().st_size > 0:
                                ncu_profile_path = profile_dir / f"{kernel_name}_ncu_error.csv"
                                try:
                                    shutil.copy2(csv_path_temp, ncu_profile_path)
                                    print(f"[ncu] Saved partial profiling results (error) to: {ncu_profile_path}")
                                except Exception as save_err:
                                    print(f"[ncu] Warning: Failed to save error CSV: {save_err}")
                        print(f"[{task_path.name}] Using previous kernel and continuing")
                        ind = current_kernel
                        scores.append(last_score_for_curve)
                        err_flags.append(True)
                        continue
                    
                    # Handle ncu timeout by calling repair
                    # Mark parent_kernel as not passing ncu and save timeout CSV if available
                    if ncu_timeout and parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path:
                        import shutil
                        kernel_name = parent_kernel.code_path.stem
                        # Use the csv_path from profile_bench if available, otherwise try to find it
                        csv_path_temp = csv_path_for_errors if 'csv_path_for_errors' in locals() else Path(f"ncu_temp_{args.subproc_id}.csv").resolve()
                        if csv_path_temp.exists() and csv_path_temp.stat().st_size > 0:
                            ncu_profile_path = profile_dir / f"{kernel_name}_ncu_timeout.csv"
                            try:
                                shutil.copy2(csv_path_temp, ncu_profile_path)
                                print(f"[ncu] Saved profiling results (timeout) to: {ncu_profile_path}")
                            except Exception as save_err:
                                print(f"[ncu] Warning: Failed to save timeout CSV: {save_err}")
                        if kernel_name in optimization_tree:
                            optimization_tree[kernel_name]["ncu_passed"] = False
                    
                    if ncu_timeout:
                        # Repair parent_kernel (best_kernel_temp) that failed ncu profiling
                        # parent_kernel has not passed profiling yet, so it's still best_kernel_temp
                        ind = _handle_compilation_timeout("ncu", ncu_error_detail, kernel_to_repair=parent_kernel)
                        # The repaired kernel has been tested by _bench_and_score
                        # It will be assigned to current_kernel at the end of the round
                        # IMPORTANT: If parent_kernel == base_kernel and the repaired kernel passed testing,
                        # we should update base_kernel even if score doesn't exceed base_score,
                        # because the original base_kernel cannot pass ncu profiling and is "invalid"
                        if parent_kernel == base_kernel:
                            runnable_repaired = bool(getattr(ind, "metrics", {}).get("runnable", False))
                            score_repaired = ind.score if (ind.score is not None and runnable_repaired) else None
                            if score_repaired is not None:
                                # The repaired kernel passed testing, so it should replace the unprofilable base_kernel
                                print(f"[ncu] Repaired kernel (score={score_repaired:.4f}) passed testing, updating base_kernel even though score < base_score ({base_score:.4f})", flush=True)
                                base_score = score_repaired
                                base_kernel = ind
                                # Also update best_kernel unconditionally if score is higher
                                if score_repaired > best_score:
                                    best_score = score_repaired
                                    best_kernel = ind
                                with open(test_kernel, "w") as f:
                                    f.write(base_kernel.code)
                        # Otherwise, only update base_kernel if the repaired kernel's score exceeds base_score
                        # (handled at the end of the round)
                        
                    else:
                        # ========== Normal optimization flow (no timeout) ==========
                        # parent_kernel (base_kernel_temp) has passed ncu profiling
                        # Now we can consider it as a valid base_kernel candidate
                        # Only update base_kernel if the optimization result exceeds base_score (with strict conditions)
                        # parent_kernel was already determined at the start of optimization phase (line ~759)
                        # Get the path for optimization history tracking
                        parent_kernel_path = getattr(parent_kernel, "code_path", None) if parent_kernel else None
                        
                        # Create optimization history directory based on parent kernel name
                        opt_history_dir = None
                        opt_history_file = None
                        optimization_history = []
                        if parent_kernel_path:
                            parent_kernel_name = parent_kernel_path.stem  # e.g., "kernel_20251225_185242"
                            opt_history_dir = code_dir / parent_kernel_name
                            opt_history_dir.mkdir(parents=True, exist_ok=True)
                            opt_history_file = opt_history_dir / f"opt_round_{round_idx:03d}.json"
                            
                            # Read all previous optimization history files from this directory
                            if opt_history_dir.exists():
                                hist_files = sorted(opt_history_dir.glob("opt_round_*.json"), 
                                                   key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0)
                                for hist_file in hist_files:
                                    # Skip the current round's file (not yet completed)
                                    if hist_file == opt_history_file:
                                        continue
                                    try:
                                        hist_data = json.loads(hist_file.read_text(encoding="utf-8"))
                                        # Only include completed attempts (with test results)
                                        if "test_timestamp" in hist_data or "speedup" in hist_data or "test_passed" in hist_data:
                                            optimization_history.append(hist_data)
                                    except Exception as e:
                                        print(f"[opt] Warning: Failed to read optimization history from {hist_file}: {e}")
                            
                            if optimization_history:
                                # Sort by round number to maintain chronological order
                                optimization_history.sort(key=lambda x: x.get("round", 0))
                                print(f"[opt] Loaded {len(optimization_history)} previous optimization attempts from {opt_history_dir}")
                        
                        # Use parent_kernel.code (base_kernel) for judge LLM, not current_kernel
                        # This is the kernel we want to analyze and optimize
                        parent_kernel_code = parent_kernel.code if parent_kernel and hasattr(parent_kernel, 'code') else (current_kernel.code if current_kernel else "")  # type: ignore[union-attr]
                        sys_judge__prompt, judge_prompt = build_judger_optimization_prompts(
                            arch_path=task_path,
                            gpu_name=args.gpu,
                            ncu_metrics_block=metrics_block,
                            metrics_df=metrics_df,  # Pass metrics_df for machine_check
                            cuda_code=parent_kernel_code,  # Use parent_kernel (best_kernel) code
                            optimization_history=optimization_history if optimization_history else None,
                            code_features=None,  # Will be extracted via judge_gate if call_llm is provided
                            call_llm=call_llm,  # Pass call_llm for code_features extraction
                            nsys_csv_path=nsys_csv_path,  # Pass nsys CSV path for kernel_launch_count
                            io_dir=io_dir,  # Pass io_dir for saving machine_check_result JSON
                            round_idx=round_idx,  # Pass round_idx for filename
                        )
                        prompt_file = io_dir / f"round{round_idx:03d}_judge_optimization_prompt.txt"
                        prompt_file.write_text(judge_prompt, encoding="utf-8")
                        raw = call_llm(judge_prompt, sys_judge__prompt, log_path=log_path,
                                       call_type="judge_optimization", round_idx=round_idx)
                        io_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                        reply_file = io_dir / f"{round_idx}_optimization_strategy_reply.txt"
                        reply_file.write_text(raw, encoding="utf-8")
                        strategy_json = extract_json(raw)

                        # Check if method was matched based on machine_check_result
                        # Read machine_check_result JSON file to determine if method was matched
                        method_matched = False
                        machine_check_result_file = io_dir / f"round{round_idx:03d}_machine_check_result.json"
                        if machine_check_result_file.exists():
                            try:
                                with open(machine_check_result_file, 'r', encoding='utf-8') as f:
                                    machine_check_result = json.load(f)
                                case_id = machine_check_result.get("case_id", "NO_MATCH")
                                # method_matched is True only if case_id is not "NO_MATCH"
                                method_matched = (case_id != "NO_MATCH")
                            except Exception as e:
                                print(f"[WARNING] Failed to read machine_check_result.json: {e}. Falling back to checking method_name.")
                                # Fallback: check if method_name exists
                                method_matched = bool(
                                    strategy_json 
                                    and isinstance(strategy_json, dict)
                                    and strategy_json.get("method_name")
                                    and str(strategy_json.get("method_name", "")).strip() != ""
                                )
                        else:
                            print(f"[WARNING] machine_check_result.json not found: {machine_check_result_file}. Falling back to checking method_name.")
                            # Fallback: check if method_name exists
                            method_matched = bool(
                                strategy_json 
                                and isinstance(strategy_json, dict)
                                and strategy_json.get("method_name")
                                and str(strategy_json.get("method_name", "")).strip() != ""
                            )

                        # Save optimization strategy to history file
                        if opt_history_file:
                            opt_history = {
                                "round": round_idx,
                                "parent_kernel": str(parent_kernel_path) if parent_kernel_path else None,
                                "parent_kernel_name": parent_kernel_name if parent_kernel_path else None,
                                "optimization_strategy": strategy_json,
                                "method_matched": method_matched,
                                "timestamp": datetime.now().isoformat(),
                                "runnable": None,  # Will be updated after testing
                                "speedup": None,   # Will be updated after testing
                                "test_passed": False,
                                "repaired": False,  # Will be set to True if repaired
                                "kernel_source": getattr(ind, "code", ""),  # Save generated kernel source for this round
                            }
                            opt_history_file.write_text(json.dumps(opt_history, indent=2, ensure_ascii=False), encoding="utf-8")
                            # Track this opt history file for potential repair updates
                            opt_history_files[round_idx] = opt_history_file
                            print(f"[opt] Optimization history saved to: {opt_history_file}")

                        # Build history block with previously generated kernels (keep last round_idx kernels, or at least 5)
                        # For round 0, keep_last=0 means no history; for round 1+, keep_last should be round_idx to include all previous rounds
                        history_block = _build_history_block(code_dir, keep_last=max(round_idx, 5))
                        # Use parent_kernel (best_kernel) for optimization, not current_kernel
                        opt_prompt = build_optimization_prompt(
                            arch_path=parent_kernel_path if parent_kernel_path else current_kernel.code_path,  # type: ignore[union-attr]
                            gpu_name=args.gpu,
                            history_block=history_block,  # Pass history_block to include previously generated kernels
                            optimization_suggestion=strategy_json
                        )
                        prompt_file = io_dir / f"round{round_idx:03d}_opt_prompt.txt"
                        prompt_file.write_text(opt_prompt, encoding="utf-8")
                        ind = _llm_to_kernel(opt_prompt, code_dir, call_llm, io_dir, round_idx,
                                             log_path=log_path, call_type="optimization")
                        _bench_and_score(
                            ind,
                            ref_py=task_path,
                            device_idx=args.device,
                            warmup=args.warmup,
                            repeat=args.repeat,
                            tol=args.tol,
                            phase="opt",
                            metrics_dir=eval_dir,
                        )
                        
                        # Check if optimized kernel failed - if so, prepare for potential repair chain
                        # The repair chain will be started in the repair phase if the kernel is not runnable
                        opt_kernel_runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                        opt_kernel_score = ind.score if (ind.score is not None and opt_kernel_runnable) else None
                        if not opt_kernel_runnable or opt_kernel_score is None:
                            # Optimized kernel failed, will need repair
                            # If there's no active repair chain, the repair phase will start one with this kernel
                            print(f"[opt] Optimized kernel failed (runnable={opt_kernel_runnable}), will start repair chain if needed")
                        
                        # Record optimized kernel in optimization tree
                        if ind and hasattr(ind, 'code_path') and ind.code_path:
                            kernel_name = ind.code_path.stem
                            parent_name = parent_kernel.code_path.stem if (parent_kernel and hasattr(parent_kernel, 'code_path') and parent_kernel.code_path) else None
                            runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                            speedup = ind.score if (ind.score is not None and runnable) else None
                            # Check if method was matched (method_name exists and is not empty)
                            method_matched = bool(
                                strategy_json 
                                and isinstance(strategy_json, dict)
                                and strategy_json.get("method_name")
                                and str(strategy_json.get("method_name", "")).strip() != ""
                            )
                            
                            optimization_tree[kernel_name] = {
                                "parent": parent_name,
                                "kernel_name": kernel_name,
                                "kernel_path": str(ind.code_path),
                                "speedup": float(speedup) if speedup is not None else None,
                                "runnable": runnable,
                                "ncu_passed": True,  # This kernel's parent went through ncu profiling
                                "phase": "opt",
                                "round": round_idx,
                                "strategy": strategy_json if strategy_json else None,
                                "method_matched": method_matched,
                                "timestamp": datetime.now().isoformat(),
                            }
                        
                        # Update optimization history after testing
                        if opt_history_file and opt_history_file.exists():
                            try:
                                opt_history = json.loads(opt_history_file.read_text(encoding="utf-8"))
                                runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
                                speedup = ind.score if (ind.score is not None and runnable) else None
                                opt_history["runnable"] = runnable
                                opt_history["speedup"] = float(speedup) if speedup is not None else None
                                opt_history["test_passed"] = runnable and speedup is not None
                                opt_history["test_kernel"] = str(getattr(ind, "code_path", None)) if hasattr(ind, "code_path") else None
                                opt_history["test_timestamp"] = datetime.now().isoformat()
                                opt_history_file.write_text(json.dumps(opt_history, indent=2, ensure_ascii=False), encoding="utf-8")
                                if runnable and speedup is not None:
                                    print(f"[opt] Optimization history updated: speedup={speedup:.4f}")
                            except Exception as e:
                                print(f"[opt] Warning: Failed to update optimization history: {e}")

        # -------- update state + record curve --------
        # current_kernel: 当前刚生成/修复完的kernel，用于记录和后续repair
        # 修复时使用current_kernel记录正在修复的kernel，避免和best、test、parent混淆
        current_kernel = ind
        runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
        this_score = ind.score if (ind.score is not None and runnable) else None

        # If a kernel from opt phase fails (not runnable or no valid score), start/continue repair chain
        # If a kernel succeeds (has valid score), clear repair chain (repair chain completed)
        if this_score is not None:
            # Kernel succeeded, clear repair chain if it exists
            if repair_chain_kernel is not None:
                print(f"[repair] Kernel succeeded (speedup={this_score:.4f}), clearing repair chain")
                repair_chain_kernel = None
        else:
            # Kernel failed, if it came from opt phase and there's no active repair chain, start one
            # Note: repair_chain_kernel is set in repair phase, so we don't set it here
            # This is just for logging/debugging
            pass

        if this_score is not None:
            last_score_for_curve = this_score
            scores.append(this_score)
            err_flags.append(False)
            
            # Update base_kernel only if this kernel's score shows significant improvement
            # Criteria: (1) score >= base_score * 1.3 (30% improvement), OR (2) score - base_score >= 0.3 (absolute improvement)
            # Note: Special case for repaired base_kernel is handled in the ncu timeout handler above
            should_update_base = False
            if base_score == float("-inf"):
                # First valid score, always update
                should_update_base = True
            elif base_score <= 0:
                # If base_score is negative or zero, use absolute improvement threshold
                should_update_base = (this_score - base_score >= 0.1)
            else:
                # base_score is positive, check both relative (30%) and absolute (0.3) improvement
                relative_improvement = this_score >= base_score * 1.3
                absolute_improvement = (this_score - base_score) >= 0.3
                should_update_base = relative_improvement or absolute_improvement
            
            if should_update_base:
                improvement_type = "relative (30%)" if (base_score > 0 and this_score >= base_score * 1.3) else "absolute (0.3)"
                print(f"[base] Updating base_kernel: {this_score:.4f} vs {base_score:.4f} ({improvement_type} improvement)", flush=True)
                base_score = this_score
                base_kernel = ind
                with open(test_kernel, "w") as f:
                    f.write(base_kernel.code)
            
            # Update best_kernel unconditionally if score is higher (for statistics)
            if this_score > best_score:
                print(f"[best] Updating best_kernel (statistics): {this_score:.4f} vs {best_score:.4f}", flush=True)
                best_score = this_score
                best_kernel = ind
            
            # Update optimization tree: update speedup if kernel already exists
            if ind and hasattr(ind, 'code_path') and ind.code_path:
                kernel_name = ind.code_path.stem
                if kernel_name in optimization_tree:
                    optimization_tree[kernel_name]["speedup"] = float(this_score)
                    optimization_tree[kernel_name]["runnable"] = runnable
                # If kernel doesn't exist in tree (shouldn't happen, but handle it)
                elif kernel_name not in optimization_tree:
                    # This might happen for compilation timeout repairs
                    parent_name = None
                    if hasattr(ind, 'code') and current_kernel and hasattr(current_kernel, 'code_path') and current_kernel.code_path:
                        parent_name = current_kernel.code_path.stem
                    optimization_tree[kernel_name] = {
                        "parent": parent_name,
                        "kernel_name": kernel_name,
                        "kernel_path": str(ind.code_path),
                        "speedup": float(this_score),
                        "runnable": runnable,
                        "ncu_passed": False,
                        "phase": "unknown",
                        "round": round_idx,
                        "strategy": None,
                        "method_matched": False,  # Unknown phase doesn't have optimization method matching
                        "timestamp": datetime.now().isoformat(),
                    }

        else:
            # on failure: keep last score and mark error
            scores.append(last_score_for_curve)
            err_flags.append(True)

    # plot per-task curve
    fig_path = fig_dir / f"{task_path.stem}_score.png"
    _plot_scores(fig_path, scores, err_flags, title=f"{task_path.stem} (best={best_score:.4f})")
    print(f"[{task_path.name}] Figure saved to: {fig_path}")

    # Save optimization tree to JSON
    tree_path = task_root / "optimization_tree.json"
    tree_data = {
        "task": str(task_path),
        "base_kernel": base_kernel.code_path.stem if (base_kernel and hasattr(base_kernel, 'code_path') and base_kernel.code_path) else None,
        "base_score": float(base_score) if base_score != float("-inf") else None,
        "best_kernel": best_kernel.code_path.stem if (best_kernel and hasattr(best_kernel, 'code_path') and best_kernel.code_path) else None,
        "best_score": float(best_score) if best_score != float("-inf") else None,
        "kernels": optimization_tree,
        "timestamp": datetime.now().isoformat(),
    }
    tree_path.write_text(json.dumps(tree_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{task_path.name}] Optimization tree saved to: {tree_path}")

    usage_totals = _append_usage_totals(log_path)
    
    # Calculate method_matched statistics from optimization_tree
    method_matched_stats = {"total_opt_rounds": 0, "matched_count": 0, "unmatched_count": 0}
    for kernel_info in optimization_tree.values():
        phase = kernel_info.get("phase", "")
        if phase == "opt":
            method_matched_stats["total_opt_rounds"] += 1
            if kernel_info.get("method_matched", False):
                method_matched_stats["matched_count"] += 1
            else:
                method_matched_stats["unmatched_count"] += 1
    
    # Also count from opt_round_*.json files for accuracy
    if code_dir.exists():
        for kernel_dir in code_dir.iterdir():
            if not kernel_dir.is_dir():
                continue
            opt_history_dir = kernel_dir
            if opt_history_dir.exists():
                opt_files = sorted(opt_history_dir.glob("opt_round_*.json"))
                for opt_file in opt_files:
                    try:
                        opt_data = json.loads(opt_file.read_text(encoding="utf-8"))
                        if opt_data.get("optimization_strategy"):
                            # Count only once (avoid double counting with optimization_tree)
                            # We'll use optimization_tree as primary source, but verify with opt_round files
                            pass
                    except Exception:
                        continue

    return {
        "task": str(task_path),
        "best_score": float(best_score) if best_score != float("-inf") else 0.0,
        "best_runnable": bool(getattr(best_kernel, "metrics", {}).get("runnable", False)) if best_kernel else False,
        "task_dir": str(task_root),
        "figure": str(fig_path),
        "input_tokens_sum": usage_totals["input_tokens"],
        "output_tokens_sum": usage_totals["output_tokens"],
        "total_tokens_sum": usage_totals["total_tokens"],
        "method_matched_stats": method_matched_stats,
    }


# --------------------- summary saving ------------------
def _save_global_summary(batch_dir: Path, summary: List[Dict[str, Any]], avg_speedup: float, accuracy: float, total_tokens_sum: float) -> None:
    """Save summary.json and summary.csv under the batch_dir."""
    batch_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    out_json = {
        "avg_speedup": avg_speedup,
        "accuracy": accuracy,
        "total_tokens_sum": total_tokens_sum,
        "num_tasks": len(summary),
        "tasks": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (batch_dir / "summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # CSV
    csv_path = batch_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "best_score", "best_runnable", "task_dir", "figure"])
        for s in summary:
            writer.writerow([s["task"], f'{s["best_score"]:.6f}', int(
                bool(s["best_runnable"])), s["task_dir"], s["figure"]])
        writer.writerow([])
        writer.writerow(["avg_speedup", f"{avg_speedup:.6f}"])
        writer.writerow(["accuracy", f"{accuracy:.6f}"])
        writer.writerow(["total_tokens_sum", f"{int(total_tokens_sum)}"])

    print(f"[GLOBAL] Saved: {batch_dir/'summary.json'}")
    print(f"[GLOBAL] Saved: {csv_path}")


# --------------------------- main ----------------------
def main():
    args = _build_arg_parser().parse_args()

    all_tasks = _collect_tasks(args.arch_py)

    # Apply filter from summary.json if specified
    if args.filter_from_summary:
        all_tasks = _filter_tasks_from_summary(all_tasks, args.filter_from_summary)
        if not all_tasks:
            print("[ERROR] No tasks found after filtering from summary.json. Exiting.")
            return

    # ---- Create ONE batch folder for this run ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = _build_run_tag(args.server_type, args.model_name)
    # batch name hints: single file uses file stem; directory uses 'batch'
    if args.arch_py.is_file():
        batch_name = f"{stamp}_{args.arch_py.stem}_{run_tag}"
    else:
        # include sampling info for traceability
        if args.filter_from_summary:
            pick_note = "filtered_from_summary"
        elif args.first_n and args.first_n > 0:
            pick_note = f"first{args.first_n}"
        else:
            pick_note = f"num{args.num_tasks}_seed{args.shuffle_seed}"
        batch_name = f"{stamp}_batch_{pick_note}_{run_tag}"
    batch_dir = (args.work_dir / batch_name).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH] Output folder: {batch_dir}")

    # single file → run once (still inside the same batch folder)
    if args.arch_py.is_file():
        res = _run_single_task(all_tasks[0], args, batch_dir=batch_dir)
        summary = [res]
        avg_speedup = res["best_score"]
        accuracy = 1.0 if res["best_runnable"] else 0.0
        total_tokens_sum = res.get("total_tokens_sum", 0)
        print(f"[SUMMARY] {res}")
        print(f"[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
        return

    # directory: first_n takes precedence; else optionally sample
    # Note: If filter_from_summary is used, we already have the filtered list,
    # so we can still apply first_n or num_tasks on the filtered list if needed
    if args.first_n and args.first_n > 0:
        # support starting from an arbitrary (1-based) index in the sorted list
        start_idx = max(0, (args.start_from or 1) - 1)
        end_idx = min(len(all_tasks), start_idx + args.first_n)
        if start_idx >= len(all_tasks):
            print(f"[Task Picker] start_from={args.start_from} exceeds number of tasks ({len(all_tasks)}); nothing to run.")
            picked = []
        else:
            picked = all_tasks[start_idx:end_idx]
            print(
                f"[Task Picker] Found {len(all_tasks)} tasks, "
                f"taking {len(picked)} tasks from sorted positions [{start_idx+1}..{end_idx}]."
            )
    elif args.filter_from_summary:
        # If filter_from_summary is used, by default use all filtered tasks
        # unless num_tasks is explicitly set to a value other than default (1)
        if args.num_tasks == 1:
            # Default num_tasks=1, but filter_from_summary means use all filtered tasks
            picked = all_tasks
            print(f"[Task Picker] Using all {len(picked)} filtered tasks from summary.json.")
        else:
            # User explicitly set num_tasks, sample from filtered tasks
            picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
            print(f"[Task Picker] Found {len(all_tasks)} filtered tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")
    else:
        # Normal sampling without filter
        picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")

    summary: List[Dict[str, Any]] = []
    for i, task in enumerate(picked, 1):
        print(f"\n===== [{i}/{len(picked)}] Running task: {task} =====")
        res = _run_single_task(task, args, batch_dir=batch_dir)
        summary.append(res)

    # global summary using each task's best kernel
    if summary:
        avg_speedup = sum(s["best_score"] for s in summary) / len(summary)
        accuracy = sum(1 for s in summary if s["best_runnable"]) / len(summary)
        total_tokens_sum = sum(int(s.get("total_tokens_sum", 0) or 0) for s in summary)
        print("\n===== SUMMARY =====")
        for s in summary:
            print(f"{s['task']}: best_score={s['best_score']:.4f}  runnable={s['best_runnable']}  fig={s['figure']}")
        print(f"\n[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        # ---- save under the SAME batch folder ----
        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
    else:
        print("No tasks were run.")


if __name__ == "__main__":
	main()
