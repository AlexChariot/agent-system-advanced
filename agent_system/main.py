import json
import subprocess
import time
import shutil
from pathlib import Path
from typing import Optional

import typer
from agent_system.graph import build_graph
from agent_system.logging_config import setup_logging
from agent_system.memory.vector_memory import _get_vectorstore, recall_memory
from agent_system.agents.planner import planner
from agent_system.agents.memory_agent import memory_agent
from agent_system.agents.researcher import researcher
from agent_system.agents.analyst import analyst
from agent_system.agents.executor import executor
from agent_system.agents.critic import critic

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # Fallback if neither is available

# Apply colored logging once at startup — all agents inherit it
setup_logging()

# Allow both --help and -h on every command and sub-command
_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(add_completion=False, context_settings=_CONTEXT_SETTINGS)

# Sub-apps for grouping commands
core_app    = typer.Typer(help="Core agent operations",      context_settings=_CONTEXT_SETTINGS)
memory_app  = typer.Typer(help="Vector memory management",   context_settings=_CONTEXT_SETTINGS)
state_app   = typer.Typer(help="State and configuration",    context_settings=_CONTEXT_SETTINGS)
testing_app = typer.Typer(help="Testing and benchmarking",   context_settings=_CONTEXT_SETTINGS)
utils_app   = typer.Typer(help="Utilities",                  context_settings=_CONTEXT_SETTINGS)

# Add sub-apps to main app
app.add_typer(core_app, name="core", help="Core agent operations")
app.add_typer(memory_app, name="memory", help="Vector memory management")
app.add_typer(state_app, name="state", help="State and configuration")
app.add_typer(testing_app, name="testing", help="Testing and benchmarking")
app.add_typer(utils_app, name="utils", help="Utilities")

graph = build_graph()

HISTORY_FILE = Path.home() / ".agent_system_history.json"
_selected_model_file = Path.home() / ".agent_system_model"
CONFIG_FILE = Path.home() / ".agent_system_config.json"

DEFAULT_MODEL = "llama3.1"


def _load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_history(history: list) -> None:
    try:
        HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2))
    except OSError as e:
        typer.echo(f"[Warning] Could not save history: {e}", err=True)


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_config(config: dict) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2))
    except OSError as e:
        typer.echo(f"[Warning] Could not save config: {e}", err=True)


def _load_model() -> str:
    if _selected_model_file.exists():
        return _selected_model_file.read_text().strip() or DEFAULT_MODEL
    return DEFAULT_MODEL


def _save_model(model: str) -> None:
    try:
        _selected_model_file.write_text(model)
    except OSError as e:
        typer.echo(f"[Warning] Could not save model: {e}", err=True)


def _list_ollama_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split("\n")[1:]
        return [line.split()[0] for line in lines if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _save_config(config: dict) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2))
    except OSError as e:
        typer.echo(f"[Warning] Could not save config: {e}", err=True)
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split("\n")[1:]
        return [line.split()[0] for line in lines if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


@core_app.command()
def run(goal: str):
    """Run the agent system with a goal."""
    if not goal or not goal.strip():
        typer.echo("Error: Goal cannot be empty.", err=True)
        raise typer.Exit(code=1)

    selected_model = _load_model()
    history = _load_history()

    typer.echo(f"Model : {selected_model}")
    typer.echo(f"Goal  : {goal}\n")

    try:
        result = graph.invoke({
            "goal": goal,
            "plan": [],
            "history": [],
            "selected_model": selected_model,
        })
    except Exception as e:
        typer.echo(f"\nError during execution: {e}", err=True)
        # Still save history for partial/failed attempts
        history.append({"goal": goal, "result": f"[ERROR] {str(e)}"})
        _save_history(history)
        raise typer.Exit(code=1)

    typer.echo("\n── FINAL RESULT ──\n")
    typer.echo(result.get("result", "(no result)"))

    history.append({"goal": goal, "result": result.get("result", "")})
    _save_history(history)


@core_app.command()
def models():
    """List available Ollama models."""
    available = _list_ollama_models()

    if not available:
        typer.echo("No models found. Is Ollama running? (`ollama serve`)")
        return

    current = _load_model()
    for m in available:
        marker = " ◀ active" if m == current else ""
        typer.echo(f"  {m}{marker}")


@core_app.command()
def set_model(model: str):
    """Set the active LLM model (persisted across sessions)."""
    available = _list_ollama_models()

    if not available:
        typer.echo("Warning: Could not verify available models (Ollama not running?).", err=True)
        typer.echo("Proceeding anyway...", err=True)
        _save_model(model)
        typer.echo(f"Model set to: {model}")
        return

    if model not in available:
        typer.echo(
            f"Error: Model '{model}' not found in Ollama. Available models:",
            err=True,
        )
        for m in available:
            typer.echo(f"  - {m}", err=True)
        raise typer.Exit(code=1)

    _save_model(model)
    typer.echo(f"Model set to: {model}")


@core_app.command()
def show_history():
    """Show past executions (persisted across sessions)."""
    history = _load_history()

    if not history:
        typer.echo("No history.")
        return

    for i, item in enumerate(history, 1):
        typer.echo(f"{i}. {item['goal']}")
        typer.echo(f"   → {item['result']}\n")


@core_app.command()
def current_model():
    """Show the current LLM model."""
    typer.echo(f"Current model: {_load_model()}")


def _get_ollama_model_info(model: str) -> Optional[dict]:
    """Fetch detailed model info from Ollama via `ollama show --json`."""
    try:
        result = subprocess.run(
            ["ollama", "show", "--json", model],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def _format_params(n: Optional[int]) -> str:
    """Format a raw parameter count into a human-readable string (e.g. 7.0B)."""
    if n is None:
        return "Unknown"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return str(n)


def _extract_model_fields(data: dict, model_name: str) -> dict:
    """Extract architecture, params, context, embedding and quantization from raw Ollama JSON."""
    details = data.get("details") or {}
    meta = data.get("model_info") or data.get("modelinfo") or {}

    arch = (
        meta.get("general.architecture")
        or details.get("architecture")
    )

    def _meta(*keys):
        for k in keys:
            v = meta.get(k)
            if v is not None:
                return v
        return None

    params   = _meta("general.parameter_count")
    ctx      = _meta(f"{arch}.context_length", "context_length")
    emb      = _meta(f"{arch}.embedding_length", "embedding_length")
    quant    = (
        details.get("quantization_level")
        or data.get("quantization_level")
    )
    # Infer quantization from the model tag when not reported (e.g. "llama3.1:q4_0")
    if not quant and ":" in model_name:
        tag = model_name.split(":")[-1]
        if tag.lower().startswith("q") or "fp" in tag.lower() or "bf" in tag.lower():
            quant = tag.upper()

    families = details.get("families") or details.get("family")
    if isinstance(families, list):
        families = ", ".join(families)

    return {
        "architecture": arch,
        "params": params,
        "context_length": ctx,
        "embedding_length": emb,
        "quantization": quant,
        "family": families,
    }


@core_app.command()
def model_info(
    model: Optional[str] = typer.Argument(None, help="Model name (defaults to active model)")
):
    """Show detailed info for a model: parameters, architecture, context length, embedding size, quantization."""
    target = model or _load_model()
    typer.echo(f"Fetching info for: {target}\n")

    data = _get_ollama_model_info(target)
    if data is None:
        typer.echo(
            f"Error: could not retrieve info for '{target}'. "
            "Is Ollama running and the model pulled?",
            err=True,
        )
        raise typer.Exit(code=1)

    f = _extract_model_fields(data, target)

    typer.echo(f"  {'Model':<22} {target}")
    typer.echo(f"  {'Architecture':<22} {f['architecture'] or 'Unknown'}")
    typer.echo(f"  {'Parameters':<22} {_format_params(f['params'])}")
    typer.echo(f"  {'Context length':<22} {f['context_length'] or 'Unknown'}")
    typer.echo(f"  {'Embedding length':<22} {f['embedding_length'] or 'Unknown'}")
    typer.echo(f"  {'Quantization':<22} {f['quantization'] or 'Unknown'}")
    if f["family"]:
        typer.echo(f"  {'Family':<22} {f['family']}")


@core_app.command()
def models_info():
    """Show a summary table of all available models with key metadata."""
    available = _list_ollama_models()
    if not available:
        typer.echo("No models found. Is Ollama running? (`ollama serve`)")
        return

    current = _load_model()
    col = (35, 12, 10, 10, 10, 10)
    header = (
        f"{'Model':<{col[0]}} {'Arch':<{col[1]}} {'Params':<{col[2]}} "
        f"{'Context':<{col[3]}} {'Embed':<{col[4]}} {'Quant':<{col[5]}}"
    )
    typer.echo(header)
    typer.echo("─" * len(header))

    for m in available:
        active = " ◀" if m == current else ""
        data = _get_ollama_model_info(m)
        if data is None:
            typer.echo(f"{(m + active):<{col[0]}} {'(unavailable)'}")
            continue

        f = _extract_model_fields(data, m)
        typer.echo(
            f"{(m + active):<{col[0]}} "
            f"{str(f['architecture'] or '?'):<{col[1]}} "
            f"{_format_params(f['params']):<{col[2]}} "
            f"{str(f['context_length'] or '?'):<{col[3]}} "
            f"{str(f['embedding_length'] or '?'):<{col[4]}} "
            f"{str(f['quantization'] or '?'):<{col[5]}}"
        )


@core_app.command()
def clear_history():
    """Clear the persisted execution history."""
    _save_history([])
    typer.echo("History cleared.")


@memory_app.command()
def inspect_memory():
    """Inspect the vector memory: show count and sample entries."""
    try:
        vs = _get_vectorstore()
        count = vs._collection.count()  # Approximate count
        typer.echo(f"Memory entries: {count}")
        if count > 0:
            docs = vs.similarity_search("", k=min(3, count))
            typer.echo("Sample entries:")
            for i, doc in enumerate(docs, 1):
                typer.echo(f"  {i}. {doc.page_content[:100]}...")
    except Exception as e:
        typer.echo(f"Error inspecting memory: {e}", err=True)


@memory_app.command()
def clear_memory():
    """Clear all vector memory (irreversible)."""
    if not typer.confirm("This will delete all stored memories. Continue?"):
        return
    try:
        vs = _get_vectorstore()
        vs.delete_collection()
        typer.echo("Memory cleared.")
    except Exception as e:
        typer.echo(f"Error clearing memory: {e}", err=True)


@memory_app.command()
def export_memory(output_file: str = "memory_export.json"):
    """Export vector memory to a JSON file."""
    try:
        vs = _get_vectorstore()
        docs = vs.get()  # Get all documents
        data = {"documents": docs.get("documents", []), "metadatas": docs.get("metadatas", [])}
        Path(output_file).write_text(json.dumps(data, indent=2))
        typer.echo(f"Memory exported to {output_file}")
    except Exception as e:
        typer.echo(f"Error exporting memory: {e}", err=True)


@memory_app.command()
def search_memory(query: str):
    """Search vector memory manually."""
    result = recall_memory(query)
    if result:
        typer.echo("Results:")
        typer.echo(result)
    else:
        typer.echo("No results found.")


@state_app.command()
def show_state():
    """Show the current or last known agent state."""
    history = _load_history()
    if history:
        last = history[-1]
        typer.echo("Last execution state:")
        typer.echo(f"  Goal: {last['goal']}")
        typer.echo(f"  Result: {last['result']}")
    else:
        typer.echo("No state available (empty history).")


@state_app.command()
def set_config(param: str, value: str):
    """Set a configuration parameter (e.g., recall_k:5)."""
    config = _load_config()
    config[param] = value
    _save_config(config)
    typer.echo(f"Config {param} set to {value}")


@testing_app.command()
def test_agents():
    """Run basic unit tests on all agents."""
    typer.echo("Testing agents...")
    model = _load_model()

    # Base state shared by most agents
    base_state = {
        "goal": "Test goal",
        "selected_model": model,
        "plan": ["Step 1", "Step 2"],
        "current_task": "Step 1",
        "research": "",
        "analysis": "",
        "result": "",
        "context": "",
        "retrieved_memory": "",
        "evaluation": "",
        "history": [],
    }

    passed = 0
    failed = 0

    def _run(label: str, fn, state: dict):
        nonlocal passed, failed
        try:
            fn(state)
            typer.echo(f"  ✓ {label}: OK")
            passed += 1
        except Exception as e:
            typer.echo(f"  ✗ {label}: {e}")
            failed += 1

    # Planner — only needs goal + model
    _run("Planner", planner, {**base_state})

    # Memory Agent — runs after planning
    _run("Memory Agent", memory_agent, {**base_state})

    # Researcher — needs a current task to search for
    _run("Researcher", researcher, {**base_state})

    # Analyst — needs research output
    _run("Analyst", analyst, {**base_state, "research": "Sample research data about test goal."})

    # Executor — needs analysis output
    _run("Executor", executor, {**base_state, "analysis": "Sample analysis: key insight extracted."})

    # Critic — needs a result to evaluate
    _run("Critic", critic, {**base_state, "result": "Sample final result for the test goal."})

    typer.echo(f"\nTests completed: {passed} passed, {failed} failed.")


@testing_app.command()
def benchmark(goal: Optional[str] = None):
    """Benchmark execution time for a goal (default: open source LLMs report)."""
    if not goal:
        goal = "Write a report on the impact of open source LLMs"
    
    typer.echo(f"Benchmarking goal: {goal}\n")
    start = time.time()
    try:
        result = graph.invoke({
            "goal": goal,
            "plan": [],
            "history": [],
            "selected_model": _load_model(),
        })
        elapsed = time.time() - start
        
        # Extract metrics
        plan_count = len(result.get("plan", []))
        history = result.get("history", [])
        steps = len(history)
        final_result = result.get("result", "")
        result_length = len(final_result)
        
        # Estimate tokens (rough: ~4 chars per token)
        estimated_tokens = (len(goal) + result_length) // 4
        
        # Display results
        typer.echo("── BENCHMARK RESULTS ──\n")
        typer.echo(f"Execution time: {elapsed:.2f}s")
        typer.echo(f"Agent steps: {steps}")
        typer.echo(f"Tasks in plan: {plan_count}")
        typer.echo(f"Output length: {result_length} chars")
        typer.echo(f"Est. tokens: ~{estimated_tokens}")
        typer.echo(f"\n── OUTPUT PREVIEW ──\n")
        typer.echo(f"{final_result[:300]}..." if len(final_result) > 300 else final_result)
        
    except Exception as e:
        typer.echo(f"Benchmark failed: {e}", err=True)
        raise typer.Exit(code=1)


@testing_app.command()
def stats():
    """Show usage statistics."""
    history = _load_history()
    total_runs = len(history)
    models_used = set()
    errors = 0
    for item in history:
        if "[ERROR]" in item["result"]:
            errors += 1
        # Assume model from config or default
    typer.echo(f"Total runs: {total_runs}")
    typer.echo(f"Errors: {errors}")
    typer.echo(f"Success rate: {((total_runs - errors) / total_runs * 100):.1f}%" if total_runs > 0 else "N/A")


@utils_app.command()
def backup(backup_dir: str = "agent_backup"):
    """Backup history and memory to a directory."""
    backup_path = Path(backup_dir)
    backup_path.mkdir(exist_ok=True)
    try:
        if HISTORY_FILE.exists():
            shutil.copy(HISTORY_FILE, backup_path / "history.json")
        export_memory(str(backup_path / "memory.json"))
        typer.echo(f"Backup created in {backup_dir}")
    except Exception as e:
        typer.echo(f"Backup failed: {e}")


@utils_app.command()
def version():
    """Show project version."""
    try:
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        version = data.get("project", {}).get("version", "Unknown")
        typer.echo(f"Version: {version}")
    except Exception as e:
        typer.echo(f"Could not read version: {e}")


if __name__ == "__main__":
    app()