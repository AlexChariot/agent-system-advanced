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

app = typer.Typer(add_completion=False)

# Sub-apps for grouping commands
core_app = typer.Typer(help="Core agent operations")
memory_app = typer.Typer(help="Vector memory management")
state_app = typer.Typer(help="State and configuration")
testing_app = typer.Typer(help="Testing and benchmarking")
utils_app = typer.Typer(help="Utilities")

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
    """Run basic unit tests on agents."""
    typer.echo("Testing agents...")
    model = _load_model()
    state = {"goal": "Test goal", "selected_model": model}

    # Test planner
    try:
        result = planner(state)
        typer.echo("✓ Planner: OK")
    except Exception as e:
        typer.echo(f"✗ Planner: {e}")

    # Test memory_agent
    try:
        result = memory_agent(state)
        typer.echo("✓ Memory Agent: OK")
    except Exception as e:
        typer.echo(f"✗ Memory Agent: {e}")

    # Test critic
    try:
        result = critic({"result": "Test result", "goal": "Test goal", "selected_model": model})
        typer.echo("✓ Critic: OK")
    except Exception as e:
        typer.echo(f"✗ Critic: {e}")

    typer.echo("Tests completed.")


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