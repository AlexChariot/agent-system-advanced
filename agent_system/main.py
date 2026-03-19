import json
import subprocess
from pathlib import Path

import typer
from agent_system.graph import build_graph
from agent_system.logging_config import setup_logging

# Apply colored logging once at startup — all agents inherit it
setup_logging()

app = typer.Typer(add_completion=False)
graph = build_graph()

HISTORY_FILE = Path.home() / ".agent_system_history.json"
_selected_model_file = Path.home() / ".agent_system_model"

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


@app.command()
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


@app.command()
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


@app.command()
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


@app.command()
def show_history():
    """Show past executions (persisted across sessions)."""
    history = _load_history()

    if not history:
        typer.echo("No history.")
        return

    for i, item in enumerate(history, 1):
        typer.echo(f"{i}. {item['goal']}")
        typer.echo(f"   → {item['result']}\n")


@app.command()
def current_model():
    """Show the current LLM model."""
    typer.echo(f"Current model: {_load_model()}")


@app.command()
def clear_history():
    """Clear the persisted execution history."""
    _save_history([])
    typer.echo("History cleared.")


if __name__ == "__main__":
    app()