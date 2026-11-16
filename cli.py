#!/usr/bin/env python3
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

from retrieval.document_loader import DocumentLoader
from retrieval.vector_store import LocalVectorStore
from feedback.level3_system_memory import FeedbackRegistry, GoldenCache, SOPArchitect
from graph.workflow import AgenticRAGWorkflow
from config.settings import settings

console = Console()

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]Agentic RAG with Multilevel Feedback System[/bold cyan]\n"
        "[dim]Level 1: Query & Document Grading  •  Level 2: Grounding & Audit  •  Level 3: Memory & Evolution[/dim]",
        border_style="cyan"
    ))

def display_trace_table(trace: list):
    table = Table(title="[bold yellow]Execution Trace & Multilevel Feedback Loops[/bold yellow]", box=box.ROUNDED)
    table.add_column("Level", style="magenta", width=22)
    table.add_column("Node", style="cyan", width=18)
    table.add_column("Action Taken", style="green")
    table.add_column("Key Metrics / Details", style="dim")

    for step in trace:
        level = step.get("level", "")
        node = step.get("node", "")
        action = step.get("action", "")
        details = str(step.get("details", {}))
        if len(details) > 80:
            details = details[:77] + "..."
        table.add_row(level, node, action, details)

    console.print(table)

def main():
    print_banner()

    # Initialize components
    console.print("[dim]Initializing local vector store and loading sample documents...[/dim]")
    loader = DocumentLoader()
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
    chunks = loader.load_directory(sample_dir)

    vector_store = LocalVectorStore()
    if chunks:
        vector_store.add_documents(chunks)
        console.print(f"[green]✓[/green] Loaded and indexed {len(chunks)} local document chunks.")
    else:
        console.print("[yellow]![/yellow] No documents found in data/sample_docs.")

    registry = FeedbackRegistry()
    golden_cache = GoldenCache(registry)
    workflow = AgenticRAGWorkflow(vector_store=vector_store, registry=registry, golden_cache=golden_cache)
    sop_architect = SOPArchitect(registry)

    console.print(f"[green]✓[/green] Active LLM Provider: [bold]{settings.llm_provider}[/bold]")
    console.print("[dim]Type your query, or 'sop' for system diagnostics, or 'exit' to quit.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Query[/bold cyan]").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Exiting Agentic RAG session.[/dim]")
                break

            if user_input.lower() == "sop":
                diag = sop_architect.analyze_system_performance()
                console.print(Panel(
                    f"[bold]Diagnosis:[/bold] {diag.get('diagnosis')}\n"
                    f"[bold]Recommended SOP Update:[/bold] {diag.get('recommended_sop_update')}\n"
                    f"[bold]Confidence:[/bold] {diag.get('confidence')}\n"
                    f"[bold]Telemetry:[/bold] {diag.get('telemetry')}",
                    title="Level 3: System Diagnostics & SOP Evolution",
                    border_style="magenta"
                ))
                continue

            # Run through Agentic RAG Graph
            with console.status("[bold green]Executing multi-agent feedback graph...[/bold green]"):
                state = workflow.run(user_input)

            # Display execution trace table
            display_trace_table(state.get("execution_trace", []))

            # Display final generation
            is_grounded = state.get("is_grounded")
            is_complete = state.get("is_complete")
            cached = state.get("cached_answer")

            status_badges = []
            if cached:
                status_badges.append("[bold green]⚡ GOLDEN CACHE HIT[/bold green]")
            if is_grounded:
                status_badges.append(f"[bold green]✓ Grounded ({state.get('groundedness_score', 0):.2f})[/bold green]")
            else:
                status_badges.append(f"[bold red]✗ Grounding Issue ({state.get('groundedness_score', 0):.2f})[/bold red]")
            if is_complete:
                status_badges.append(f"[bold green]✓ Complete ({state.get('completeness_score', 0):.2f})[/bold green]")

            console.print(Panel(
                f"{state.get('generation')}\n\n[dim]Status: {' | '.join(status_badges)}[/dim]",
                title=f"[bold]Response for: '{state.get('query')}'[/bold]",
                border_style="green" if (is_grounded and is_complete) else "yellow"
            ))

            # Level 3 Human-in-the-Loop Feedback Prompt
            if not cached and state.get("system_log_id"):
                ask_fb = Confirm.ask("[cyan]Provide Level 3 human feedback on this response?[/cyan]", default=False)
                if ask_fb:
                    fb_choice = Prompt.ask("Rating", choices=["+1", "-1", "0"], default="+1")
                    comment = Prompt.ask("Comment (optional)", default="")
                    rating = int(fb_choice)

                    # Log feedback
                    registry.log_human_feedback(
                        inference_id=state["system_log_id"],
                        rating=rating,
                        comment=comment
                    )

                    # Update vector dynamic weights
                    delta = settings.dynamic_weight_increment * rating
                    for doc in state.get("graded_docs", []):
                        vector_store.update_weight(doc["id"], delta)

                    # Add to golden cache if positive
                    if rating > 0:
                        add_cache = Confirm.ask("Promote this response to Golden Cache?", default=True)
                        if add_cache:
                            golden_cache.store_exemplar(state["query"], state["generation"], verified_by="terminal_user")
                            console.print("[green]✓ Added to Golden Cache for zero-latency future hits![/green]")

                    console.print(f"[green]✓ Level 3 feedback registered. Vector weights updated by delta {delta:+.2f}[/green]")

        except KeyboardInterrupt:
            console.print("\n[dim]Session terminated by user.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Error executing workflow:[/bold red] {e}")

if __name__ == "__main__":
    main()

