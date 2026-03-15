import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.panel import Panel
from rich.tree import Tree
import json
import re

from automl_kb.database.query import QueryEngine
from automl_kb.recommendation.engine import RecommendationEngine 
from automl_kb.export.artifacts import export_package

app = typer.Typer()
console = Console()
query_engine = QueryEngine()
engine = RecommendationEngine() 

def get_real_algorithm_name(candidate):
    """
    Extracts specific algorithm names from TPOT pipelines.
    """
    algo = candidate.get('algorithm', 'Unknown')
    fw = candidate.get('framework', 'unknown')
    
    if fw == 'tpot':
        try:
            params = json.loads(candidate.get('params_json', '{}'))
            pipeline_str = params.get('pipeline', '')
            if '(' in pipeline_str:
                return pipeline_str.split('(')[0].strip()
        except:
            pass
    return algo

def get_best_candidates_per_framework(candidates):
    """Returns a dict of {framework: candidate} for the best accuracy."""
    best_by_fw = {}
    for c in candidates:
        fw = c.get('framework', 'unknown')
        metrics = c.get('metrics', {})
        acc = metrics.get('accuracy', 0.0)
        
        if fw not in best_by_fw or acc > best_by_fw[fw]['acc']:
            best_by_fw[fw] = {'c': c, 'acc': acc}
    return best_by_fw

def display_leaderboard(candidates):
    """Show the best model from each framework."""
    table = Table(title="🏆 Framework Leaderboard (Best Accuracy)", header_style="bold yellow")
    table.add_column("Framework", style="cyan")
    table.add_column("Best Algo")
    table.add_column("Best Accuracy", justify="right")
    
    best_by_fw = get_best_candidates_per_framework(candidates)
            
    for fw, data in best_by_fw.items():
        c = data['c']
        algo_name = get_real_algorithm_name(c)
        acc = data['acc']
        table.add_row(fw, algo_name[:30], f"{acc:.4f}")
        
    console.print(table)

def explore_variability(candidate):
    """Break down the JSON params into a tree."""
    algo = get_real_algorithm_name(candidate)
    fw = candidate.get('framework', 'unknown')
    raw_json = candidate.get('params_json', '{}')
    
    try:
        params = json.loads(raw_json)
    except:
        params = {"error": "Could not parse JSON"}

    tree = Tree(f"[bold cyan]{algo} ({fw}) Configuration[/bold cyan]")
    
    if 'pipeline' in params:
        p_str = params['pipeline']
        pipeline_node = tree.add("[bold]Pipeline Structure[/bold]")
        parts = re.split(r',\s*(?=[A-Za-z_]+\()', p_str) 
        for part in parts:
            if '(' in part:
                func_name = part.split('(')[0]
                args = part[len(func_name)+1:].rstrip(')')
                func_node = pipeline_node.add(f"[green]{func_name}[/green]")
                arg_list = args.split(',')
                for arg in arg_list:
                    if '=' in arg:
                        k, v = arg.split('=', 1)
                        func_node.add(f"{k.strip()}: [yellow]{v.strip()}[/yellow]")
    else:
        for k, v in params.items():
            if isinstance(v, dict):
                sub = tree.add(f"[bold]{k}[/bold]")
                for sk, sv in v.items():
                    sub.add(f"{sk}: [yellow]{sv}[/yellow]")
            else:
                tree.add(f"{k}: [yellow]{v}[/yellow]")
                
    console.print(tree)

@app.command()
def chat():
    console.print(Panel.fit(
        "[bold cyan]AutoML Knowledge Base Agent[/bold cyan]\nI help you find the best models based on historical experiments.",
        border_style="cyan"))

    # 1. Dataset ID
    task_id = IntPrompt.ask("[bold]Enter OpenML Task ID[/bold]")
    dataset = query_engine.get_dataset_by_task(task_id)

    if not dataset:
        console.print(f"[bold red]Unknown Task {task_id}.[/bold red]")
        if Confirm.ask("Would you like to search for similar datasets? (Simulated)"):
            console.print("[dim]Feature not implemented yet. Please run the runner first.[/dim]")
        raise typer.Exit()

    console.print(f"\n[green]Dataset Found:[/green] {dataset['name']} ({dataset['rows']} rows, {dataset['task_type']})")

    # 2. Leaderboard
    with console.status("Fetching candidates..."):
        candidates = query_engine.fetch_candidates_with_metrics(task_id)
        
    if not candidates:
        console.print("[red]No candidates found for this task.[/red]")
        return

    if Confirm.ask("Show the leaderboard (Best from each tool)?"):
        display_leaderboard(candidates)
        
        if Confirm.ask("Do you want to inspect the configurations of these best models?"):
            best_by_fw = get_best_candidates_per_framework(candidates)
            frameworks = list(best_by_fw.keys())
            
            while True:
                choice = Prompt.ask(
                    "Which framework's best model do you want to check?",
                    choices=frameworks + ["done"],
                    default="done"
                )
                
                if choice == "done":
                    break
                
                selected_candidate = best_by_fw[choice]['c']
                explore_variability(selected_candidate)
                console.print("\n")

    # 3. Intent & Constraints
    objective = Prompt.ask(
        "\nWhat is your primary goal?",
        choices=["accuracy", "energy", "latency"],
        default="accuracy"
    )

    constraints = {}
    if Confirm.ask("Do you have hard constraints?"):
        if Confirm.ask("Max Inference Latency?"):
            constraints['max_latency_ms'] = FloatPrompt.ask("Max ms", default=50.0)
        if Confirm.ask("Min Accuracy?"):
            constraints['min_accuracy'] = FloatPrompt.ask("Value 0-1", default=0.8)

    # 4. Processing
    with console.status("[bold green]Analyzing..."):
        filtered = engine.filter_candidates(candidates, constraints)
        results = engine.select_recommendations(filtered, objective)

    if not results or not results.get('best'):
        console.print("[bold red]No models matched your constraints![/bold red]")
        return

    # 5. Recommendation Display
    best = results['best']
    b_metrics = best.get('metrics', {})
    b_acc = b_metrics.get('accuracy', 0)
    b_eng = b_metrics.get('inference_energy_kwh', 0)
    b_lat = b_metrics.get('inference_latency_per_row_ms', 0)
    
    b_eng_str = f"{b_eng:.2e} kWh" if b_eng > 0 else "N/A"
    b_lat_str = f"{b_lat:.2f} ms" if b_lat > 0 else "N/A"

    console.print(Panel(
        f"[bold]Top Recommendation:[/bold] {get_real_algorithm_name(best)} ({best['framework']})\n"
        f"Score: {b_acc:.4f} | Energy: {b_eng_str} | Latency: {b_lat_str}",
        title="Winner", style="green"
    ))
    
    # 6. Alternatives Table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Type", style="dim")
    table.add_column("Algo")
    table.add_column("Acc", justify="right")
    table.add_column("Energy (kWh)", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Train Time (s)", justify="right")

    def get_val(c, key, fmt="{:.4f}"):
        v = c.get(key)
        if v is None and 'metrics' in c:
            v = c['metrics'].get(key)
            
        if v is None: return "N/A"
        if v == 0.0 and key in ['inference_energy_kwh', 'inference_latency_per_row_ms', 'training_duration_secs']:
            return "N/A"
        return fmt.format(v)

    def add_row(label, c, check_metric=None):
        if not c: return
        if check_metric:
            val = c.get('metrics', {}).get(check_metric, 0)
            if val == 0: return 

        m = c.get('metrics', {})
        t_time = c.get('training_duration_secs')
        t_time_str = f"{t_time:.1f}" if t_time and t_time > 0 else "N/A"

        table.add_row(
            label,
            get_real_algorithm_name(c)[:25],
            f"{m.get('accuracy', 0):.4f}",
            get_val(c, 'inference_energy_kwh', "{:.2e}"),
            get_val(c, 'inference_latency_per_row_ms', "{:.2f}"),
            t_time_str
        )

    add_row("Alt: Max Acc", results.get('alt_performance'))
    add_row("Alt: Greenest", results.get('alt_energy'), check_metric='inference_energy_kwh')
    add_row("Alt: Fastest", results.get('alt_speed'), check_metric='inference_latency_per_row_ms')
    
    console.print(table)

    # 7. Variability Exploration
    if Confirm.ask(f"\nDo you want to explore the configuration of the WINNER ({get_real_algorithm_name(best)})?"):
        explore_variability(best)

    # 8. Export
    if Confirm.ask("\nExport reproduction scripts for this model?"):
        path = export_package(best, dataset)
        console.print(f"[green]Saved to:[/green] {path}")

if __name__ == "__main__":
    app()