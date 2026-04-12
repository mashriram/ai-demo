"""
╔══════════════════════════════════════════════════════════════════╗
║        TurboQuant vs RotorQuant — Ultimate Video Benchmark       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import time
import sys
import requests
from openai import OpenAI

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich.rule import Rule
from rich.align import Align

console = Console()

SERVERS = [
    {
        "label": "TurboQuant",
        "cache_flag": "turbo3 (128-dim dense mixing)",
        "url": "http://localhost:8082/v1",
        "color": "red",
        "icon": "🐌",
    },
    {
        "label": "RotorQuant",
        "cache_flag": "iso3 (4D Quaternion blocks)",
        "url": "http://localhost:8081/v1",
        "color": "cyan",
        "icon": "⚡",
    },
]

MODEL_NAME = "local"

CONTEXT_URLS = [
    (
        "main.py",
        "https://raw.githubusercontent.com/lcandy2/enable-chrome-ai/main/main.py",
    ),
    (
        "README.md",
        "https://raw.githubusercontent.com/lcandy2/enable-chrome-ai/main/README.md",
    ),
    (
        "uv.lock",
        "https://raw.githubusercontent.com/lcandy2/enable-chrome-ai/main/uv.lock",
    ),
]

# Simplified prompt to guarantee a response from Qwen
QUESTION = (
    "\n\n---\n"
    "I have provided the source code above. Please reply with exactly one sentence confirming "
    "that you have successfully received and read the code."
)

SYSTEM_PROMPT = "You are a helpful AI assistant."


def print_banner():
    parts = []
    for s in SERVERS:
        parts.append(f"[bold {s['color']}]{s['icon']} {s['label']}[/bold {s['color']}]")
    body = Text.from_markup("   vs   ".join(parts))
    console.print()
    console.print(
        Panel(
            Align.center(body),
            title="[bold white]Raw Algorithmic Benchmark (CPU)[/bold white]",
            border_style="bright_black",
            padding=(1, 4),
        )
    )
    console.print()


def fetch_context() -> str:
    console.print(
        Rule("[bold cyan]Fetching Context Files[/bold cyan]", style="bright_black")
    )
    console.print()
    parts = []
    total = 0
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as prog:
        task = prog.add_task("", total=len(CONTEXT_URLS))
        for fname, url in CONTEXT_URLS:
            prog.update(task, description=f"[cyan]Downloading[/cyan] {fname}...")
            r = requests.get(url, timeout=20)
            parts.append(f"\n\n### FILE: {fname}\n\n{r.text}")
            total += len(r.text)
            console.print(f"  [dim]✓ Fetched {fname}[/dim]")
            prog.advance(task)
    console.print(
        f"\n  [bold green]✓[/bold green] Total context loaded: [bold cyan]{total:,}[/bold cyan] characters\n"
    )
    return "".join(parts)


def run_benchmark(server: dict, prompt: str) -> dict:
    label, cache_flag, url, color, icon = (
        server["label"],
        server["cache_flag"],
        server["url"],
        server["color"],
        server["icon"],
    )

    console.print(
        Rule(
            f"[bold {color}]{icon} {label}[/bold {color}]  [dim]({cache_flag})[/dim]",
            style="bright_black",
        )
    )
    console.print()

    client = OpenAI(base_url=url, api_key="sk-no-key-required")
    t_start = time.perf_counter()

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            stream=True,
            temperature=0.1,
        )
    except Exception as e:
        console.print(
            f"[bold red]X Request failed. Is server {url} running?[/bold red]\n"
        )
        return {}

    first_chunk_time = None
    token_count = 0

    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    frame_i = 0

    for chunk in stream:
        now = time.perf_counter()

        # Safely extract text (handles newer Qwen streaming formats)
        delta_obj = chunk.choices[0].delta
        content = getattr(delta_obj, "content", "") or ""
        reasoning = getattr(delta_obj, "reasoning_content", "") or ""
        text = content + reasoning

        if text:
            if first_chunk_time is None:
                first_chunk_time = now
                prefill = first_chunk_time - t_start
                sys.stdout.write("\r" + " " * 60 + "\r")
                console.print(
                    f"[bold {color}]{icon} Prefill complete:[/bold {color}] [bold white]{prefill:.2f}s[/bold white]"
                )
                console.print(f"  [dim]Answer:[/dim] ", end="")

            token_count += 1
            sys.stdout.write(text)
            sys.stdout.flush()
        elif first_chunk_time is None:
            elapsed = now - t_start
            spin = spinner_frames[frame_i % len(spinner_frames)]
            frame_i += 1
            sys.stdout.write(
                f"\r  [{color}]{spin}[/{color}] Processing raw math... {elapsed:.1f}s  "
            )
            sys.stdout.flush()

    t_end = time.perf_counter()
    sys.stdout.write("\n")
    console.print()

    # Fallback if model instantly hits EOS
    if first_chunk_time is None:
        first_chunk_time = t_end
        console.print(
            "[red]Warning: Model generated 0 tokens (hit instant EOS). Prefill time estimated.[/red]"
        )

    dec_time = t_end - first_chunk_time
    stats = {
        "label": label,
        "cache": cache_flag,
        "color": color,
        "icon": icon,
        "prefill_s": first_chunk_time - t_start,
        "tokens": token_count,
        "decode_tps": token_count / dec_time if dec_time > 0 else 0,
    }

    console.print(
        Panel(
            Text.from_markup(
                f"[dim]Prefill Time:[/dim] [bold {color}]{stats['prefill_s']:.2f}s[/bold {color}]\n"
                f"  [dim]Decode Speed:[/dim][bold white]{stats['decode_tps']:.1f} tok/s[/bold white]\n"
                f"  [dim]Tokens Gen:[/dim]   [white]{stats['tokens']}[/white]"
            ),
            title=f"[bold {color}]{icon} {label} Stats[/bold {color}]",
            border_style=color,
            padding=(0, 2),
        )
    )
    console.print()
    return stats


def print_race(results: list):
    valid = [r for r in results if r]
    if len(valid) < 2:
        return

    console.print(
        Rule(
            "[bold yellow] 🏆 RAW ALGORITHMIC RACE RESULTS 🏆 [/bold yellow]",
            style="yellow",
        )
    )
    console.print()

    valid_sorted = sorted(valid, key=lambda r: r["prefill_s"])
    fastest = valid_sorted[0]
    slowest = valid_sorted[-1]
    speedup = (
        slowest["prefill_s"] / fastest["prefill_s"] if fastest["prefill_s"] > 0 else 1
    )

    BAR = 45
    bar_lines = [Text("  Prefill Race (Shorter bar = Faster)\n\n", style="bold white")]
    for r in valid:
        fraction = r["prefill_s"] / slowest["prefill_s"]
        filled = max(1, int(BAR * fraction))
        line = Text()
        line.append(f"  {r['icon']} {r['label']:<12}", style=f"bold {r['color']}")
        line.append("█" * filled, style=f"bold {r['color']}")
        line.append("░" * (BAR - filled), style="bright_black")
        line.append(f"  {r['prefill_s']:.2f}s", style=f"bold {r['color']}")
        if r["label"] == fastest["label"]:
            line.append("  (WINNER!)", style="bold green")
        bar_lines.append(line)
        bar_lines.append(Text(""))

    summary = Text(
        f"\n  X  {fastest['icon']} {fastest['label']} is {speedup:.1f}x faster at processing the raw math.\n",
        style="white",
    )
    bar_lines.append(summary)
    console.print(
        Panel(Text.assemble(*bar_lines), border_style="yellow", padding=(1, 2))
    )


def main():
    print_banner()
    prompt = fetch_context() + QUESTION
    results = [run_benchmark(s, prompt) for s in SERVERS]
    print_race(results)


if __name__ == "__main__":
    main()
