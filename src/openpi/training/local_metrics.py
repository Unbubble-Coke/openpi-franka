from __future__ import annotations

import csv
import html
import logging
import math
import pathlib
import time
from collections.abc import Mapping
from typing import Any


class LocalMetricPlotter:
    """Write training metrics to CSV and refresh a small local HTML/SVG dashboard."""

    def __init__(self, output_dir: pathlib.Path | str, *, refresh_seconds: int = 10):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "metrics.csv"
        self.html_path = self.output_dir / "index.html"
        self.refresh_seconds = refresh_seconds
        self.rows: list[dict[str, float]] = []
        self.keys: list[str] = []
        self._load_existing()
        self._write_html()
        logging.info("Local metric dashboard: %s", self.html_path)

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        row: dict[str, float] = {"step": float(step)}
        for key, value in metrics.items():
            parsed = _to_float(value)
            if parsed is not None and math.isfinite(parsed):
                row[str(key)] = parsed

        if len(row) <= 1:
            return

        for key in row:
            if key != "step" and key not in self.keys:
                self.keys.append(key)

        self.rows.append(row)
        self._write_csv()
        self._write_svgs()
        self._write_html()

    def _load_existing(self) -> None:
        if not self.csv_path.exists():
            return
        try:
            with self.csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return
                self.keys = [k for k in reader.fieldnames if k != "step"]
                for raw_row in reader:
                    row: dict[str, float] = {}
                    for key, value in raw_row.items():
                        parsed = _to_float(value)
                        if parsed is not None and math.isfinite(parsed):
                            row[key] = parsed
                    if "step" in row:
                        self.rows.append(row)
        except Exception as exc:  # pragma: no cover - best effort diagnostics only.
            logging.warning("Failed to load existing local metrics from %s: %s", self.csv_path, exc)
            self.rows = []
            self.keys = []

    def _write_csv(self) -> None:
        fieldnames = ["step", *self.keys]
        tmp_path = self.csv_path.with_suffix(".tmp")
        with tmp_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        tmp_path.replace(self.csv_path)

    def _write_svgs(self) -> None:
        for key in self.keys:
            points = [(row["step"], row[key]) for row in self.rows if key in row]
            if points:
                (self.output_dir / f"{_safe_name(key)}.svg").write_text(_line_svg(points, title=key))

    def _write_html(self) -> None:
        chart_cards = []
        for key in self.keys or ["loss"]:
            svg_name = f"{_safe_name(key)}.svg"
            latest = _latest_value(self.rows, key)
            latest_text = "n/a" if latest is None else f"{latest:.6g}"
            chart_cards.append(
                f"""
                <section class="card">
                  <header>
                    <h2>{html.escape(key)}</h2>
                    <span>latest: {html.escape(latest_text)}</span>
                  </header>
                  <img src="{html.escape(svg_name)}?t={int(time.time())}" alt="{html.escape(key)} chart">
                </section>
                """
            )

        latest_step = int(self.rows[-1]["step"]) if self.rows else "n/a"
        self.html_path.write_text(
            f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{self.refresh_seconds}">
  <title>Training Metrics</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    .top {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 18px;
    }}
    h1 {{
      font-size: 22px;
      margin: 0;
      font-weight: 650;
    }}
    .meta {{
      color: #52606d;
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 14px;
    }}
    .card header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 10px;
    }}
    h2 {{
      font-size: 15px;
      margin: 0;
    }}
    .card span {{
      color: #52606d;
      font-size: 12px;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    a {{
      color: #1266f1;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <main>
    <div class="top">
      <h1>Training Metrics</h1>
      <div class="meta">step: {latest_step} | auto refresh: {self.refresh_seconds}s | <a href="metrics.csv">metrics.csv</a></div>
    </div>
    <div class="grid">
      {''.join(chart_cards)}
    </div>
  </main>
</body>
</html>
"""
        )


def _to_float(value: Any) -> float | None:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _latest_value(rows: list[dict[str, float]], key: str) -> float | None:
    for row in reversed(rows):
        if key in row:
            return row[key]
    return None


def _line_svg(points: list[tuple[float, float]], *, title: str, width: int = 760, height: int = 320) -> str:
    margin_left, margin_right, margin_top, margin_bottom = 60, 24, 30, 42
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        pad = abs(y_min) * 0.1 if y_min else 1.0
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_height

    path = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    y_ticks = _ticks(y_min, y_max, 5)
    x_ticks = _ticks(x_min, x_max, 5)

    grid = []
    for tick in y_ticks:
        y = sy(tick)
        grid.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#edf2f7"/>')
        grid.append(f'<text x="{margin_left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#52606d">{tick:.4g}</text>')
    for tick in x_ticks:
        x = sx(tick)
        grid.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#edf2f7"/>')
        grid.append(f'<text x="{x:.2f}" y="{height - 16}" text-anchor="middle" font-size="11" fill="#52606d">{tick:.0f}</text>')

    latest_x, latest_y = points[-1]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin_left}" y="20" font-size="14" font-weight="600" fill="#1f2933">{html.escape(title)}</text>
  {''.join(grid)}
  <polyline points="{path}" fill="none" stroke="#1266f1" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{sx(latest_x):.2f}" cy="{sy(latest_y):.2f}" r="4" fill="#1266f1"/>
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9fb3c8"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9fb3c8"/>
  <text x="{width - margin_right}" y="{height - 4}" text-anchor="end" font-size="11" fill="#52606d">step</text>
</svg>
"""


def _ticks(v_min: float, v_max: float, count: int) -> list[float]:
    if count <= 1:
        return [v_min]
    return [v_min + (v_max - v_min) * i / (count - 1) for i in range(count)]
