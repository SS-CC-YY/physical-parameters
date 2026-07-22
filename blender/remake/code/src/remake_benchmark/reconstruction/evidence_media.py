"""Small, defensive media helpers for paper-facing evidence reports.

The functions in this module are deliberately independent from the benchmark
grader.  They materialize already-produced evidence and render it for human
inspection; they never rerun tracking, reconstruction, or parameter fitting.

``cv2`` is an optional dependency and is imported only by
:func:`write_video_montage`.  Static diagnostics and fit cards therefore keep
working in CPU-only or minimal Python environments.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import shutil
import struct
import textwrap
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .evidence_formulas import summarize_fit_diagnostics


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# A compact uppercase bitmap font keeps placeholder images and diagnostic
# plots dependency-free.  Unsupported glyphs are represented by ``?`` rather
# than silently disappearing.
_FONT_5X7 = {
    " ": ("00000",) * 7,
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    ":": ("00000", "00110", "00110", "00000", "00110", "00110", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "<": ("00010", "00100", "01000", "10000", "01000", "00100", "00010"),
    ">": ("01000", "00100", "00010", "00001", "00010", "00100", "01000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00001", "00001", "00001", "00001", "10001", "10001", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed", "ok"}


class _Canvas:
    """Minimal RGB raster canvas used for dependency-free PNG evidence."""

    def __init__(self, width: int, height: int, colour: tuple[int, int, int]) -> None:
        self.width = int(width)
        self.height = int(height)
        self.data = bytearray(bytes(colour) * self.width * self.height)

    def pixel(self, x: int, y: int, colour: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = 3 * (y * self.width + x)
            self.data[offset : offset + 3] = bytes(colour)

    def rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        colour: tuple[int, int, int],
        *,
        fill: bool = True,
    ) -> None:
        if fill:
            left, right = max(0, x), min(self.width, x + width)
            top, bottom = max(0, y), min(self.height, y + height)
            row = bytes(colour) * max(0, right - left)
            for yy in range(top, bottom):
                offset = 3 * (yy * self.width + left)
                self.data[offset : offset + len(row)] = row
            return
        self.line(x, y, x + width - 1, y, colour)
        self.line(x, y, x, y + height - 1, colour)
        self.line(x + width - 1, y, x + width - 1, y + height - 1, colour)
        self.line(x, y + height - 1, x + width - 1, y + height - 1, colour)

    def line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        colour: tuple[int, int, int],
        *,
        width: int = 1,
    ) -> None:
        dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
        dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            radius = max(0, width // 2)
            for yy in range(y0 - radius, y0 + radius + 1):
                for xx in range(x0 - radius, x0 + radius + 1):
                    self.pixel(xx, yy, colour)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy

    def circle(self, cx: int, cy: int, radius: int, colour: tuple[int, int, int]) -> None:
        radius = max(1, radius)
        for yy in range(-radius, radius + 1):
            for xx in range(-radius, radius + 1):
                if xx * xx + yy * yy <= radius * radius:
                    self.pixel(cx + xx, cy + yy, colour)

    def text(
        self,
        x: int,
        y: int,
        value: Any,
        colour: tuple[int, int, int] = (28, 37, 54),
        *,
        scale: int = 2,
        max_chars: int | None = None,
    ) -> None:
        text = str(value).upper()
        if max_chars is not None:
            text = text[:max_chars]
        cursor = x
        for character in text:
            pattern = _FONT_5X7.get(character, _FONT_5X7["?"])
            for row, bits in enumerate(pattern):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        self.rect(
                            cursor + column * scale,
                            y + row * scale,
                            scale,
                            scale,
                            colour,
                        )
            cursor += 6 * scale

    def write_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        stride = self.width * 3
        for row in range(self.height):
            raw.append(0)
            start = row * stride
            raw.extend(self.data[start : start + stride])

        def chunk(kind: bytes, payload: bytes) -> bytes:
            checksum = zlib.crc32(kind)
            checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

        payload = bytearray(_PNG_SIGNATURE)
        payload.extend(
            chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
            )
        )
        payload.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=6)))
        payload.extend(chunk(b"IEND", b""))
        path.write_bytes(payload)


def write_placeholder_image(
    output: str | Path,
    title: str = "Evidence unavailable",
    message: str = "No renderable evidence was found.",
    *,
    width: int = 1280,
    height: int = 720,
) -> dict[str, Any]:
    """Write a valid dependency-free PNG explaining missing evidence."""

    destination = Path(output)
    width, height = max(320, int(width)), max(180, int(height))
    canvas = _Canvas(width, height, (246, 248, 252))
    canvas.rect(20, 20, width - 40, height - 40, (255, 255, 255))
    canvas.rect(20, 20, width - 40, height - 40, (180, 188, 202), fill=False)
    canvas.text(55, 62, title, (153, 27, 27), scale=3, max_chars=62)
    y = 128
    wrap = max(24, min(95, (width - 110) // 14))
    for line in textwrap.wrap(str(message), width=wrap)[:12]:
        canvas.text(56, y, line, (55, 65, 81), scale=2, max_chars=wrap)
        y += 34
    canvas.text(
        56,
        height - 65,
        "THIS PLACEHOLDER DOES NOT CHANGE THE BENCHMARK GRADE.",
        (95, 104, 119),
        scale=1,
        max_chars=80,
    )
    canvas.write_png(destination)
    return {
        "status": "ok",
        "kind": "placeholder_image",
        "output": str(destination),
        "width": width,
        "height": height,
        "title": str(title),
    }


def materialize_media(
    source: str | Path,
    destination: str | Path,
    mode: str = "hardlink",
) -> dict[str, Any]:
    """Materialize an existing media file without changing its contents.

    ``hardlink`` falls back to a metadata-preserving copy when links are not
    supported (for example across filesystems).  All errors are represented in
    the returned record so a report builder can continue and show a gap.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    requested = str(mode).strip().lower()
    if requested not in {"hardlink", "copy", "symlink"}:
        return {
            "status": "unavailable",
            "reason_code": "unsupported_materialization_mode",
            "source": str(source_path),
            "destination": str(destination_path),
            "mode_requested": requested,
        }
    if not source_path.is_file():
        return {
            "status": "unavailable",
            "reason_code": "source_media_missing",
            "source": str(source_path),
            "destination": str(destination_path),
            "mode_requested": requested,
        }
    try:
        if destination_path.exists() and os.path.samefile(source_path, destination_path):
            return {
                "status": "ok",
                "source": str(source_path),
                "destination": str(destination_path),
                "mode_requested": requested,
                "mode_used": "existing_same_file",
                "size_bytes": destination_path.stat().st_size,
            }
    except OSError:
        pass

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.{os.getpid()}.tmp")
    try:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        mode_used = requested
        if requested == "hardlink":
            try:
                os.link(source_path, temporary)
            except OSError:
                shutil.copy2(source_path, temporary)
                mode_used = "copy_fallback_from_hardlink"
        elif requested == "copy":
            shutil.copy2(source_path, temporary)
        else:
            try:
                temporary.symlink_to(source_path.resolve())
            except OSError:
                shutil.copy2(source_path, temporary)
                mode_used = "copy_fallback_from_symlink"
        os.replace(temporary, destination_path)
        return {
            "status": "ok",
            "source": str(source_path),
            "destination": str(destination_path),
            "mode_requested": requested,
            "mode_used": mode_used,
            "size_bytes": destination_path.stat().st_size,
        }
    except OSError as error:
        try:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass
        return {
            "status": "unavailable",
            "reason_code": "materialization_failed",
            "error": f"{type(error).__name__}: {error}",
            "source": str(source_path),
            "destination": str(destination_path),
            "mode_requested": requested,
        }


def _row_number(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = _number(row.get(name))
        if value is not None:
            return value
    return None


def _bounds(values: Sequence[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        pad = max(abs(low) * 0.05, 1.0)
    else:
        pad = 0.08 * (high - low)
    return low - pad, high + pad


def _draw_panel(
    canvas: _Canvas,
    box: tuple[int, int, int, int],
    title: str,
    points: Sequence[tuple[float, float, str]],
    *,
    x_label: str,
    y_label: str,
    invert_y: bool = False,
    missing_message: str = "COORDINATES NOT AVAILABLE",
) -> bool:
    x, y, width, height = box
    canvas.rect(x, y, width, height, (255, 255, 255))
    canvas.rect(x, y, width, height, (193, 200, 212), fill=False)
    canvas.text(x + 14, y + 12, title, (26, 38, 58), scale=2, max_chars=38)
    finite = [(a, b, state) for a, b, state in points if math.isfinite(a) and math.isfinite(b)]
    if not finite:
        canvas.text(x + 24, y + height // 2, missing_message, (145, 51, 51), scale=2, max_chars=42)
        return False
    left, top = x + 58, y + 54
    right, bottom = x + width - 24, y + height - 45
    x_low, x_high = _bounds([item[0] for item in finite])
    y_low, y_high = _bounds([item[1] for item in finite])
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        xx = left + round(fraction * (right - left))
        yy = top + round(fraction * (bottom - top))
        canvas.line(xx, top, xx, bottom, (229, 232, 238))
        canvas.line(left, yy, right, yy, (229, 232, 238))
    canvas.rect(left, top, right - left + 1, bottom - top + 1, (128, 139, 156), fill=False)

    def pixel(a: float, b: float) -> tuple[int, int]:
        px = left + round((a - x_low) / (x_high - x_low) * (right - left))
        fraction = (b - y_low) / (y_high - y_low)
        py = top + round(fraction * (bottom - top)) if invert_y else bottom - round(fraction * (bottom - top))
        return px, py

    previous: tuple[int, int] | None = None
    for a, b, state in finite:
        current = pixel(a, b)
        colour = {
            "invalid": (190, 45, 52),
            "interpolated": (22, 151, 181),
        }.get(state, (37, 99, 190))
        if previous is not None:
            canvas.line(previous[0], previous[1], current[0], current[1], colour, width=2)
        canvas.circle(current[0], current[1], 2 if state == "interpolated" else 3, colour)
        previous = current
    canvas.text(left, bottom + 18, x_label, (71, 82, 99), scale=1, max_chars=20)
    canvas.text(x + 8, top + 2, y_label, (71, 82, 99), scale=1, max_chars=18)
    canvas.text(left, y + height - 20, f"RANGE {x_low:.3G} TO {x_high:.3G}", (95, 105, 121), scale=1, max_chars=32)
    return True


def render_trajectory_diagnostics(
    csv_path: str | Path,
    output: str | Path,
    title: str = "",
    route: str = "",
) -> dict[str, Any]:
    """Render image-plane and metric trajectory projections to one PNG.

    Five evidence panels are emitted: image plane, world X-Z, and the X-Y,
    X-Z, Y-Z orthographic views used to audit a 3-D trajectory.  Missing
    coordinate families receive explicit in-image placeholders.
    """

    source = Path(csv_path)
    destination = Path(output)
    if not source.is_file():
        placeholder = write_placeholder_image(
            destination,
            "Trajectory evidence unavailable",
            f"CSV not found: {source}",
            width=1600,
            height=1000,
        )
        return {
            **placeholder,
            "status": "unavailable",
            "reason_code": "trajectory_csv_missing",
            "csv": str(source),
        }
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        placeholder = write_placeholder_image(
            destination,
            "Trajectory evidence unreadable",
            f"{type(error).__name__}: {error}",
            width=1600,
            height=1000,
        )
        return {
            **placeholder,
            "status": "unavailable",
            "reason_code": "trajectory_csv_unreadable",
            "error": str(error),
            "csv": str(source),
        }
    if not rows:
        placeholder = write_placeholder_image(
            destination,
            "Trajectory evidence empty",
            f"No frame rows were found in {source.name}.",
            width=1600,
            height=1000,
        )
        return {
            **placeholder,
            "status": "unavailable",
            "reason_code": "trajectory_csv_empty",
            "csv": str(source),
        }

    samples: list[dict[str, Any]] = []
    for row in rows:
        validity_keys = ("fit_eligible", "valid", "valid_3d", "valid_2d", "tracking_valid")
        validity_value = next((row.get(key) for key in validity_keys if key in row and row.get(key) not in (None, "")), None)
        valid = True if validity_value is None else _truth(validity_value)
        interpolated = _truth(row.get("interpolated"))
        samples.append(
            {
                "u": _row_number(row, ("center_u_px", "u_px", "center_x_px", "x_px")),
                "v": _row_number(row, ("center_v_px", "v_px", "center_y_px", "y_px")),
                "x": _row_number(row, ("fit_x_m", "x_m", "x_raw_m")),
                "y": _row_number(row, ("fit_y_m", "y_m", "y_raw_m")),
                "z": _row_number(row, ("fit_z_m", "z_m", "z_raw_m")),
                "state": "invalid" if not valid else "interpolated" if interpolated else "direct",
            }
        )

    def points(first: str, second: str) -> list[tuple[float, float, str]]:
        values = []
        for sample in samples:
            a, b = sample[first], sample[second]
            if a is not None and b is not None:
                values.append((float(a), float(b), str(sample["state"])))
        return values

    width, height = 1600, 1000
    canvas = _Canvas(width, height, (241, 244, 249))
    canvas.rect(20, 20, width - 40, height - 40, (250, 251, 253))
    display_title = title or source.stem
    canvas.text(44, 38, display_title, (16, 30, 53), scale=3, max_chars=72)
    canvas.text(46, 72, f"ROUTE: {route or 'UNSPECIFIED'} | ROWS: {len(rows)}", (74, 85, 104), scale=1, max_chars=100)
    panel_width, panel_height = 500, 405
    boxes = [
        (35, 105, panel_width, panel_height),
        (550, 105, panel_width, panel_height),
        (1065, 105, panel_width, panel_height),
        (35, 530, panel_width, panel_height),
        (550, 530, panel_width, panel_height),
    ]
    availability = {
        "image_plane_uv": _draw_panel(
            canvas,
            boxes[0],
            "IMAGE-PLANE TRAJECTORY",
            points("u", "v"),
            x_label="U (PX)",
            y_label="V (PX)",
            invert_y=True,
        ),
        "world_xz": _draw_panel(
            canvas,
            boxes[1],
            "WORLD MOTION PLANE X-Z",
            points("x", "z"),
            x_label="X (M)",
            y_label="Z (M)",
        ),
        "orthographic_xy": _draw_panel(
            canvas,
            boxes[2],
            "3D ORTHOGRAPHIC X-Y",
            points("x", "y"),
            x_label="X (M)",
            y_label="Y (M)",
        ),
        "orthographic_xz": _draw_panel(
            canvas,
            boxes[3],
            "3D ORTHOGRAPHIC X-Z",
            points("x", "z"),
            x_label="X (M)",
            y_label="Z (M)",
        ),
        "orthographic_yz": _draw_panel(
            canvas,
            boxes[4],
            "3D ORTHOGRAPHIC Y-Z",
            points("y", "z"),
            x_label="Y (M)",
            y_label="Z (M)",
        ),
    }
    summary_x, summary_y = 1065, 530
    canvas.rect(summary_x, summary_y, panel_width, panel_height, (255, 255, 255))
    canvas.rect(summary_x, summary_y, panel_width, panel_height, (193, 200, 212), fill=False)
    canvas.text(summary_x + 18, summary_y + 18, "EVIDENCE AVAILABILITY", (26, 38, 58), scale=2, max_chars=34)
    yy = summary_y + 70
    for name, available in availability.items():
        canvas.text(
            summary_x + 24,
            yy,
            f"{name}: {'AVAILABLE' if available else 'MISSING'}",
            (22, 111, 70) if available else (159, 44, 44),
            scale=1,
            max_chars=50,
        )
        yy += 30
    canvas.text(summary_x + 24, summary_y + 270, "BLUE: DIRECT / CYAN: INTERPOLATED", (74, 85, 104), scale=1, max_chars=50)
    canvas.text(summary_x + 24, summary_y + 295, "RED: INVALID / EXCLUDED FROM FIT", (159, 44, 44), scale=1, max_chars=50)
    canvas.text(summary_x + 24, summary_y + 330, "MISSING PANELS ARE NOT ZERO MOTION.", (159, 44, 44), scale=1, max_chars=50)
    canvas.text(summary_x + 24, summary_y + 360, "THEY ARE UNAVAILABLE MEASUREMENTS.", (159, 44, 44), scale=1, max_chars=50)
    canvas.write_png(destination)
    missing = [name for name, available in availability.items() if not available]
    return {
        "status": "ok" if not missing else "partial",
        "csv": str(source),
        "output": str(destination),
        "row_count": len(rows),
        "route": str(route),
        "panels": availability,
        "missing_panels": missing,
        "image_plane_point_count": len(points("u", "v")),
        "world_xz_point_count": len(points("x", "z")),
        "world_xyz_point_count": sum(
            sample["x"] is not None and sample["y"] is not None and sample["z"] is not None
            for sample in samples
        ),
        "direct_point_count": sum(sample["state"] == "direct" for sample in samples),
        "interpolated_point_count": sum(sample["state"] == "interpolated" for sample in samples),
        "invalid_point_count": sum(sample["state"] == "invalid" for sample in samples),
    }


def _load_mapping(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"expected a JSON object in {path}")
    return dict(loaded)


def _collect_fit_series(value: Any, path: str = "fit") -> list[tuple[str, dict[str, Any]]]:
    output: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key == "fit_series" and isinstance(item, Mapping):
                output.append((child_path, dict(item)))
            else:
                output.extend(_collect_fit_series(item, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            output.extend(_collect_fit_series(item, f"{path}[{index}]"))
    return output


def _series_values(series: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    time_values = series.get("time_s", [])
    observed = series.get("observed", [])
    predicted = series.get("predicted", [])
    residual = series.get("residual", [])
    if not all(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) for value in (time_values, observed, predicted)):
        return []
    length = min(len(time_values), len(observed), len(predicted))
    rows: list[tuple[float, float, float, float]] = []
    for index in range(length):
        t = _number(time_values[index])
        obs = _number(observed[index])
        pred = _number(predicted[index])
        supplied_residual = _number(residual[index]) if isinstance(residual, Sequence) and index < len(residual) else None
        if t is None or obs is None or pred is None:
            continue
        rows.append((t, obs, pred, obs - pred if supplied_residual is None else supplied_residual))
    return rows


def _svg_series_chart(series: Mapping[str, Any], title: str) -> str:
    rows = _series_values(series)
    if not rows:
        return '<div class="missing">No valid serialized fit-series samples.</div>'
    width, height = 900, 300
    left, right, top, mid, bottom = 65, 870, 42, 205, 270
    times = [row[0] for row in rows]
    values = [row[1] for row in rows] + [row[2] for row in rows]
    residuals = [row[3] for row in rows]
    t_low, t_high = _bounds(times)
    q_low, q_high = _bounds(values)
    r_low, r_high = _bounds(residuals)

    def point(t: float, q: float, low: float, high: float, y0: int, y1: int) -> tuple[float, float]:
        x = left + (t - t_low) / (t_high - t_low) * (right - left)
        y = y1 - (q - low) / (high - low) * (y1 - y0)
        return x, y

    observed = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(row[0], row[1], q_low, q_high, top, mid) for row in rows))
    predicted = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(row[0], row[2], q_low, q_high, top, mid) for row in rows))
    residual = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(row[0], row[3], r_low, r_high, mid + 24, bottom) for row in rows))
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}">'
        f'<rect width="{width}" height="{height}" fill="#fff"/>'
        f'<text x="18" y="24" font-family="sans-serif" font-size="14" font-weight="700">{html.escape(title)}</text>'
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{mid-top}" fill="#fafafa" stroke="#cbd5e1"/>'
        f'<polyline points="{observed}" fill="none" stroke="#2563eb" stroke-width="2"/>'
        f'<polyline points="{predicted}" fill="none" stroke="#ea580c" stroke-width="2"/>'
        f'<rect x="{left}" y="{mid+24}" width="{right-left}" height="{bottom-mid-24}" fill="#fafafa" stroke="#cbd5e1"/>'
        f'<polyline points="{residual}" fill="none" stroke="#64748b" stroke-width="1.5"/>'
        '<text x="70" y="58" font-family="sans-serif" font-size="11" fill="#2563eb">observed</text>'
        '<text x="145" y="58" font-family="sans-serif" font-size="11" fill="#ea580c">fitted</text>'
        f'<text x="{left}" y="{height-8}" font-family="sans-serif" font-size="10" fill="#475569">time: {times[0]:.4g} to {times[-1]:.4g}; residual shown below</text>'
        "</svg>"
    )


def _formula_lines(formula_spec: Mapping[str, Any] | str | Path) -> tuple[str, list[str], list[str]]:
    if isinstance(formula_spec, Mapping):
        spec = dict(formula_spec)
    elif isinstance(formula_spec, Path) or _is_file_path(formula_spec):
        spec = _load_mapping(formula_spec)
    else:
        return "Formula", [str(formula_spec)], []
    title = str(spec.get("title") or spec.get("name") or spec.get("equation_id") or "Formula")
    raw_equations = spec.get("equations", spec.get("equation", []))
    if isinstance(raw_equations, str):
        equations = [raw_equations]
    elif isinstance(raw_equations, Sequence):
        equations = [str(value) for value in raw_equations]
    else:
        equations = []
    notes = []
    for key in ("observables", "parameter_transform", "notes"):
        value = spec.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            notes.extend(f"{key}: {item}" for item in value)
        else:
            notes.append(f"{key}: {value}")
    return title, equations, notes


def _is_file_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return Path(value).is_file()
    except OSError:
        return False


def render_fit_process_card(
    result: Mapping[str, Any] | str | Path,
    output: str | Path,
    formula_spec: Mapping[str, Any] | str | Path,
    grade: str | None = None,
) -> dict[str, Any]:
    """Render a self-contained HTML/Markdown card from serialized fit data.

    No model is rerun and no coefficient is inferred.  Curves are drawn only
    when the supplied result contains an explicit ``fit_series`` object.
    """

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = _load_mapping(result)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        text = f"Fit result unavailable: {type(error).__name__}: {error}"
        if destination.suffix.lower() == ".md":
            destination.write_text(f"# Fit process unavailable\n\n{text}\n", encoding="utf-8")
        else:
            destination.write_text(f"<!doctype html><meta charset='utf-8'><h1>Fit process unavailable</h1><p>{html.escape(text)}</p>", encoding="utf-8")
        return {
            "status": "unavailable",
            "reason_code": "fit_result_unreadable",
            "error": str(error),
            "output": str(destination),
        }
    fit_value = payload.get("fit")
    fit = dict(fit_value) if isinstance(fit_value, Mapping) else payload
    formula_title, equations, formula_notes = _formula_lines(formula_spec)
    collected = _collect_fit_series(fit)
    series_records = []
    for path, series in collected:
        rows = _series_values(series)
        if not rows:
            continue
        rmse = math.sqrt(sum(row[3] ** 2 for row in rows) / len(rows))
        series_records.append(
            {
                "path": path,
                "name": str(series.get("series_name") or path),
                "series": series,
                "rows": rows,
                "rmse_from_serialized_series": rmse,
            }
        )
    estimates = fit.get("parameter_estimates", {})
    if not isinstance(estimates, Mapping):
        estimates = {}
    fit_status = str(fit.get("status") or "unknown")
    method = str(fit.get("method") or "not recorded")
    target_not_used = fit.get("target_not_used_for_fit")
    diagnostics = fit.get("diagnostics") if isinstance(fit.get("diagnostics"), Mapping) else {}
    diagnostic_summary = summarize_fit_diagnostics(diagnostics)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    parameter_metrics = metrics.get("parameters") if isinstance(metrics.get("parameters"), Mapping) else {}
    warning = None if series_records else (
        "No serialized fit_series was found. No observed/fitted curve or intermediate fit value was invented."
    )

    if destination.suffix.lower() == ".md":
        lines = [
            f"# {formula_title}",
            "",
            f"- Grade: `{grade or 'not supplied'}`",
            f"- Fit status: `{fit_status}`",
            f"- Method: `{method}`",
            f"- Target parameter used during fit: `{'no' if target_not_used is True else 'not verified'}`",
            "",
            "## Formula supplied by the benchmark",
            "",
        ]
        if equations:
            lines.extend(f"- `{equation}`" for equation in equations)
        else:
            lines.append("- Formula was not supplied; no equation was inferred.")
        lines.extend(["", *[f"- {note}" for note in formula_notes], "", "## Recovered parameters", "", "|Parameter|Recovered|", "|---|---:|"])
        lines.extend(f"|{name}|{value}|" for name, value in estimates.items())
        lines.extend(["", "## Requested versus recovered (post-fit only)", "", "|Parameter|Target|Recovered|Valid-range NAE|", "|---|---:|---:|---:|"])
        if parameter_metrics:
            for name, values in parameter_metrics.items():
                values = values if isinstance(values, Mapping) else {}
                lines.append(
                    f"|{name}|{values.get('gt', 'N/A')}|{values.get('estimate_raw', estimates.get(name, 'N/A'))}|"
                    f"{values.get('normalized_absolute_error', 'N/A')}|"
                )
        else:
            lines.append("|N/A|N/A|N/A|N/A|")
        lines.extend(
            [
                "",
                "## Serialized fit quality",
                "",
                f"- Minimum fit points: `{diagnostic_summary.get('minimum_fit_points')}`",
                f"- Worst serialized fit NRMSE: `{diagnostic_summary.get('worst_fit_nrmse')}`",
            ]
        )
        for item in diagnostic_summary.get("intermediates", []):
            lines.append(f"- `{item.get('name')}` = `{item.get('value')}` ({item.get('path')})")
        if warning:
            lines.extend(["", f"> {warning}"])
        for record in series_records:
            lines.extend(
                [
                    "",
                    f"## Serialized fit series: {record['name']}",
                    "",
                    f"Path: `{record['path']}`; points: {len(record['rows'])}; display RMSE: {record['rmse_from_serialized_series']:.6g}",
                    "",
                    "|t|observed|fitted|residual|",
                    "|---:|---:|---:|---:|",
                ]
            )
            sample = record["rows"] if len(record["rows"]) <= 20 else record["rows"][:10] + record["rows"][-10:]
            lines.extend(f"|{t:.6g}|{observed:.6g}|{predicted:.6g}|{residual:.6g}|" for t, observed, predicted, residual in sample)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        equation_html = "".join(f"<li><code>{html.escape(equation)}</code></li>" for equation in equations)
        if not equation_html:
            equation_html = "<li>Formula was not supplied; no equation was inferred.</li>"
        note_html = "".join(f"<li>{html.escape(note)}</li>" for note in formula_notes)
        estimate_html = "".join(
            f"<tr><td>{html.escape(str(name))}</td><td>{html.escape(str(value))}</td></tr>"
            for name, value in estimates.items()
        ) or "<tr><td colspan='2'>No recovered parameter was serialized.</td></tr>"
        metric_html = "".join(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{html.escape(str(values.get('gt', 'N/A') if isinstance(values, Mapping) else 'N/A'))}</td>"
            f"<td>{html.escape(str(values.get('estimate_raw', estimates.get(name, 'N/A')) if isinstance(values, Mapping) else estimates.get(name, 'N/A')))}</td>"
            f"<td>{html.escape(str(values.get('normalized_absolute_error', 'N/A') if isinstance(values, Mapping) else 'N/A'))}</td>"
            "</tr>"
            for name, values in parameter_metrics.items()
        ) or "<tr><td colspan='4'>Target/error metrics were not serialized; nothing was inferred.</td></tr>"
        intermediate_html = "".join(
            f"<li><code>{html.escape(str(item.get('name')))}</code> = {html.escape(str(item.get('value')))} "
            f"<small>({html.escape(str(item.get('path')))})</small></li>"
            for item in diagnostic_summary.get("intermediates", [])
        ) or "<li>No readable intermediate scalar was serialized.</li>"
        charts = []
        for record in series_records:
            chart = _svg_series_chart(record["series"], str(record["name"]))
            rows = record["rows"]
            sample = rows if len(rows) <= 40 else rows[:20] + rows[-20:]
            table_rows = "".join(
                f"<tr><td>{t:.6g}</td><td>{observed:.6g}</td><td>{predicted:.6g}</td><td>{residual:.6g}</td></tr>"
                for t, observed, predicted, residual in sample
            )
            charts.append(
                f"<section><h2>{html.escape(str(record['name']))}</h2>{chart}"
                f"<p>Serialized path: <code>{html.escape(str(record['path']))}</code>; "
                f"points: {len(rows)}; display RMSE: {record['rmse_from_serialized_series']:.6g}</p>"
                "<details><summary>Show serialized samples</summary><table><thead><tr>"
                "<th>t</th><th>observed</th><th>fitted</th><th>residual</th></tr></thead>"
                f"<tbody>{table_rows}</tbody></table></details></section>"
            )
        warning_html = f"<p class='warning'>{html.escape(warning)}</p>" if warning else ""
        target_label = (
            "no" if target_not_used is True
            else "yes" if target_not_used is False
            else "not verified"
        )
        destination.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(formula_title)}</title><style>"
            "body{font:15px/1.5 Arial,sans-serif;color:#172033;max-width:1050px;margin:32px auto;padding:0 24px}"
            "h1,h2{color:#0f2747}code{background:#f1f5f9;padding:2px 5px}"
            "table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #cbd5e1;padding:6px;text-align:left}"
            ".meta{background:#eff6ff;padding:12px 16px;border-left:4px solid #2563eb}"
            ".warning,.missing{background:#fff7ed;color:#9a3412;padding:10px;border-left:4px solid #f97316}"
            "svg{width:100%;height:auto;border:1px solid #e2e8f0}</style></head><body>"
            f"<h1>{html.escape(formula_title)}</h1>"
            f"<div class='meta'><b>Grade:</b> {html.escape(grade or 'not supplied')} · "
            f"<b>Fit status:</b> {html.escape(fit_status)} · <b>Method:</b> {html.escape(method)} · "
            f"<b>Target used during fit:</b> {target_label}</div>"
            f"<h2>Formula supplied by the benchmark</h2><ul>{equation_html}{note_html}</ul>"
            f"<h2>Recovered parameters</h2><table><tr><th>Parameter</th><th>Recovered</th></tr>{estimate_html}</table>"
            f"<h2>Requested versus recovered (post-fit only)</h2><table><tr><th>Parameter</th><th>Target</th><th>Recovered</th><th>Valid-range NAE</th></tr>{metric_html}</table>"
            f"<h2>Serialized fit quality</h2><ul><li>Minimum fit points: {html.escape(str(diagnostic_summary.get('minimum_fit_points')))}</li>"
            f"<li>Worst serialized fit NRMSE: {html.escape(str(diagnostic_summary.get('worst_fit_nrmse')))}</li>{intermediate_html}</ul>"
            f"{warning_html}{''.join(charts)}"
            "<p><small>This card only visualizes serialized fit evidence; it does not rerun or modify the evaluator.</small></p>"
            "</body></html>",
            encoding="utf-8",
        )
    return {
        "status": "ok" if series_records else "partial",
        "output": str(destination),
        "fit_status": fit_status,
        "method": method,
        "grade": grade,
        "target_not_used_for_fit": target_not_used is True,
        "parameter_estimates": dict(estimates),
        "series_count": len(series_records),
        "series_names": [record["name"] for record in series_records],
        "warning": warning,
        "diagnostic_summary": diagnostic_summary,
        "parameter_metric_count": len(parameter_metrics),
    }


def write_video_montage(
    sources: Sequence[str | Path],
    labels: Sequence[str],
    output: str | Path,
    fps: int | float = 16,
    tile_width: int = 480,
) -> dict[str, Any]:
    """Write a synchronized comparison montage using lazily imported OpenCV.

    Failures are returned as ``status='unavailable'`` and accompanied by a PNG
    placeholder.  The montage ends at the shortest input duration and samples
    every source by timestamp, so no input motion is time-warped.
    """

    source_paths = [Path(value) for value in sources]
    destination = Path(output)
    placeholder = destination.with_name(f"{destination.stem}_unavailable.png")

    def unavailable(reason: str, message: str, *, error: str | None = None) -> dict[str, Any]:
        placeholder_result = write_placeholder_image(
            placeholder,
            "Video montage unavailable",
            message,
            width=max(960, int(tile_width) * 2),
            height=540,
        )
        result = {
            "status": "unavailable",
            "reason_code": reason,
            "output": None,
            "requested_output": str(destination),
            "sources": [str(path) for path in source_paths],
            "placeholder_image": placeholder_result["output"],
        }
        if error:
            result["error"] = error
        return result

    if not source_paths:
        return unavailable("no_video_sources", "No video sources were supplied.")
    if len(source_paths) != len(labels):
        return unavailable(
            "source_label_count_mismatch",
            f"Received {len(source_paths)} sources but {len(labels)} labels.",
        )
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        return unavailable("video_source_missing", "Missing source(s): " + "; ".join(missing))
    output_fps = _number(fps)
    if output_fps is None or output_fps <= 0:
        return unavailable("invalid_output_fps", f"fps must be positive; received {fps!r}.")
    if int(tile_width) < 64:
        return unavailable("invalid_tile_width", f"tile_width must be at least 64; received {tile_width!r}.")

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as error:  # pragma: no cover - environment dependent
        return unavailable(
            "opencv_unavailable",
            "OpenCV is not installed or could not be imported; individual videos remain authoritative.",
            error=f"{type(error).__name__}: {error}",
        )

    captures = [cv2.VideoCapture(str(path)) for path in source_paths]
    writer = None
    try:
        if any(not capture.isOpened() for capture in captures):
            return unavailable("video_decode_open_failed", "At least one source could not be opened by OpenCV.")
        metadata = []
        for path, capture in zip(source_paths, captures):
            native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = frame_count / native_fps if native_fps > 0 and frame_count > 0 else 0.0
            metadata.append(
                {
                    "source": str(path),
                    "fps": native_fps,
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "duration_s": duration,
                }
            )
        if any(item["duration_s"] <= 0 or item["width"] <= 0 or item["height"] <= 0 for item in metadata):
            return unavailable("video_metadata_unavailable", "A source has no trustworthy FPS, duration, or dimensions.")
        duration = min(float(item["duration_s"]) for item in metadata)
        output_frames = max(1, int(math.floor(duration * output_fps + 1e-9)))
        tile_width = int(tile_width)
        tile_height = max(64, int(round(tile_width * 9 / 16)))
        label_height = 68
        if len(source_paths) <= 3:
            columns = len(source_paths)
        else:
            columns = 2
        rows = int(math.ceil(len(source_paths) / columns))
        canvas_size = (columns * tile_width, rows * (tile_height + label_height))
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(output_fps),
            canvas_size,
        )
        if not writer.isOpened():
            return unavailable("video_writer_unavailable", "OpenCV could not create an MP4V video writer.")

        decoded_frames = 0
        for frame_index in range(output_frames):
            timestamp_s = frame_index / float(output_fps)
            canvas = np.full((canvas_size[1], canvas_size[0], 3), 18, dtype=np.uint8)
            failed = False
            for index, (capture, label) in enumerate(zip(captures, labels)):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    failed = True
                    break
                source_height, source_width = frame.shape[:2]
                scale = min(tile_width / source_width, tile_height / source_height)
                resized_width = max(1, int(round(source_width * scale)))
                resized_height = max(1, int(round(source_height * scale)))
                resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
                tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                offset_x = (tile_width - resized_width) // 2
                offset_y = (tile_height - resized_height) // 2
                tile[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
                row, column = divmod(index, columns)
                x = column * tile_width
                y = row * (tile_height + label_height)
                canvas[y : y + tile_height, x : x + tile_width] = tile
                cv2.rectangle(
                    canvas,
                    (x, y + tile_height),
                    (x + tile_width - 1, y + tile_height + label_height - 1),
                    (35, 39, 48),
                    -1,
                )
                lines = textwrap.wrap(str(label), width=max(18, tile_width // 12))[:2]
                for line_number, line in enumerate(lines):
                    cv2.putText(
                        canvas,
                        line,
                        (x + 12, y + tile_height + 25 + line_number * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (245, 245, 245),
                        1,
                        cv2.LINE_AA,
                    )
            if failed:
                break
            writer.write(canvas)
            decoded_frames += 1
        writer.release()
        writer = None
        if decoded_frames == 0 or not destination.is_file() or destination.stat().st_size == 0:
            return unavailable("montage_render_failed", "No synchronized montage frame could be written.")

        verification = cv2.VideoCapture(str(destination))
        verified = verification.isOpened()
        verified_frames = int(verification.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if verified else 0
        verification.release()
        if not verified or verified_frames <= 0:
            return unavailable("montage_verification_failed", "The written montage could not be decoded again.")
        return {
            "status": "ok",
            "output": str(destination),
            "codec": "mp4v",
            "fps": float(output_fps),
            "duration_s": decoded_frames / float(output_fps),
            "frame_count": decoded_frames,
            "verified_frame_count": verified_frames,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "columns": columns,
            "rows": rows,
            "source_metadata": metadata,
            "time_alignment": "timestamp_sampled_no_time_warp_shortest_duration",
        }
    except Exception as error:  # pragma: no cover - backend-specific failures
        return unavailable(
            "montage_runtime_error",
            "OpenCV raised an error while rendering the synchronized montage.",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        if writer is not None:
            writer.release()
        for capture in captures:
            capture.release()


__all__ = [
    "materialize_media",
    "render_fit_process_card",
    "render_trajectory_diagnostics",
    "write_placeholder_image",
    "write_video_montage",
]
