"""Minimal YAML loader/dumper for simple configuration files."""
from __future__ import annotations

from typing import Any, Dict, List


def safe_load(stream: str) -> Any:
    if hasattr(stream, 'read'):
        text = stream.read()
    else:
        text = stream
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.strip().startswith('#')]
    index = 0

    def parse_block(indent: int = 0) -> Any:
        nonlocal index
        mapping: Dict[str, Any] = {}
        sequence: List[Any] = []
        is_sequence = False
        while index < len(lines):
            line = lines[index]
            current_indent = len(line) - len(line.lstrip(' '))
            if current_indent < indent:
                break
            stripped = line.strip()
            if stripped.startswith('- '):
                is_sequence = True
                value_text = stripped[2:].strip()
                index += 1
                if value_text:
                    sequence.append(_convert_scalar(value_text))
                else:
                    sequence.append(parse_block(current_indent + 2))
                continue
            if ':' in stripped:
                key, value_text = stripped.split(':', 1)
                key = key.strip()
                value_text = value_text.strip()
                index += 1
                if value_text:
                    mapping[key] = _convert_scalar(value_text)
                else:
                    mapping[key] = parse_block(current_indent + 2)
                continue
            index += 1
        if is_sequence:
            return sequence
        return mapping

    def _convert_scalar(token: str) -> Any:
        lowered = token.lower()
        if lowered == 'true':
            return True
        if lowered == 'false':
            return False
        if lowered == 'null':
            return None
        try:
            if '.' in token:
                return float(token)
            return int(token)
        except ValueError:
            return token.strip('"')

    return parse_block(0)


def safe_dump(data: Any, stream=None) -> str:
    def render(obj: Any, indent: int = 0) -> List[str]:
        prefix = ' ' * indent
        lines: List[str] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(render(value, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {format_scalar(value)}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.extend(render(item, indent + 2))
                else:
                    lines.append(f"{prefix}- {format_scalar(item)}")
        else:
            lines.append(f"{prefix}{format_scalar(obj)}")
        return lines

    def format_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if value is None:
            return 'null'
        return str(value)

    content = '\n'.join(render(data, 0)) + '\n'
    if stream is not None:
        stream.write(content)
        return ''
    return content

