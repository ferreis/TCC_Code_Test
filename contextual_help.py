"""Presentation-only contextual help; keep paired projects' glossary in sync.

Widget labels use Streamlit's native help icon. Prose uses accessible inline ⓘ
icons; tables preserve their original keys and values and add header tooltips.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

TERMS = json.loads(Path(__file__).with_name('technical_terms.json').read_text(encoding='utf-8'))
_ALIASES = {alias.casefold(): key for key, term in TERMS.items() for alias in term['aliases']}
_PATTERN = re.compile(
    r'(?<![^\W_])(?:' + '|'.join(re.escape(alias) for alias in sorted(_ALIASES, key=len, reverse=True))
    + r')(?![^\W_])', re.IGNORECASE,
)


def help_for(label: str) -> str | None:
    """Resolve UI labels and flattened result columns without changing values."""
    label = str(label)
    exact = _ALIASES.get(label.casefold())
    if exact:
        return TERMS[exact]['text']
    keys = list(dict.fromkeys(_ALIASES[m.group().casefold()] for m in _PATTERN.finditer(label)))
    if not keys:
        return None
    if label.startswith('speed_ms.'):
        return TERMS['latency']['text']
    text = '\n\n'.join(TERMS[key]['text'] for key in keys[:2])
    if label.endswith(('_box', '(B)')):
        text += ' Nesta coluna, a avaliação usa caixas delimitadoras.'
    elif label.endswith(('_mask', '(M)')):
        text += ' Nesta coluna, a avaliação usa máscaras de pixels.'
    return text


_STYLE = """<style>
.tcc-help { display:inline-block; position:relative; vertical-align:baseline; }
.tcc-help summary { display:inline-block; cursor:help; color:inherit; opacity:.8;
  list-style:none; font-size:.85em; padding:0 .15em; }
.tcc-help summary::-webkit-details-marker { display:none; }
.tcc-help summary:focus-visible { outline:2px solid currentColor; border-radius:3px; }
.tcc-help .tcc-tip { display:none; position:absolute; z-index:999; left:0; top:1.5em;
  width:min(310px,65vw); padding:.65rem .8rem; border:1px solid #64748b;
  border-radius:.5rem; background:var(--secondary-background-color, #1e293b);
  color:var(--text-color, #f8fafc); font-size:14px; line-height:1.5;
  font-weight:normal; box-shadow:0 4px 12px #0003; white-space:normal; }
.tcc-help:hover .tcc-tip, .tcc-help:focus-within .tcc-tip,
.tcc-help[open] .tcc-tip { display:block; }
.tcc-help-caption { font-size:.875rem; }
</style>"""


def annotated_html(text: str) -> str:
    """Escape all user/process text; only our own tooltip markup becomes HTML."""
    parts, end = [], 0
    for match in _PATTERN.finditer(str(text)):
        term = match.group()
        definition = TERMS[_ALIASES[term.casefold()]]['text']
        parts.append(html.escape(text[end:match.end()]))
        parts.append(
            '<details class="tcc-help"><summary aria-label="Ajuda sobre '
            + html.escape(term, quote=True) + '" title="' + html.escape(definition, quote=True)
            + '">ⓘ</summary><span class="tcc-tip">' + html.escape(definition) + '</span></details>'
        )
        end = match.end()
    parts.append(html.escape(text[end:]))
    return ''.join(parts)


def help_text(text: str, *, caption: bool = False, target: Any = None) -> None:
    css_class = 'tcc-help-caption' if caption else 'tcc-help-text'
    (target if target is not None else st).markdown(
        _STYLE + f'<div class="{css_class}">' + annotated_html(str(text)) + '</div>',
        unsafe_allow_html=True,
    )


def help_columns(columns: Any) -> dict[str, Any]:
    """Generic Column metadata preserves inferred number/bool editor types."""
    return {
        str(column): st.column_config.Column(str(column) + ' ⓘ', help=description)
        for column in columns
        if (description := help_for(str(column)))
    }


def help_dataframe(data: Any, **kwargs: Any) -> Any:
    configured = help_columns(data.columns)
    configured.update(kwargs.pop('column_config', {}) or {})
    return st.dataframe(data, column_config=configured, **kwargs)


def help_editor(data: Any, **kwargs: Any) -> Any:
    configured = help_columns(data.columns)
    configured.update(kwargs.pop('column_config', {}) or {})
    return st.data_editor(data, column_config=configured, **kwargs)


def help_json(data: Any) -> None:
    """Keep the original JSON view and offer help for its actual fields nearby."""
    with st.popover('ⓘ Campos deste resultado'):
        fields = data.keys() if isinstance(data, dict) else []
        for key in fields:
            if description := help_for(str(key)):
                st.markdown(f'**{key}** — {description}')
    st.json(data)
