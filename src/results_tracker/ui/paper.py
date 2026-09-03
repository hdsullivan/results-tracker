"""Paper page: the pinned assets of a project, in manuscript order, with their status and whether the last
export still matches the database; edit bookkeeping, open an asset in the Export page, export the paper.

Nothing here renders a table or figure by hand: `export.paper.render_paper` regenerates every asset from its
stored spec, exactly as `results-tracker export paper` does.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import streamlit as st

from .. import aggregate as agg
from ..api import delete_asset, list_assets, update_asset
from ..export.paper import EXPORT_STATUSES, KIND_TITLES, mark_exported, render_paper, write_paper, zip_paper
from ..export.paper import staleness as asset_staleness
from .common import db_path, engine_for, keyed, keyed_selectbox, load_records, page_url, select_project, sidebar_db
from .tables import generic_html

STATE_HELP = ("never exported: pinned but not yet in a paper export · current: the exported files were rendered from exactly "
              "these runs · stale: runs were added, replaced or deleted since the export · no data: no completed run matches")


def _fmt_ts(ts: Any) -> str:
    try:
        return (ts.astimezone() if ts.tzinfo is not None else ts).strftime("%Y-%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        return "—"


def render() -> None:
    st.title("Paper")
    sidebar_db()
    project = select_project()
    if project is None:
        return
    engine = engine_for(db_path())
    assets = list_assets(project, engine=engine)
    if not assets:
        st.info("No assets pinned yet. Open a Comparison, Sweep, Ablation, Visual or Export view and use **Pin to paper**; "
                "everything pinned here is what `results-tracker export paper` regenerates.")
        return

    states = {a.label: asset_staleness(a, load_records(project, a.experiment)) for a in assets}
    counts = Counter(s for s, _ in states.values())
    active = [a for a in assets if a.status.value != "dropped"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets", f"{len(active)}" + (f" (+{len(assets) - len(active)} dropped)" if len(assets) > len(active) else ""))
    c2.metric("Final", sum(a.status.value == "final" for a in assets))
    c3.metric("Stale", counts.get("stale", 0), help=STATE_HELP)
    c4.metric("Never exported", counts.get("never exported", 0), help=STATE_HELP)

    rows = []
    for a in assets:
        state, detail = states[a.label]
        rows.append([a.position, a.label, KIND_TITLES.get(a.kind, a.kind), a.status.value, a.experiment,
                     agg.where_text(a.filters or {}) or "—", state, detail, _fmt_ts(a.exported_at) if a.exported_at else "—"])
    st.markdown(generic_html(["#", "Label", "Kind", "Status", "Experiment", "Filter", "State", "Detail", "Exported"], rows,
                             caption=f"Paper assets of {project} in manuscript order. {STATE_HELP}.", number=1, left_cols=9),
                unsafe_allow_html=True)

    st.subheader("Asset")
    labels = [a.label for a in assets]
    label = keyed_selectbox("Asset", labels, "paper_asset", labels[0])
    a = next(x for x in assets if x.label == label)
    _asset_detail(a, project, engine)

    st.divider()
    _export_section(project, engine)


def _asset_detail(a, project: str, engine) -> None:
    state, detail = asset_staleness(a, load_records(project, a.experiment))
    st.caption(f"{KIND_TITLES.get(a.kind, a.kind)} of **{a.experiment}**"
               + (f" with filter {agg.where_text(a.filters)}" if a.filters else "") + f" · {state}: {detail}")
    href = page_url("export", project=project, asset=a.label)
    st.markdown(f'<a href="{href}" target="_self">Open in Export</a> — restores this asset\'s experiment, filter and options; '
                "pin again from there to change what it renders.", unsafe_allow_html=True)
    with st.form(key=f"asset_form_{a.label}"):
        c1, c2, c3 = st.columns([1, 1, 2])
        status = c1.selectbox("Status", ["planned", "draft", "final", "dropped"], index=["planned", "draft", "final", "dropped"].index(a.status.value))
        position = int(c2.number_input("Position", value=int(a.position), step=1, help="Manuscript order"))
        new_label = c3.text_input("Label", value=a.label)
        caption = st.text_area("Caption in the manuscript (blank = auto-generated)", value=a.caption, height=70)
        notes = st.text_area("Notes", value=a.notes, height=70, placeholder="decisions, what the reviewer asked, …")
        if st.form_submit_button("Save"):
            fields = {"status": status, "position": position, "caption": caption, "notes": notes}
            if new_label.strip() and new_label.strip() != a.label:
                fields["label"] = new_label.strip()
            update_asset(project, a.label, engine=engine, **fields)
            st.session_state["paper_asset"] = fields.get("label", a.label)
            st.rerun()
    with st.expander("Rendering options"):
        st.json({"kind": a.kind, "experiment": a.experiment, "filters": a.filters, "options": a.options})
    with st.expander("Delete asset"):
        st.warning("Forgets the pin only; files already exported are left alone.")
        if st.checkbox(f"Yes, delete `{a.label}`", key=f"del_{a.label}") and st.button("Delete", key=f"delbtn_{a.label}", type="primary"):
            delete_asset(project, a.label, engine=engine)
            st.session_state.pop("paper_asset", None)
            st.rerun()


def _export_section(project: str, engine) -> None:
    st.subheader("Export the paper")
    default_dir = str(Path(db_path()).expanduser().resolve().parent / "paper_assets") if db_path() != ":memory:" else "paper_assets"
    c1, c2 = st.columns([2, 1])
    with c1:
        out_dir = keyed(st.text_input, "Directory", "paper_out_dir", default_dir,
                        help="tables/, figures/, data/, preamble.tex and MANIFEST.json are written here; stable names, so the "
                             "manuscript can \\input them and a re-export replaces them in place.")
    with c2:
        statuses = st.multiselect("Statuses", list(EXPORT_STATUSES) + ["dropped"], default=list(EXPORT_STATUSES), key="paper_statuses")
    st.code(f"results-tracker export paper -p {project} -o {out_dir} --db {db_path()}", language="bash")
    b1, b2 = st.columns(2)
    if b1.button("Write to directory", type="primary", disabled=not out_dir or not statuses):
        with st.spinner("Rendering assets…"):
            rendered = render_paper(engine, project, source=db_path(), statuses=statuses, records_for=lambda e: load_records(project, e))
            paths = write_paper(rendered, out_dir, project=project, source=db_path())
            mark_exported(engine, project, rendered)
        st.session_state["paper_rendered"] = (rendered, f"{len(paths)} files written under `{out_dir}`", None)
        st.rerun()
    if b2.button("Render for download (zip)", disabled=not statuses):
        with st.spinner("Rendering assets…"):
            rendered = render_paper(engine, project, source=db_path(), statuses=statuses, records_for=lambda e: load_records(project, e))
            data = zip_paper(rendered, project=project, source=db_path())
            mark_exported(engine, project, rendered)
        st.session_state["paper_rendered"] = (rendered, f"{len(data) // 1024} KB zip", data)
        st.rerun()
    if "paper_rendered" in st.session_state:
        rendered, summary, data = st.session_state["paper_rendered"]
        failed = [r for r in rendered if r.error]
        (st.warning if failed else st.success)(f"{len(rendered) - len(failed)} assets rendered, {len(failed)} failed · {summary}")
        rows = [[r.label, KIND_TITLES.get(r.kind, r.kind), r.status, r.experiment, " ".join(r.filters) or "—", r.runs,
                 "<br>".join(f for f, _ in r.files) or "—", f"<b>{r.error}</b>" if r.error else r.note] for r in rendered]
        st.markdown(generic_html(["Label", "Kind", "Status", "Experiment", "Filter", "Runs", "Files", "Note"], rows,
                                 caption="What the last export produced (also in MANIFEST.json).", number=2, left_cols=8, raw_html_cols=[6, 7]),
                    unsafe_allow_html=True)
        if data:
            st.download_button("Download paper zip", data, file_name=f"{project}-paper.zip", mime="application/zip")
