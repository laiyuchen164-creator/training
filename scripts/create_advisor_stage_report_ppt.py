"""Generate the advisor stage-report PowerPoint deck.

The environment used for this project does not always include python-pptx.
This script writes a minimal Office Open XML .pptx directly with the Python
standard library so the deck can be regenerated without extra dependencies.
"""

from __future__ import annotations

import html
import os
import zipfile
from pathlib import Path


OUT = Path("presentations/advisor_stage_report_2026-04-20.pptx")
SLIDE_W = 13_333_333
SLIDE_H = 7_500_000

COLORS = {
    "navy": "203A43",
    "teal": "0F766E",
    "mint": "DDF7F1",
    "ink": "1F2937",
    "muted": "52606D",
    "line": "D7DEE8",
    "paper": "FFFFFF",
    "soft": "F5F7FA",
    "orange": "C2410C",
    "green": "15803D",
}


slides = [
    {
        "kind": "title",
        "title": "Source-Aware Belief Revision",
        "subtitle": "A stage report on incremental reasoning, CIPC training, and the preserve-vs-revise trade-off",
        "footer": "Advisor update | 2026-04-20",
    },
    {
        "kind": "bullets",
        "title": "Background",
        "bullets": [
            "Large language models often receive information incrementally rather than all at once.",
            "A model may form an early commitment, then receive new evidence that either preserves or changes it.",
            "The central capability is not only answering correctly, but maintaining and updating commitments reliably.",
            "Belief-R gives us a controlled setting for suppression-style, non-monotonic reasoning.",
        ],
        "takeaway": "We study whether models can decide when to keep an early belief and when to revise it.",
    },
    {
        "kind": "bullets",
        "title": "Motivation",
        "bullets": [
            "Real multi-turn reasoning needs stable state management across evidence updates.",
            "Models often fail in two opposite ways: over-preserve early commitments or over-revise on harmless updates.",
            "Prompting alone exposed the failure mode but did not solve the trade-off.",
            "A trainable commitment-control formulation lets us supervise both the update decision and the final answer.",
        ],
        "takeaway": "The goal is calibrated preserve-vs-revise control: revise when needed, preserve when justified.",
    },
    {
        "kind": "two_col",
        "title": "Related Work Positioning",
        "left_title": "Conceptual roots",
        "left": [
            "Belief revision and non-monotonic reasoning",
            "Suppression tasks and Belief-R",
            "Incremental and multi-turn reasoning",
        ],
        "right_title": "Methodological context",
        "right": [
            "Source-aware reasoning and provenance tracking",
            "Prompt-based revision policies",
            "Fine-tuning with control-oriented supervision",
        ],
        "takeaway": "Our project connects classic belief revision with trainable LLM commitment control.",
    },
    {
        "kind": "bullets",
        "title": "Main Contributions",
        "bullets": [
            "Reframed Belief-R as an incremental source-aware belief revision problem.",
            "Built a deterministic commitment-control dataset with preserve/replace and final-answer labels.",
            "Implemented CIPC: Commitment Integration and Propagation Control.",
            "Built NumPy and HF/LoRA training pipelines on a fixed Belief-R split.",
            "Isolated a reproducible preserve-vs-revise trade-off in the loss geometry.",
        ],
        "takeaway": "The project has moved from diagnostic prompting to trainable method design.",
    },
    {
        "kind": "diagram",
        "title": "Task Formulation",
        "nodes": [
            ("Early evidence", "forms commitment"),
            ("New evidence", "semantic role"),
            ("Control decision", "preserve / replace"),
            ("Final answer", "a / b / c"),
        ],
        "takeaway": "We evaluate both integration control and answer propagation.",
    },
    {
        "kind": "bullets",
        "title": "Experiment Design",
        "bullets": [
            "Dataset: Belief-R strong paired subset converted to commitment-control format.",
            "Split sizes: train 2050, dev 254, test 260.",
            "Conditions: full_info, incremental_no_overturn, incremental_overturn_reasoning.",
            "Models: prompt baselines, NumPy CIPC, HF/LoRA CIPC with Qwen2.5-0.5B-Instruct, direct API baselines.",
            "Metrics: answer accuracy, control accuracy, joint accuracy, early persistence, late evidence takeover.",
        ],
        "takeaway": "The design separates whether the model chose the right update policy from whether it propagated it.",
    },
    {
        "kind": "table",
        "title": "Key Completed Results",
        "headers": ["Method", "Overall", "Overturn", "No-overturn"],
        "rows": [
            ["Direct OpenAI gpt-5.4-mini", "0.2077", "0.0000", "1.0000"],
            ["Direct DeepSeek chat", "0.2192", "0.0097", "1.0000"],
            ["Frozen prompt baseline", "0.3615", "0.2718", "0.5926"],
            ["NumPy CIPC baseline", "0.7846", "0.7961", "0.9259"],
            ["HF local highrank v1", "0.8462", "0.8641", "0.7778"],
            ["Runpod highrank v1", "0.8538", "0.9029", "0.6667"],
        ],
        "takeaway": "CIPC strongly improves over prompt and direct API baselines, but the frontier still trades off stability and revision.",
    },
    {
        "kind": "table",
        "title": "Loss Redesign Findings",
        "headers": ["Run", "Overall", "Overturn", "No-overturn"],
        "rows": [
            ["conditional_consistency_v1", "0.8923", "0.9806", "0.5556"],
            ["preserve_margin_v1", "0.8077", "0.8350", "0.7037"],
            ["preserve_hybrid_v1", "0.8615", "0.9223", "0.6296"],
            ["preserve_hybrid_v2", "0.8615", "0.9515", "0.5185"],
        ],
        "takeaway": "Preserve-side structure matters, but the current hybrid objective is not the final repair.",
    },
    {
        "kind": "bullets",
        "title": "Expected Effect",
        "bullets": [
            "A successful model should revise on true overturn cases instead of preserving stale commitments.",
            "It should remain stable on no-overturn cases instead of treating every new premise as a correction.",
            "Control decisions should align with final answers, reducing propagation failures.",
            "The next target is a new Pareto point: high overturn accuracy with materially better no-overturn accuracy.",
        ],
        "takeaway": "Success means calibrated behavior, not simply more aggressive revision.",
    },
    {
        "kind": "bullets",
        "title": "Current Conclusion",
        "bullets": [
            "The task is not solved by direct prompting of stronger external models.",
            "The CIPC training formulation is effective and outperforms frozen prompt baselines.",
            "The remaining bottleneck is objective shape, not data access, infrastructure, checkpointing, or simple sampling.",
            "Preserve-side supervision needs a new form beyond plain CE, pure margin, or the current CE+margin hybrid.",
        ],
        "takeaway": "We now have a focused method-design problem with clear baselines and diagnostics.",
    },
    {
        "kind": "bullets",
        "title": "Next Steps",
        "bullets": [
            "Design a new preserve-side objective form rather than repeating scalar weight sweeps.",
            "Keep split preserve/replace aggregation to avoid replace examples dominating the loss.",
            "Compare the next run against highrank_v1, tradeoff_repair_v2, preserve_margin_v1, and preserve_hybrid variants.",
            "Target: overturn near 0.90 while lifting no-overturn clearly above the current 0.6667 aggressive anchor.",
        ],
        "takeaway": "The next contribution should come from loss geometry, especially preserve-side influence.",
    },
]


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def emu(v: float) -> int:
    return int(v)


def color_fill(color: str) -> str:
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def text_box(
    shape_id: int,
    name: str,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    size: int = 24,
    color: str = COLORS["ink"],
    bold: bool = False,
    fill: str | None = None,
    align: str = "l",
) -> str:
    fill_xml = color_fill(fill) if fill else "<a:noFill/>"
    bold_attr = ' b="1"' if bold else ""
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        {fill_xml}
        <a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="120000" tIns="80000" rIns="120000" bIns="80000"/>
        <a:lstStyle/>
        <a:p>
          <a:pPr algn="{align}"/>
          <a:r><a:rPr lang="en-US" sz="{size * 100}"{bold_attr}>{color_fill(color)}</a:rPr><a:t>{esc(text)}</a:t></a:r>
        </a:p>
      </p:txBody>
    </p:sp>
    """


def bullet_box(shape_id: int, x: int, y: int, w: int, h: int, bullets: list[str]) -> str:
    paras = []
    for item in bullets:
        paras.append(
            f"""
        <a:p>
          <a:pPr marL="280000" indent="-160000"/>
          <a:r><a:rPr lang="en-US" sz="2350">{color_fill(COLORS["ink"])}</a:rPr><a:t>• {esc(item)}</a:t></a:r>
        </a:p>"""
        )
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="Bullets"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/><a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="100000" tIns="60000" rIns="100000" bIns="60000"/>
        <a:lstStyle/>
        {''.join(paras)}
      </p:txBody>
    </p:sp>
    """


def rect(shape_id: int, x: int, y: int, w: int, h: int, fill: str, line: str | None = None) -> str:
    line_xml = f'<a:ln w="12000">{color_fill(line)}</a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="Shape"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        {color_fill(fill)}
        {line_xml}
      </p:spPr>
      <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
    </p:sp>
    """


def line(shape_id: int, x1: int, y1: int, x2: int, y2: int, color: str = COLORS["line"]) -> str:
    return f"""
    <p:cxnSp>
      <p:nvCxnSpPr><p:cNvPr id="{shape_id}" name="Connector"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{min(x1, x2)}" y="{min(y1, y2)}"/><a:ext cx="{abs(x2 - x1)}" cy="{abs(y2 - y1)}"/></a:xfrm>
        <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
        <a:ln w="26000" cap="rnd">{color_fill(color)}<a:tailEnd type="triangle"/></a:ln>
      </p:spPr>
    </p:cxnSp>
    """


def title_header(shape_start: int, title: str) -> tuple[str, int]:
    xml = rect(shape_start, 0, 0, SLIDE_W, 520000, COLORS["navy"])
    xml += text_box(shape_start + 1, "Title", 480000, 85000, 10_600_000, 330000, title, 28, COLORS["paper"], True)
    xml += rect(shape_start + 2, 0, 520000, SLIDE_W, 55000, COLORS["teal"])
    return xml, shape_start + 3


def takeaway(shape_id: int, text: str) -> str:
    return text_box(
        shape_id,
        "Takeaway",
        610000,
        6_520_000,
        12_100_000,
        500000,
        "Takeaway: " + text,
        17,
        COLORS["navy"],
        True,
        COLORS["mint"],
    )


def table_xml(shape_id: int, headers: list[str], rows: list[list[str]]) -> str:
    x, y, w = 630000, 1_380_000, 12_080_000
    row_h = 540000
    col_ws = [4_450_000, 2_390_000, 2_390_000, 2_850_000]
    xml = ""
    sid = shape_id
    for r_idx, row in enumerate([headers] + rows):
        fill = COLORS["navy"] if r_idx == 0 else (COLORS["soft"] if r_idx % 2 == 0 else COLORS["paper"])
        color = COLORS["paper"] if r_idx == 0 else COLORS["ink"]
        bold = r_idx == 0
        cx = x
        for c_idx, cell in enumerate(row):
            cw = col_ws[c_idx]
            xml += rect(sid, cx, y + r_idx * row_h, cw, row_h, fill, COLORS["line"])
            sid += 1
            align = "ctr" if c_idx > 0 else "l"
            xml += text_box(sid, "Cell", cx + 40000, y + r_idx * row_h + 45000, cw - 80000, row_h - 80000, cell, 15, color, bold, None, align)
            sid += 1
            cx += cw
    return xml


def slide_xml(slide: dict, idx: int) -> str:
    sp = rect(2, 0, 0, SLIDE_W, SLIDE_H, COLORS["paper"])
    sid = 3
    kind = slide["kind"]

    if kind == "title":
        sp += rect(sid, 0, 0, SLIDE_W, SLIDE_H, COLORS["navy"])
        sid += 1
        sp += rect(sid, 0, 5_980_000, SLIDE_W, 260000, COLORS["teal"])
        sid += 1
        sp += text_box(sid, "Deck Title", 720000, 1_780_000, 11_600_000, 980000, slide["title"], 44, COLORS["paper"], True)
        sid += 1
        sp += text_box(sid, "Subtitle", 760000, 2_900_000, 10_900_000, 720000, slide["subtitle"], 23, "DDF7F1")
        sid += 1
        sp += text_box(sid, "Footer", 760000, 6_320_000, 6_000_000, 320000, slide["footer"], 16, "D7DEE8")
    elif kind == "bullets":
        hdr, sid = title_header(sid, slide["title"])
        sp += hdr
        sp += bullet_box(sid, 720000, 1_200_000, 11_800_000, 4_880_000, slide["bullets"])
        sid += 1
        sp += takeaway(sid, slide["takeaway"])
    elif kind == "two_col":
        hdr, sid = title_header(sid, slide["title"])
        sp += hdr
        sp += rect(sid, 690000, 1_250_000, 5_700_000, 4_850_000, COLORS["soft"], COLORS["line"])
        sid += 1
        sp += rect(sid, 6_930_000, 1_250_000, 5_700_000, 4_850_000, COLORS["soft"], COLORS["line"])
        sid += 1
        sp += text_box(sid, "Left Title", 900000, 1_440_000, 5_100_000, 400000, slide["left_title"], 22, COLORS["teal"], True)
        sid += 1
        sp += bullet_box(sid, 880000, 1_980_000, 5_150_000, 3_600_000, slide["left"])
        sid += 1
        sp += text_box(sid, "Right Title", 7_140_000, 1_440_000, 5_100_000, 400000, slide["right_title"], 22, COLORS["teal"], True)
        sid += 1
        sp += bullet_box(sid, 7_120_000, 1_980_000, 5_150_000, 3_600_000, slide["right"])
        sid += 1
        sp += takeaway(sid, slide["takeaway"])
    elif kind == "diagram":
        hdr, sid = title_header(sid, slide["title"])
        sp += hdr
        xs = [760000, 3_900_000, 7_040_000, 10_180_000]
        y = 2_350_000
        for i, (title, desc) in enumerate(slide["nodes"]):
            sp += rect(sid, xs[i], y, 2_420_000, 1_080_000, COLORS["soft"], COLORS["teal"])
            sid += 1
            sp += text_box(sid, "Node Title", xs[i] + 100000, y + 145000, 2_200_000, 300000, title, 20, COLORS["navy"], True, None, "ctr")
            sid += 1
            sp += text_box(sid, "Node Desc", xs[i] + 140000, y + 560000, 2_120_000, 300000, desc, 15, COLORS["muted"], False, None, "ctr")
            sid += 1
            if i < 3:
                sp += line(sid, xs[i] + 2_420_000, y + 540000, xs[i + 1], y + 540000, COLORS["teal"])
                sid += 1
        sp += takeaway(sid, slide["takeaway"])
    elif kind == "table":
        hdr, sid = title_header(sid, slide["title"])
        sp += hdr
        sp += table_xml(sid, slide["headers"], slide["rows"])
        sp += takeaway(80, slide["takeaway"])

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {sp}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def content_types() -> str:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(slides) + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  {slide_overrides}
</Types>
"""


def root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def presentation() -> str:
    ids = "\n".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, len(slides) + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldIdLst>{ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""


def presentation_rels() -> str:
    rels = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, len(slides) + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {rels}
</Relationships>
"""


def empty_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""


def core_props() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Source-Aware Belief Revision Advisor Stage Report</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-04-20T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-04-20T00:00:00Z</dcterms:modified>
</cp:coreProperties>
"""


def app_props() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex OOXML Generator</Application>
  <PresentationFormat>Widescreen</PresentationFormat>
  <Slides>{len(slides)}</Slides>
</Properties>
"""


def write_deck() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types())
        z.writestr("_rels/.rels", root_rels())
        z.writestr("docProps/core.xml", core_props())
        z.writestr("docProps/app.xml", app_props())
        z.writestr("ppt/presentation.xml", presentation())
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels())
        for i, slide in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(slide, i))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", empty_rels())


if __name__ == "__main__":
    write_deck()
    print(os.fspath(OUT))
