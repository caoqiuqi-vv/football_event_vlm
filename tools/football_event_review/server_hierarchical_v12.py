#!/usr/bin/env python3
"""Multi-label review UI with conditional shot and set-piece details."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v11  # noqa: F401


_html_v11 = hierarchical.patch_hierarchical_html
_js_v11 = hierarchical.patch_hierarchical_js


def replace_required(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"v12 patch anchor missing: {old[:100]!r}")
    return text.replace(old, new, 1)


def patch_html_v12(text: str) -> str:
    text = _html_v11(text).replace(
        "hierarchical-v11-single-label-source-20260818",
        "hierarchical-v12-conditional-details-20260818",
    )
    return replace_required(
        text,
        '<div class="label-flow-heading"><span>人工确认主事件</span><small id="correctionHint">默认与模型预测一致</small></div>',
        '<div class="label-flow-heading detail-heading"><span>事件详情（按已选标签补充）</span><small id="correctionHint">射门结果与定位球类型会写入对应标签</small></div>',
    )


def patch_js_v12(text: str) -> str:
    text = _js_v11(text)
    text = replace_required(
        text,
        "  state.secondaryLabels = new Set(event.review.secondary_labels || []);",
        "  state.secondaryLabels = new Set(group.flatMap((item) => item.review.secondary_labels || []));",
    )
    text = replace_required(
        text,
        '  shotPanel.classList.toggle("hidden", state.selectedLabel !== "shot");',
        '  shotPanel.classList.toggle("hidden", !state.selectedSegmentLabels.has("shot"));',
    )
    text = replace_required(
        text,
        '  setPiecePanel.classList.toggle("hidden", state.selectedLabel !== "set_piece");',
        '  setPiecePanel.classList.toggle("hidden", !state.selectedSegmentLabels.has("set_piece"));',
    )
    text = replace_required(
        text,
        '''      renderSegmentLabelEditor();
    };
  });
  const selectedNames = labels.filter''',
        '''      renderSegmentLabelEditor();
      renderClassButtons();
    };
  });
  const selectedNames = labels.filter''',
    )
    old_validation = '''  const missingSetPieceType = group.some((item) => item.label === "set_piece" && selected.has("set_piece") && !(item.review.secondary_labels || []).some((label) => (state.bootstrap.set_piece_type_labels || []).includes(label)) && item.id !== event.id);
  if (missingSetPieceType) {
    toast("该片段包含定位球：请先点定位球标签单独选择具体类型");
    return;
  }
  if (event.label === "set_piece" && selected.has("set_piece") && !(state.bootstrap.set_piece_type_labels || []).some((label) => state.secondaryLabels.has(label))) {
    toast("请先选择定位球类型");
    return;
  }'''
    new_validation = '''  if (selected.has("set_piece") && !(state.bootstrap.set_piece_type_labels || []).some((label) => state.secondaryLabels.has(label))) {
    toast("已保留定位球，请先选择任意球、点球、角球等具体类型");
    $("#setPieceTypePanel").scrollIntoView({ block: "nearest" });
    return;
  }'''
    text = replace_required(text, old_validation, new_validation)
    text = replace_required(
        text,
        '      const details = isCurrent ? [...state.secondaryLabels] : (item.review.secondary_labels || []);',
        '''      const shotDetails = new Set(state.bootstrap.shot_detail_labels || []);
      const setPieceDetails = new Set(state.bootstrap.set_piece_type_labels || []);
      const details = [...state.secondaryLabels].filter((label) =>
        item.label === "shot" ? shotDetails.has(label) :
        item.label === "set_piece" ? setPieceDetails.has(label) : false
      );''',
    )
    return text


hierarchical.patch_hierarchical_html = patch_html_v12
hierarchical.patch_hierarchical_js = patch_js_v12
hierarchical.HIERARCHICAL_CSS += r'''
/* v12: one primary-label source, with visible conditional detail editors. */
.label-flow-heading.detail-heading { display: flex !important; margin: 1px 0 6px; }
.label-flow-heading.detail-heading span { color: #dce6ef; }
.label-flow-heading.detail-heading #correctionHint { display: inline !important; color: #8190a0; }
.primary-label-buttons, #saveModify { display: none !important; }
.conditional-label-panel:not(.hidden) { display: block; }
'''


if __name__ == "__main__":
    hierarchical.main()
