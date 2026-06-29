/**
 * Generate 14-slide mid-project progress presentation for SSL WikiChurches.
 *
 * Uses PptxGenJS to create the deck from scratch.
 * Images are pre-generated in outputs/slides/.
 *
 * Run: node experiments/scripts/create_presentation.js
 */

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
const ROOT = path.resolve(__dirname, "../..");
const SLIDES_DIR = path.join(ROOT, "outputs", "slides");
const FIGURES_DIR = path.join(ROOT, "outputs", "figures");
const OUT_FILE = path.join(SLIDES_DIR, "presentation.pptx");

function imgPath(name) {
  const p = path.join(SLIDES_DIR, name);
  if (!fs.existsSync(p)) {
    console.warn(`WARNING: Image not found: ${p}`);
  }
  return p;
}

function figPath(name) {
  const p = path.join(FIGURES_DIR, name);
  if (!fs.existsSync(p)) {
    console.warn(`WARNING: Figure not found: ${p}`);
  }
  return p;
}

// ---------------------------------------------------------------------------
// Design system
// ---------------------------------------------------------------------------
const C = {
  CHARCOAL: "2D3436",
  STEEL: "4E79A7",
  TEAL: "93B7BE",
  TERRA: "D4764E",
  LIGHT_BG: "F8F9FA",
  WHITE: "FFFFFF",
  BODY: "333333",
  MUTED: "64748B",
  SUCCESS: "3A7D44",
  FAIL: "C04E4E",
  WARM_GRAY: "8A817C",
  LIGHT_GRAY: "E8E8E8",
  ACCENT_BAR: "4E79A7",
};

const FONT = {
  TITLE: "Avenir Next",
  BODY: "Avenir Next",
  MONO: "Menlo",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function addAccentBar(slide) {
  slide.addShape("rect", {
    x: 0, y: 5.525, w: 10, h: 0.1,
    fill: { color: C.ACCENT_BAR },
  });
}

function addSlideNumber(slide, num) {
  slide.addText(String(num), {
    x: 9.2, y: 5.2, w: 0.6, h: 0.3,
    fontSize: 10, color: C.MUTED, fontFace: FONT.BODY,
    align: "right",
  });
}

function addSectionTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontSize: 28, fontFace: FONT.TITLE, color: C.STEEL, bold: true,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.5, y: 0.85, w: 9, h: 0.35,
      fontSize: 14, fontFace: FONT.BODY, color: C.MUTED,
    });
  }
}

// Factory for text options (avoid mutation pitfall)
function bodyOpts(overrides) {
  return Object.assign({
    fontSize: 14, fontFace: FONT.BODY, color: C.BODY,
    valign: "top", paraSpaceAfter: 6,
  }, overrides || {});
}

// ---------------------------------------------------------------------------
// SLIDE 1: Title
// ---------------------------------------------------------------------------
function slide1_title(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.CHARCOAL };

  // Hero image on right half
  slide.addImage({
    path: imgPath("slide01_title_hero.png"),
    x: 5.0, y: 0, w: 5.0, h: 5.625,
    sizing: { type: "cover", w: 5.0, h: 5.625 },
  });

  // Semi-transparent overlay on right for text readability
  slide.addShape("rect", {
    x: 5.0, y: 0, w: 5.0, h: 5.625,
    fill: { color: C.CHARCOAL, transparency: 40 },
  });

  // Title text
  slide.addText("Do Self-Supervised Vision\nModels Learn What\nExperts See?", {
    x: 0.6, y: 0.8, w: 4.6, h: 2.0,
    fontSize: 30, fontFace: FONT.TITLE, color: C.WHITE, bold: true,
    lineSpacingMultiple: 1.15,
  });

  // Subtitle
  slide.addText("Attention Alignment with Human-Annotated\nArchitectural Features", {
    x: 0.6, y: 2.9, w: 4.6, h: 0.8,
    fontSize: 16, fontFace: FONT.BODY, color: C.TEAL,
  });

  // Course info
  slide.addText("ISY5004 \u2014 Intelligent Sensing Systems Practice Module\nMid-Project Progress Update \u2014 March 2026", {
    x: 0.6, y: 4.2, w: 4.6, h: 0.7,
    fontSize: 12, fontFace: FONT.BODY, color: C.WARM_GRAY,
  });

  // Accent bar
  slide.addShape("rect", {
    x: 0, y: 5.4, w: 10, h: 0.225,
    fill: { color: C.STEEL },
  });
}

// ---------------------------------------------------------------------------
// SLIDE 2: Motivation & Research Gap
// ---------------------------------------------------------------------------
function slide2_motivation(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Why This Matters", "SSL models achieve strong benchmarks \u2014 but what are they looking at?");
  addAccentBar(slide);
  addSlideNumber(slide, 2);

  // Left: bullet points
  slide.addText([
    { text: "The Problem", options: { fontSize: 16, bold: true, color: C.STEEL, breakLine: true } },
    { text: "SSL models can classify \u201cGothic cathedral\u201d by attending to ", options: { fontSize: 13, breakLine: false } },
    { text: "overcast skies", options: { fontSize: 13, bold: true, color: C.FAIL, breakLine: false } },
    { text: " instead of ", options: { fontSize: 13, breakLine: false } },
    { text: "pointed arches", options: { fontSize: 13, bold: true, color: C.SUCCESS, breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "The Gap", options: { fontSize: 16, bold: true, color: C.STEEL, breakLine: true } },
    { text: "\u2022 No cross-model benchmark for expert alignment", options: { fontSize: 13, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 No study on how fine-tuning shifts attention", options: { fontSize: 13, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 No strategy comparison (LP vs LoRA vs Full)", options: { fontSize: 13, breakLine: true } },
  ], {
    x: 0.5, y: 1.3, w: 4.2, h: 3.8,
    valign: "top", margin: 0,
  });

  // Right: good vs bad attention image
  slide.addImage({
    path: imgPath("slide02_good_vs_bad.png"),
    x: 4.9, y: 1.3, w: 4.8, h: 2.8,
    sizing: { type: "contain", w: 4.8, h: 2.8 },
  });

  // Caption
  slide.addText("Same image: DINOv3 focuses on architectural features (left),\nMAE attention is diffuse (right)", {
    x: 4.9, y: 4.2, w: 4.8, h: 0.6,
    fontSize: 10, fontFace: FONT.BODY, color: C.MUTED, align: "center",
  });
}

// ---------------------------------------------------------------------------
// SLIDE 3: Research Questions
// ---------------------------------------------------------------------------
function slide3_rqs(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Three Research Questions");
  addAccentBar(slide);
  addSlideNumber(slide, 3);

  const cards = [
    { q: "Q1", title: "Frozen Model Alignment", desc: "Do SSL models attend to expert-identified diagnostic features?", method: "IoU between attention maps and 631 expert bounding boxes across 7 models \u00d7 12 layers", status: "Complete", statusColor: C.SUCCESS },
    { q: "Q2", title: "Fine-Tuning Effects", desc: "Does fine-tuning shift attention toward expert features? Does strategy matter?", method: "\u0394 IoU (fine-tuned \u2212 frozen) with paired Wilcoxon tests, Holm correction, Cohen\u2019s d", status: "Complete", statusColor: C.SUCCESS },
    { q: "Q3", title: "Per-Head Specialization", desc: "Do individual attention heads specialize for different architectural features?", method: "Per-head IoU \u00d7 feature-type matrix, rank-based head analysis", status: "Planned", statusColor: C.WARM_GRAY },
  ];

  cards.forEach((c, i) => {
    const y = 1.2 + i * 1.35;

    // Card background
    slide.addShape("roundRect", {
      x: 0.5, y: y, w: 9.0, h: 1.2,
      fill: { color: C.WHITE },
      rectRadius: 0.08,
      line: { color: C.LIGHT_GRAY, width: 0.5 },
    });

    // Status indicator bar (left edge)
    slide.addShape("rect", {
      x: 0.5, y: y, w: 0.08, h: 1.2,
      fill: { color: c.statusColor },
    });

    // Q label
    slide.addText(c.q, {
      x: 0.75, y: y + 0.15, w: 0.6, h: 0.5,
      fontSize: 22, fontFace: FONT.TITLE, color: C.STEEL, bold: true, margin: 0,
    });

    // Title
    slide.addText(c.title, {
      x: 1.4, y: y + 0.1, w: 5.0, h: 0.35,
      fontSize: 16, fontFace: FONT.BODY, color: C.BODY, bold: true, margin: 0,
    });

    // Description
    slide.addText(c.desc, {
      x: 1.4, y: y + 0.42, w: 5.5, h: 0.35,
      fontSize: 12, fontFace: FONT.BODY, color: C.BODY, margin: 0,
    });

    // Method
    slide.addText(c.method, {
      x: 1.4, y: y + 0.78, w: 5.5, h: 0.3,
      fontSize: 10, fontFace: FONT.BODY, color: C.MUTED, margin: 0,
    });

    // Status badge
    slide.addShape("roundRect", {
      x: 8.2, y: y + 0.35, w: 1.1, h: 0.4,
      fill: { color: c.statusColor },
      rectRadius: 0.05,
    });
    slide.addText(c.status, {
      x: 8.2, y: y + 0.35, w: 1.1, h: 0.4,
      fontSize: 11, fontFace: FONT.BODY, color: C.WHITE, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
  });
}

// ---------------------------------------------------------------------------
// SLIDE 4: Professor Feedback (NEW)
// ---------------------------------------------------------------------------
function slide4_feedback(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Professor\u2019s Feedback", "Four suggestions that shaped our methodology");
  addAccentBar(slide);
  addSlideNumber(slide, 4);

  const items = [
    {
      num: "1", title: "Continuous Metrics",
      quote: "Apply Gaussian filtering to bounding boxes and compare with MSE, like crowd density estimation",
      status: "Implemented", statusColor: C.SUCCESS,
      detail: "MSE, KL Divergence, EMD added alongside IoU",
    },
    {
      num: "2", title: "Fine-Tuning Impact",
      quote: "Study whether SFT preserves, enhances, or destroys attention consistency",
      status: "Implemented", statusColor: C.SUCCESS,
      detail: "Preserve / Enhance / Destroy taxonomy applied to 6 models \u00d7 3 strategies",
    },
    {
      num: "3", title: "Cross-Layer Aggregation",
      quote: "Aggregate attention from different layers (e.g. max pooling) to construct a new map",
      status: "Planned", statusColor: C.WARM_GRAY,
      detail: "Max pooling, depth-weighted mean, ALTI",
    },
    {
      num: "4", title: "Unimodal vs Multimodal Leaderboard",
      quote: "Compare unimodal models (DINO) and VLM vision encoders (CLIP, SigLIP)",
      status: "Partial", statusColor: C.TEAL,
      detail: "Paradigm grouping applied in analysis; formal dashboard split planned",
    },
  ];

  items.forEach((item, i) => {
    const y = 1.15 + i * 1.05;

    // Card background
    slide.addShape("roundRect", {
      x: 0.5, y: y, w: 9.0, h: 0.9,
      fill: { color: C.WHITE },
      rectRadius: 0.06,
      line: { color: C.LIGHT_GRAY, width: 0.5 },
    });

    // Left accent bar
    slide.addShape("rect", {
      x: 0.5, y: y, w: 0.06, h: 0.9,
      fill: { color: item.statusColor },
    });

    // Number
    slide.addText(item.num, {
      x: 0.7, y: y + 0.1, w: 0.4, h: 0.35,
      fontSize: 20, fontFace: FONT.TITLE, color: C.STEEL, bold: true, margin: 0,
    });

    // Title
    slide.addText(item.title, {
      x: 1.15, y: y + 0.08, w: 3.0, h: 0.3,
      fontSize: 14, fontFace: FONT.BODY, color: C.BODY, bold: true, margin: 0,
    });

    // Quote (professor's words)
    slide.addText("\u201C" + item.quote + "\u201D", {
      x: 1.15, y: y + 0.38, w: 5.8, h: 0.25,
      fontSize: 10, fontFace: FONT.BODY, color: C.MUTED, italic: true, margin: 0,
    });

    // Detail
    slide.addText(item.detail, {
      x: 1.15, y: y + 0.62, w: 5.8, h: 0.2,
      fontSize: 10, fontFace: FONT.BODY, color: C.BODY, margin: 0,
    });

    // Status badge
    slide.addShape("roundRect", {
      x: 8.2, y: y + 0.25, w: 1.1, h: 0.35,
      fill: { color: item.statusColor },
      rectRadius: 0.05,
    });
    slide.addText(item.status, {
      x: 8.2, y: y + 0.25, w: 1.1, h: 0.35,
      fontSize: 10, fontFace: FONT.BODY, color: C.WHITE, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
  });
}

// ---------------------------------------------------------------------------
// SLIDE 5: Dataset (was 4)
// ---------------------------------------------------------------------------
function slide5_dataset(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "WikiChurches Dataset", "Barz & Denzler, NeurIPS 2021");
  addAccentBar(slide);
  addSlideNumber(slide, 5);

  // Left: stats
  const statsRows = [
    [{ text: "Total images", options: { bold: true, color: C.BODY, fill: { color: C.LIGHT_GRAY } } },
     { text: "9,485", options: { align: "right", fill: { color: C.LIGHT_GRAY } } }],
    [{ text: "Annotated eval set", options: { bold: true, color: C.BODY } },
     { text: "139 images, 631 bboxes", options: { align: "right" } }],
    [{ text: "Styles", options: { bold: true, color: C.BODY, fill: { color: C.LIGHT_GRAY } } },
     { text: "Romanesque, Gothic, Renaissance, Baroque", options: { align: "right", fill: { color: C.LIGHT_GRAY } } }],
    [{ text: "Feature types", options: { bold: true, color: C.BODY } },
     { text: "106 categories", options: { align: "right" } }],
    [{ text: "Training (Q2)", options: { bold: true, color: C.BODY, fill: { color: C.LIGHT_GRAY } } },
     { text: "~4,588 labelled images", options: { align: "right", fill: { color: C.LIGHT_GRAY } } }],
    [{ text: "Eval holdout", options: { bold: true, color: C.BODY } },
     { text: "139 images (zero leakage)", options: { align: "right", color: C.SUCCESS, bold: true } }],
  ];

  slide.addTable(statsRows, {
    x: 0.5, y: 1.3, w: 4.2, h: 2.5,
    fontSize: 12, fontFace: FONT.BODY, color: C.BODY,
    border: { type: "none" },
    colW: [2.0, 2.2],
    margin: [4, 6, 4, 6],
  });

  // Style distribution
  slide.addText("Style Distribution", {
    x: 0.5, y: 3.9, w: 4.2, h: 0.3,
    fontSize: 12, fontFace: FONT.BODY, color: C.MUTED, bold: true,
  });

  const styles = [
    { name: "Romanesque", count: 54, color: C.STEEL },
    { name: "Gothic", count: 49, color: C.TEAL },
    { name: "Renaissance", count: 22, color: C.TERRA },
    { name: "Baroque", count: 17, color: C.WARM_GRAY },
  ];
  const maxCount = 54;
  styles.forEach((s, i) => {
    const barY = 4.25 + i * 0.28;
    const barW = (s.count / maxCount) * 2.5;
    slide.addText(s.name, {
      x: 0.5, y: barY, w: 1.2, h: 0.22,
      fontSize: 10, fontFace: FONT.BODY, color: C.BODY, align: "right", margin: 0,
    });
    slide.addShape("rect", {
      x: 1.8, y: barY + 0.02, w: barW, h: 0.18,
      fill: { color: s.color },
    });
    slide.addText(String(s.count), {
      x: 1.85 + barW, y: barY, w: 0.5, h: 0.22,
      fontSize: 10, fontFace: FONT.BODY, color: C.MUTED, margin: 0,
    });
  });

  // Right: 2x2 grid image
  slide.addImage({
    path: imgPath("slide04_style_grid.png"),
    x: 5.0, y: 1.2, w: 4.6, h: 4.0,
    sizing: { type: "contain", w: 4.6, h: 4.0 },
  });
}

// ---------------------------------------------------------------------------
// SLIDE 6: Models (was 5)
// ---------------------------------------------------------------------------
function slide6_models(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "7 Models, 4 SSL Paradigms + 1 Supervised Baseline", "All ViT-B architecture (12 layers, 768 dim, 12 heads, ~86M params) except ResNet-50");
  addAccentBar(slide);
  addSlideNumber(slide, 6);

  const headerOpts = { bold: true, color: C.WHITE, fill: { color: C.STEEL }, fontSize: 11, align: "center" };
  const rows = [
    [
      { text: "Model", options: Object.assign({}, headerOpts) },
      { text: "Paradigm", options: Object.assign({}, headerOpts) },
      { text: "Patch", options: Object.assign({}, headerOpts) },
      { text: "Method", options: Object.assign({}, headerOpts) },
      { text: "Key Feature", options: Object.assign({}, headerOpts) },
    ],
    makeModelRow("DINOv2", "Self-distillation", "14\u00d714", "CLS, Rollout", "4 register tokens", C.STEEL, false),
    makeModelRow("DINOv3", "Self-distillation + Gram", "16\u00d716", "CLS, Rollout", "RoPE encoding", C.STEEL, true),
    makeModelRow("MAE", "Masked Autoencoding", "16\u00d716", "CLS, Rollout", "Pixel reconstruction", C.SUCCESS, false),
    makeModelRow("CLIP", "Contrastive (softmax)", "16\u00d716", "CLS, Rollout", "Language-image align", C.TERRA, true),
    makeModelRow("SigLIP", "Contrastive (sigmoid)", "16\u00d716", "Mean", "No CLS token", C.TERRA, false),
    makeModelRow("SigLIP 2", "Contrastive (sigmoid)", "16\u00d716", "Mean", "Dense features", C.TERRA, true),
    makeModelRow("ResNet-50", "Supervised (ImageNet)", "\u2014", "Grad-CAM", "CNN baseline", C.WARM_GRAY, false),
  ];

  slide.addTable(rows, {
    x: 0.5, y: 1.3, w: 9.0, h: 3.8,
    fontSize: 12, fontFace: FONT.BODY, color: C.BODY,
    border: { type: "solid", color: C.LIGHT_GRAY, pt: 0.5 },
    colW: [1.2, 2.2, 0.8, 1.3, 1.8],
    margin: [4, 6, 4, 6],
    autoPage: false,
  });
}

function makeModelRow(name, paradigm, patch, method, feature, accentColor, altBg) {
  const bg = altBg ? { fill: { color: "F0F4F8" } } : {};
  return [
    { text: name, options: Object.assign({ bold: true, color: accentColor }, bg) },
    { text: paradigm, options: Object.assign({}, bg) },
    { text: patch, options: Object.assign({ align: "center" }, bg) },
    { text: method, options: Object.assign({}, bg) },
    { text: feature, options: Object.assign({ fontSize: 11, color: C.MUTED }, bg) },
  ];
}

// ---------------------------------------------------------------------------
// SLIDE 7: Methodology (was 6) — framed around feedback point 1
// ---------------------------------------------------------------------------
function slide7_methodology(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Measuring Attention-Expert Alignment", "Addressing Feedback #1: continuous metrics beyond IoU");
  addAccentBar(slide);
  addSlideNumber(slide, 7);

  // App screenshot showing the full pipeline in context
  slide.addImage({
    path: imgPath("slide06_pipeline.png"),
    x: 0.3, y: 1.2, w: 9.4, h: 2.2,
    sizing: { type: "contain", w: 9.4, h: 2.2 },
  });

  // Caption explaining what the screenshot shows
  slide.addText("Image Detail view: attention overlay (left), expert annotations (bboxes), layer-wise metrics (right)", {
    x: 0.3, y: 3.35, w: 9.4, h: 0.2,
    fontSize: 9, fontFace: FONT.BODY, color: C.MUTED, align: "center", italic: true,
  });

  // Metrics table below
  slide.addText("5 Complementary Metrics", {
    x: 0.5, y: 3.5, w: 3, h: 0.3,
    fontSize: 14, fontFace: FONT.BODY, color: C.STEEL, bold: true,
  });

  const mHeaderOpts = { bold: true, color: C.WHITE, fill: { color: C.STEEL }, fontSize: 10, align: "center" };
  const mRows = [
    [
      { text: "Metric", options: Object.assign({}, mHeaderOpts) },
      { text: "Measures", options: Object.assign({}, mHeaderOpts) },
      { text: "Threshold?", options: Object.assign({}, mHeaderOpts) },
    ],
    [{ text: "IoU", options: { bold: true } }, { text: "Spatial overlap of top-k attention with expert regions" }, { text: "Yes", options: { align: "center" } }],
    [{ text: "Coverage", options: { bold: true } }, { text: "Fraction of attention energy inside expert regions" }, { text: "No", options: { align: "center" } }],
    [{ text: "Gaussian MSE", options: { bold: true } }, { text: "Distance from attention to Gaussian-blurred GT" }, { text: "No", options: { align: "center" } }],
    [{ text: "KL Divergence", options: { bold: true } }, { text: "Distribution divergence vs ground truth" }, { text: "No", options: { align: "center" } }],
    [{ text: "EMD", options: { bold: true } }, { text: "Optimal transport cost between distributions" }, { text: "No", options: { align: "center" } }],
  ];

  slide.addTable(mRows, {
    x: 0.5, y: 3.85, w: 5.5, h: 1.6,
    fontSize: 10, fontFace: FONT.BODY, color: C.BODY,
    border: { type: "solid", color: C.LIGHT_GRAY, pt: 0.5 },
    colW: [1.2, 3.2, 1.1],
    margin: [3, 4, 3, 4],
    autoPage: false,
  });

  // Statistical rigor box
  slide.addShape("roundRect", {
    x: 6.3, y: 3.85, w: 3.3, h: 1.6,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    line: { color: C.LIGHT_GRAY, width: 0.5 },
  });
  slide.addText([
    { text: "Statistical Rigor", options: { fontSize: 13, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "\u2022 Paired Wilcoxon signed-rank tests", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "   (139 image pairs per comparison)", options: { fontSize: 10, color: C.MUTED, breakLine: true, paraSpaceAfter: 5 } },
    { text: "\u2022 Holm-Bonferroni correction", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 5 } },
    { text: "\u2022 Cohen\u2019s d effect sizes", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "   with 95% bootstrap CIs", options: { fontSize: 10, color: C.MUTED, breakLine: true } },
  ], {
    x: 6.5, y: 3.95, w: 3.0, h: 1.4,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 8: Q1 Results (was 7) — with paradigm grouping per feedback point 4
// ---------------------------------------------------------------------------
function slide8_q1(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Q1: Frozen Model Leaderboard", "Addressing Feedback #4: unimodal vs multimodal comparison");
  addAccentBar(slide);
  addSlideNumber(slide, 8);

  // Leaderboard bar chart — 3 series for paradigm grouping
  slide.addChart(pres.charts.BAR, [
    {
      name: "Unimodal SSL",
      labels: ["MAE", "SigLIP2", "SigLIP", "CLIP", "DINOv2", "ResNet50", "DINOv3"],
      values: [0.037, 0, 0, 0, 0.082, 0, 0.133],
    },
    {
      name: "Multimodal VLM",
      labels: ["MAE", "SigLIP2", "SigLIP", "CLIP", "DINOv2", "ResNet50", "DINOv3"],
      values: [0, 0.047, 0.047, 0.049, 0, 0, 0],
    },
    {
      name: "Supervised",
      labels: ["MAE", "SigLIP2", "SigLIP", "CLIP", "DINOv2", "ResNet50", "DINOv3"],
      values: [0, 0, 0, 0, 0, 0.090, 0],
    },
  ], {
    x: 0.4, y: 1.3, w: 4.5, h: 3.5,
    barDir: "bar",
    barGrouping: "stacked",
    showTitle: true, title: "Frozen IoU @ 90th Percentile (Best Layer)",
    titleFontSize: 11, titleColor: C.BODY,
    chartColors: [C.STEEL, C.TERRA, C.WARM_GRAY],
    catAxisLabelColor: C.BODY, catAxisLabelFontSize: 10,
    valAxisLabelColor: C.MUTED, valAxisLabelFontSize: 9,
    valGridLine: { color: C.LIGHT_GRAY, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9,
    dataLabelColor: C.BODY,
    valAxisMaxVal: 0.16,
    showLegend: true, legendPos: "b", legendFontSize: 9,
  });

  // Key findings — reframed with paradigm comparison
  slide.addShape("roundRect", {
    x: 5.2, y: 1.3, w: 4.4, h: 3.5,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    line: { color: C.LIGHT_GRAY, width: 0.5 },
  });
  slide.addText([
    { text: "Paradigm Ranking", options: { fontSize: 15, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Unimodal self-distillation leads", options: { fontSize: 13, bold: true, color: C.BODY, breakLine: true } },
    { text: "DINOv3 (0.133) achieves 1.6\u00d7 the IoU of DINOv2 (0.082). Self-distillation learns part-level segmentation.", options: { fontSize: 11, color: C.MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Supervised > Multimodal", options: { fontSize: 13, bold: true, color: C.BODY, breakLine: true } },
    { text: "ResNet-50 (0.090) outperforms all VLMs. Language alignment doesn\u2019t help localisation.", options: { fontSize: 11, color: C.MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "VLMs cluster together", options: { fontSize: 13, bold: true, color: C.BODY, breakLine: true } },
    { text: "CLIP/SigLIP/SigLIP2 all \u2248 0.047\u20130.049. Contrastive objective produces similar attention patterns.", options: { fontSize: 11, color: C.MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Reconstruction is worst", options: { fontSize: 13, bold: true, color: C.BODY, breakLine: true } },
    { text: "MAE (0.037) \u2014 pixel reconstruction \u2260 object localisation.", options: { fontSize: 11, color: C.MUTED, breakLine: true } },
  ], {
    x: 5.4, y: 1.4, w: 4.0, h: 3.3,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 9: Q2 Setup (was 8) — with Preserve/Enhance/Destroy taxonomy
// ---------------------------------------------------------------------------
function slide9_q2setup(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Q2: Fine-Tuning Experiment Design", "Addressing Feedback #2: does SFT preserve, enhance, or destroy consistency?");
  addAccentBar(slide);
  addSlideNumber(slide, 9);

  // Three strategy cards
  const strategies = [
    { name: "Linear Probe", params: "~3K", desc: "Classifier head only\nBackbone frozen", color: C.TEAL, purpose: "Baseline \u2014 attention unchanged" },
    { name: "LoRA", params: "~300K", desc: "Low-rank adapters on\nattention layers + head", color: C.STEEL, purpose: "Parameter-efficient; preserves pre-training" },
    { name: "Full Fine-tune", params: "~86M", desc: "Entire backbone +\nclassifier head", color: C.TERRA, purpose: "Maximum adaptation capacity" },
  ];

  strategies.forEach((s, i) => {
    const x = 0.5 + i * 3.1;

    slide.addShape("roundRect", {
      x: x, y: 1.3, w: 2.9, h: 2.3,
      fill: { color: C.WHITE },
      rectRadius: 0.06,
      line: { color: C.LIGHT_GRAY, width: 0.5 },
    });

    // Top accent
    slide.addShape("rect", {
      x: x, y: 1.3, w: 2.9, h: 0.06,
      fill: { color: s.color },
    });

    slide.addText(s.name, {
      x: x + 0.15, y: 1.45, w: 2.6, h: 0.35,
      fontSize: 16, fontFace: FONT.BODY, color: s.color, bold: true, margin: 0,
    });

    slide.addText(s.params + " params", {
      x: x + 0.15, y: 1.8, w: 2.6, h: 0.25,
      fontSize: 20, fontFace: FONT.BODY, color: C.BODY, bold: true, margin: 0,
    });

    slide.addText(s.desc, {
      x: x + 0.15, y: 2.15, w: 2.6, h: 0.55,
      fontSize: 11, fontFace: FONT.BODY, color: C.MUTED, margin: 0,
    });

    slide.addText(s.purpose, {
      x: x + 0.15, y: 2.8, w: 2.6, h: 0.35,
      fontSize: 10, fontFace: FONT.BODY, color: C.BODY, italic: true, margin: 0,
    });
  });

  // Training config box
  slide.addShape("roundRect", {
    x: 0.5, y: 3.85, w: 9.0, h: 1.3,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
  });

  slide.addText([
    { text: "Training Configuration:  ", options: { fontSize: 12, bold: true, color: C.STEEL } },
    { text: "4-class style classification (Romanesque, Gothic, Renaissance, Baroque)  \u2022  ", options: { fontSize: 11 } },
    { text: "3 epochs, batch 16  \u2022  ", options: { fontSize: 11 } },
    { text: "Cosine LR + warmup  \u2022  ", options: { fontSize: 11 } },
    { text: "Class-weighted loss  \u2022  ", options: { fontSize: 11 } },
    { text: "139 annotated images strictly excluded from training", options: { fontSize: 11, bold: true, color: C.SUCCESS } },
  ], {
    x: 0.7, y: 3.95, w: 8.6, h: 0.5,
    valign: "top", margin: 0,
  });

  slide.addText([
    { text: "Measurement:  ", options: { fontSize: 12, bold: true, color: C.STEEL } },
    { text: "\u0394 IoU = IoU(fine-tuned) \u2212 IoU(frozen), per image, paired Wilcoxon tests with Holm correction", options: { fontSize: 11 } },
  ], {
    x: 0.7, y: 4.45, w: 8.6, h: 0.3,
    valign: "top", margin: 0,
  });

  // Taxonomy definition (professor's framework)
  slide.addText([
    { text: "Taxonomy:  ", options: { fontSize: 12, bold: true, color: C.STEEL } },
    { text: "Enhance", options: { fontSize: 11, bold: true, color: C.SUCCESS } },
    { text: " (\u0394 > 0, significant)  |  ", options: { fontSize: 11 } },
    { text: "Preserve", options: { fontSize: 11, bold: true, color: C.WARM_GRAY } },
    { text: " (\u0394 \u2248 0, not significant)  |  ", options: { fontSize: 11 } },
    { text: "Destroy", options: { fontSize: 11, bold: true, color: C.FAIL } },
    { text: " (\u0394 < 0, significant)", options: { fontSize: 11 } },
  ], {
    x: 0.7, y: 4.75, w: 8.6, h: 0.3,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 10: Q2 Results (was 9) — with Preserve/Enhance/Destroy taxonomy
// ---------------------------------------------------------------------------
function slide10_q2results(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Q2: Preserve, Enhance, or Destroy?", "Applying the professor\u2019s taxonomy to fine-tuning results");
  addAccentBar(slide);
  addSlideNumber(slide, 10);

  // Embed the diverging bars figure
  slide.addImage({
    path: figPath("03_all_metrics_diverging_bars.png"),
    x: 0.2, y: 1.2, w: 5.3, h: 3.8,
    sizing: { type: "contain", w: 5.3, h: 3.8 },
  });

  // Taxonomy results on right
  slide.addShape("roundRect", {
    x: 5.7, y: 1.2, w: 4.0, h: 4.0,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    line: { color: C.LIGHT_GRAY, width: 0.5 },
  });

  slide.addText([
    { text: "Enhance", options: { fontSize: 16, bold: true, color: C.SUCCESS, breakLine: true } },
    { text: "Contrastive models (CLIP, SigLIP)", options: { fontSize: 11, color: C.BODY, breakLine: true } },
    { text: "CLIP LoRA: \u0394 +0.063 (d=1.33)", options: { fontSize: 10, color: C.SUCCESS, breakLine: true } },
    { text: "SigLIP/2 Full: \u0394 +0.036 (d=0.78\u20130.94)", options: { fontSize: 10, color: C.SUCCESS, breakLine: true } },
    { text: "All 6 comparisons statistically significant", options: { fontSize: 10, color: C.MUTED, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Preserve", options: { fontSize: 16, bold: true, color: C.WARM_GRAY, breakLine: true } },
    { text: "Self-distillation & reconstruction", options: { fontSize: 11, color: C.BODY, breakLine: true } },
    { text: "DINOv3: \u0394 +0.003\u20130.009 (not significant)", options: { fontSize: 10, color: C.MUTED, breakLine: true } },
    { text: "MAE: \u0394 \u2248 0 across all strategies", options: { fontSize: 10, color: C.MUTED, breakLine: true } },
    { text: "Already aligned (DINO) or unreachable (MAE)", options: { fontSize: 10, color: C.MUTED, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Destroy", options: { fontSize: 16, bold: true, color: C.FAIL, breakLine: true } },
    { text: "No model conclusively destroys", options: { fontSize: 11, color: C.BODY, breakLine: true } },
    { text: "DINOv2 Full: \u0394 \u22120.003 (not significant)", options: { fontSize: 10, color: C.FAIL, breakLine: true } },
    { text: "Directionally concerning but not catastrophic", options: { fontSize: 10, color: C.MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "LoRA \u2265 Full across all models", options: { fontSize: 11, bold: true, color: C.STEEL, breakLine: true } },
    { text: "285\u00d7 fewer params, no forgetting risk", options: { fontSize: 10, color: C.MUTED, breakLine: true } },
  ], {
    x: 5.9, y: 1.3, w: 3.6, h: 3.8,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 11: App Demo (was 10)
// ---------------------------------------------------------------------------
function slide11_demo(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Interactive Visualization Platform", "Precompute \u2192 HDF5/SQLite/PNG cache \u2192 FastAPI (25 endpoints) \u2192 React + Vite");
  addAccentBar(slide);
  addSlideNumber(slide, 11);

  // 2x2 screenshot grid
  const screenshots = [
    { file: "screenshot_gallery.png", label: "Gallery" },
    { file: "screenshot_dashboard.png", label: "Dashboard" },
    { file: "screenshot_image_detail.png", label: "Image Detail" },
    { file: "screenshot_compare.png", label: "Compare" },
  ];

  screenshots.forEach((s, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.4 + col * 4.7;
    const y = 1.2 + row * 2.15;

    slide.addImage({
      path: imgPath(s.file),
      x: x, y: y, w: 4.4, h: 1.85,
      sizing: { type: "contain", w: 4.4, h: 1.85 },
    });

    // Label
    slide.addText(s.label, {
      x: x, y: y + 1.85, w: 4.4, h: 0.22,
      fontSize: 10, fontFace: FONT.BODY, color: C.MUTED, align: "center",
    });
  });
}

// ---------------------------------------------------------------------------
// SLIDE 12: Engineering (was 11)
// ---------------------------------------------------------------------------
function slide12_engineering(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Engineering Highlights", "Built for reproducibility and scale");
  addAccentBar(slide);
  addSlideNumber(slide, 12);

  // Architecture flow (shapes)
  const flowBoxes = [
    { label: "WikiChurches\nDataset", x: 0.3, color: C.WARM_GRAY },
    { label: "Precompute\nPipeline", x: 2.3, color: C.STEEL },
    { label: "FastAPI\nBackend", x: 4.8, color: C.TEAL },
    { label: "React\nFrontend", x: 7.3, color: C.TERRA },
  ];

  flowBoxes.forEach((b) => {
    slide.addShape("roundRect", {
      x: b.x, y: 1.35, w: 1.7, h: 0.9,
      fill: { color: b.color },
      rectRadius: 0.06,
    });
    slide.addText(b.label, {
      x: b.x, y: 1.35, w: 1.7, h: 0.9,
      fontSize: 12, fontFace: FONT.BODY, color: C.WHITE, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
  });

  // Arrows between flow boxes
  [2.05, 4.55, 7.05].forEach((ax) => {
    slide.addText("\u2192", {
      x: ax, y: 1.5, w: 0.25, h: 0.6,
      fontSize: 24, fontFace: FONT.BODY, color: C.MUTED,
      align: "center", valign: "middle",
    });
  });

  // Output formats under precompute
  slide.addText("HDF5  \u2022  SQLite  \u2022  PNG", {
    x: 2.3, y: 2.3, w: 2.2, h: 0.25,
    fontSize: 9, fontFace: FONT.BODY, color: C.MUTED, align: "center",
  });

  // Stats
  const stats = [
    { num: "7", label: "Vision models with unified VisionBackbone protocol" },
    { num: "25", label: "REST API endpoints across 4 routers" },
    { num: "5", label: "Precompute scripts (attention, features, heatmaps, metrics)" },
    { num: "50", label: "Tracked issues via bd (beads), 48 closed" },
    { num: "0", label: "GPU needed at runtime \u2014 all cache-served, <100ms responses" },
  ];

  stats.forEach((s, i) => {
    const y = 2.75 + i * 0.52;
    slide.addText(s.num, {
      x: 0.5, y: y, w: 0.7, h: 0.4,
      fontSize: 22, fontFace: FONT.TITLE, color: C.STEEL, bold: true,
      align: "right", valign: "middle", margin: 0,
    });
    slide.addText(s.label, {
      x: 1.35, y: y, w: 5.0, h: 0.4,
      fontSize: 13, fontFace: FONT.BODY, color: C.BODY,
      valign: "middle", margin: 0,
    });
  });

  // Quality box
  slide.addShape("roundRect", {
    x: 6.5, y: 2.75, w: 3.1, h: 2.35,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    line: { color: C.LIGHT_GRAY, width: 0.5 },
  });
  slide.addText([
    { text: "Quality Gates", options: { fontSize: 13, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 5, breakLine: true } },
    { text: "\u2022 pytest test suite", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 mypy type checking", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 ruff linting", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 IoU quantile bias fix", options: { fontSize: 11, breakLine: true } },
    { text: "   (torch.quantile \u2192 torch.topk)", options: { fontSize: 10, color: C.MUTED, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 Deep audit epic (#9ct)", options: { fontSize: 11, breakLine: true } },
    { text: "   metrics, frontend, security", options: { fontSize: 10, color: C.MUTED, breakLine: true } },
  ], {
    x: 6.7, y: 2.85, w: 2.7, h: 2.15,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 13: Roadmap (was 12) — with professor feedback points 3 & 4
// ---------------------------------------------------------------------------
function slide13_roadmap(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Remaining Work & Roadmap", "Addressing remaining professor feedback");
  addAccentBar(slide);
  addSlideNumber(slide, 13);

  // Feedback status summary bar
  const fbItems = [
    { label: "#1 Metrics", color: C.SUCCESS },
    { label: "#2 SFT Impact", color: C.SUCCESS },
    { label: "#3 Aggregation", color: C.TEAL },
    { label: "#4 Leaderboard", color: C.TEAL },
  ];
  fbItems.forEach((fb, i) => {
    const x = 0.5 + i * 2.3;
    slide.addShape("roundRect", {
      x: x, y: 1.2, w: 2.1, h: 0.4,
      fill: { color: fb.color },
      rectRadius: 0.05,
    });
    slide.addText(fb.label, {
      x: x, y: 1.2, w: 2.1, h: 0.4,
      fontSize: 11, fontFace: FONT.BODY, color: C.WHITE, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
  });

  // Left: Feedback Point 3 — Cross-Layer Aggregation
  slide.addShape("roundRect", {
    x: 0.5, y: 1.9, w: 4.3, h: 3.2,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    line: { color: C.LIGHT_GRAY, width: 0.5 },
  });
  slide.addText([
    { text: "Feedback #3: Cross-Layer Aggregation", options: { fontSize: 13, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 4, breakLine: true } },
    { text: "\u2022 Max pooling across 12 layers", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "   Pixel-wise max \u2192 comprehensive saliency map", options: { fontSize: 10, color: C.MUTED, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 Depth-weighted mean pooling", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "   Exponential decay weighting deeper (semantic) layers", options: { fontSize: 10, color: C.MUTED, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 ALTI (Aggregation of Layer-wise Token Interactions)", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "   State-of-the-art recursive aggregation (Ferrando et al., 2022)", options: { fontSize: 10, color: C.MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Q3: Per-Head Specialization", options: { fontSize: 13, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 3, breakLine: true } },
    { text: "\u2022 Per-head IoU \u00d7 feature-type matrix", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "\u2022 Head ranking by alignment consistency", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "\u2022 Voita et al. (2019), Caron et al. (2021)", options: { fontSize: 10, color: C.MUTED, breakLine: true } },
  ], {
    x: 0.7, y: 2.0, w: 3.9, h: 3.0,
    valign: "top", margin: 0,
  });

  // Right: Feedback Point 4 + Open Items
  slide.addShape("roundRect", {
    x: 5.1, y: 1.9, w: 4.5, h: 1.8,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    line: { color: C.LIGHT_GRAY, width: 0.5 },
  });
  slide.addText([
    { text: "Feedback #4: Paradigm-Split Leaderboard", options: { fontSize: 13, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 4, breakLine: true } },
    { text: "\u2022 Formal unimodal vs multimodal grouping in dashboard", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "\u2022 Sub-group averages for IoU, MSE, Coverage", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "\u2022 Potentially add SAM (Segment Anything) encoder", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "\u2022 Text-prompted evaluation for VLMs (CLIP, SigLIP)", options: { fontSize: 11, breakLine: true } },
  ], {
    x: 5.3, y: 2.0, w: 4.1, h: 1.6,
    valign: "top", margin: 0,
  });

  // Open items
  slide.addShape("roundRect", {
    x: 5.1, y: 3.9, w: 4.5, h: 1.2,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
  });
  slide.addText([
    { text: "Other Open Items", options: { fontSize: 12, bold: true, color: C.MUTED, breakLine: true } },
    { text: "", options: { fontSize: 3, breakLine: true } },
    { text: "\u2022 Attention shift diff-map visualization", options: { fontSize: 10, breakLine: true, paraSpaceAfter: 3 } },
    { text: "\u2022 Fine-tuning training loop test coverage", options: { fontSize: 10, breakLine: true, paraSpaceAfter: 3 } },
    { text: "\u2022 Feature-specific forgetting analysis (per architectural element)", options: { fontSize: 10, breakLine: true } },
  ], {
    x: 5.3, y: 4.0, w: 4.1, h: 1.0,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 14: Summary (was 13) — framed around professor feedback
// ---------------------------------------------------------------------------
function slide14_summary(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.CHARCOAL };

  // Title
  slide.addText("Key Takeaways", {
    x: 0.6, y: 0.4, w: 4.5, h: 0.6,
    fontSize: 28, fontFace: FONT.TITLE, color: C.WHITE, bold: true,
  });

  // Takeaway bullets — framed around professor feedback
  slide.addText([
    { text: "Feedback #1 \u2014 Continuous Metrics", options: { fontSize: 13, bold: true, color: C.TEAL, breakLine: true } },
    { text: "5 complementary metrics implemented (IoU + Coverage + MSE + KL + EMD). Threshold-free metrics reveal patterns IoU alone misses.", options: { fontSize: 11, color: C.WHITE, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Feedback #2 \u2014 Preserve / Enhance / Destroy", options: { fontSize: 13, bold: true, color: C.TEAL, breakLine: true } },
    { text: "Contrastive models Enhance; self-distillation Preserves; no model Destroys. LoRA is the safest strategy with 285\u00d7 fewer parameters.", options: { fontSize: 11, color: C.WHITE, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Feedback #4 \u2014 Unimodal vs Multimodal", options: { fontSize: 13, bold: true, color: C.TEAL, breakLine: true } },
    { text: "Unimodal self-distillation > supervised > multimodal. Language alignment does not improve localisation of architectural features.", options: { fontSize: 11, color: C.WHITE, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Next Steps", options: { fontSize: 13, bold: true, color: C.TERRA, breakLine: true } },
    { text: "Feedback #3 (cross-layer aggregation) + Q3 (per-head specialisation) + feedback #4 (formal dashboard split, SAM).", options: { fontSize: 11, color: C.WHITE, breakLine: true } },
  ], {
    x: 0.6, y: 1.1, w: 4.5, h: 3.8,
    valign: "top", margin: 0,
  });

  // Scatter plot on right
  slide.addImage({
    path: imgPath("slide13_scatter.png"),
    x: 5.2, y: 0.3, w: 4.6, h: 4.5,
    sizing: { type: "contain", w: 4.6, h: 4.5 },
  });

  // Caption
  slide.addText("Frozen IoU vs \u0394 IoU: pre-training determines alignment & plasticity", {
    x: 5.2, y: 4.85, w: 4.6, h: 0.3,
    fontSize: 9, fontFace: FONT.BODY, color: C.MUTED, align: "center",
  });

  // Bottom accent
  slide.addShape("rect", {
    x: 0, y: 5.4, w: 10, h: 0.225,
    fill: { color: C.STEEL },
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  console.log("Creating presentation...");

  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "SSL WikiChurches Team";
  pres.title = "SSL Attention Alignment - Mid-Project Progress Update";

  slide1_title(pres);
  console.log("  Slide 1: Title");

  slide2_motivation(pres);
  console.log("  Slide 2: Motivation");

  slide3_rqs(pres);
  console.log("  Slide 3: Research Questions");

  slide4_feedback(pres);
  console.log("  Slide 4: Professor Feedback (NEW)");

  slide5_dataset(pres);
  console.log("  Slide 5: Dataset");

  slide6_models(pres);
  console.log("  Slide 6: Models");

  slide7_methodology(pres);
  console.log("  Slide 7: Methodology");

  slide8_q1(pres);
  console.log("  Slide 8: Q1 Results");

  slide9_q2setup(pres);
  console.log("  Slide 9: Q2 Setup");

  slide10_q2results(pres);
  console.log("  Slide 10: Q2 Results");

  slide11_demo(pres);
  console.log("  Slide 11: App Demo");

  slide12_engineering(pres);
  console.log("  Slide 12: Engineering");

  slide13_roadmap(pres);
  console.log("  Slide 13: Roadmap");

  slide14_summary(pres);
  console.log("  Slide 14: Summary");

  await pres.writeFile({ fileName: OUT_FILE });
  console.log(`\nDone! Presentation saved to: ${OUT_FILE}`);
}

main().catch(console.error);
