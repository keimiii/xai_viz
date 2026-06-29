---
name: playwright-testing
description: Systematic Playwright MCP testing workflow for SSL Attention Visualization app. Use when testing the frontend, running visual tests, verifying UI changes, or when asked to "test the app". Provides detailed checklists for all app pages and features.
---

# Playwright MCP Testing Guide

This skill provides a systematic checklist for visually testing and inspecting the SSL Attention Visualization app using Playwright MCP.

---

## Workflow Instructions

**IMPORTANT**: Follow these instructions exactly:

1. **Use the Checklist System**: Work through the structured checklists systematically. Do NOT skip items or test ad-hoc.

2. **Track Progress**: Track checklist progress in the active `bd` issue notes or your structured test notes. Do not create markdown todo trackers for this.

3. **Sequential Testing**: Work through the phases in order (Phase 1 → Phase 2 → Phase 3 → etc.). Within each phase, complete all checklist items before moving to the next phase.

4. **Document Everything**: For each checklist item:
   - Take a `browser_snapshot` or `browser_take_screenshot` as evidence
   - Note whether the item passed or failed
   - If failed, stop and follow the Bug Handling Workflow below

5. **Report Format**: When reporting results, use this format:
   ```
   ✅ [Item description] - PASSED
   ❌ [Item description] - FAILED: [brief description of issue]
   ```

---

## Prerequisites

1. Start the development servers:
   ```bash
   ./dev.sh
   ```

2. Ensure Playwright MCP is configured in Codex

3. **App URLs:**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

4. **Fallback if Playwright MCP cannot start**:
   - If the browser MCP fails before navigation because of a local environment issue, do not stop at "could not test".
   - Document the exact MCP error first.
   - Then run the repo's Playwright test suite with `npx playwright test` from `app/frontend`, using the same checklist scope you intended to cover.
   - If the existing suite does not cover the changed behavior, add or update Playwright tests before reporting the result.

---

## Bug Handling Workflow

**CRITICAL**: When any bug or visual error is detected during testing, follow this workflow exactly. Do NOT continue testing until the bug is fixed and verified.

1. **STOP testing immediately** - Do not continue to the next checklist item
2. **Document the bug** - Note the page, steps to reproduce, and expected vs actual behavior
3. **Fix the bug** - Make the necessary code changes
4. **Restart the servers**:
   ```bash
   # Kill existing servers
   pkill -f "uvicorn"; pkill -f "vite"

   # Restart
   ./dev.sh
   ```
5. **Verify the fix with Playwright MCP**:
   - Navigate back to the same page/state where the bug occurred
   - Confirm the bug is resolved
   - Take a screenshot or snapshot as evidence
6. **Resume testing** from where you left off

This ensures bugs are caught and fixed immediately rather than accumulating a backlog of issues.

---

## Getting Started with Playwright MCP

### Navigation Commands
```
# Navigate to the app
mcp__playwright__browser_navigate → url: "http://localhost:5173"

# Take a snapshot (preferred over screenshot for accessibility)
mcp__playwright__browser_snapshot

# Take a screenshot for visual inspection
mcp__playwright__browser_take_screenshot

# Click an element (use ref from snapshot)
mcp__playwright__browser_click → element: "description", ref: "e123"

# Run custom JavaScript for complex interactions
mcp__playwright__browser_run_code → code: "async (page) => { ... }"
```

### Useful Patterns
```javascript
// Check for element visibility
await page.locator('text=Some Text').isVisible();

// Wait for element
await page.waitForSelector('selector');

// Get element count
await page.locator('.grid-item').count();

// Check network requests
await page.waitForResponse(response => response.url().includes('/api/'));
```

---

## Testing Workflow

### Phase 1: Navigation & Layout

#### Initial Load
- [ ] Page loads at http://localhost:5173 without console errors
- [ ] "SSL Attention" heading is visible in navigation
- [ ] Navigation bar renders with all links: Gallery, Compare, Dashboard
- [ ] Footer is visible with correct text

#### Navigation Links
- [ ] "Gallery" nav link is clickable and highlights when active
- [ ] "Compare" nav link navigates to /compare
- [ ] "Dashboard" nav link navigates to /dashboard
- [ ] "SSL Attention" logo/title returns to home

---

### Phase 2: Gallery Page (Home)

#### Layout
- [ ] Gallery page loads at "/"
- [ ] Page title or heading indicates gallery content
- [ ] Image grid/list container is visible

#### Image Display
- [ ] Images load and display correctly
- [ ] Image thumbnails are appropriately sized
- [ ] Loading states show while images fetch
- [ ] No broken image placeholders

#### Interaction
- [ ] Hovering over images shows visual feedback (if applicable)
- [ ] Clicking an image navigates to /image/:imageId detail page
- [ ] Pagination or infinite scroll works (if applicable)

#### API Integration
- [ ] Network requests to backend succeed (check for /api/ calls)
- [ ] Error states display gracefully if API fails

---

### Phase 3: Image Detail Page

#### Navigation
- [ ] Can reach page by clicking image in gallery
- [ ] URL shows correct pattern: /image/:imageId
- [ ] Back navigation returns to gallery

#### Layout
- [ ] Page-level tabs render with `Image Detail` and `Q3`
- [ ] Main tab uses a three-column desktop layout: View Settings (left), visualization + annotations stack (center), Metrics (right)
- [ ] Q3 tab keeps the left column focused on Q3 controls, the center column on the viewer + annotations stack, and the right column on the Q3 scope card
- [ ] Main tab center column displays the main image with attention overlay, the playback slider card, and the annotations card beneath the viewer

#### Annotations Panel (Below Viewer)
- [ ] Architectural styles are listed as badges
- [ ] Number of bounding boxes is shown
- [ ] Scrollable list of bboxes with dimensions (% width x % height)
- [ ] Clicking bbox in list selects it (when bboxes are shown)
- [ ] Helper text matches the active mode and explains how bbox interaction works when bounding boxes are shown

#### Attention Viewer (Center Column)
- [ ] Main image displays with attention heatmap overlay
- [ ] Overlay toggle button appears on hover (top-right of image)
- [ ] Clicking toggle switches between attention overlay and original image
- [ ] Info badges display: current model, layer number
- [ ] Colormap legend is visible below the image

#### Control Panel (Left Column)
- [ ] Model selector dropdown shows all 7 models (dinov2, dinov3, mae, clip, siglip, siglip2, resnet50)
- [ ] Layer slider range adjusts per model (e.g., 0-11 for ViTs, 0-3 for ResNet-50)
- [ ] Percentile threshold selector shows options: 90%, 85%, 80%, 70%, 60%, 50%
- [ ] "Show Bounding Boxes" toggle is present (defaults to ON)
- [ ] Changing model updates the attention visualization
- [ ] Changing layer updates the attention visualization
- [ ] Changing percentile updates the attention overlay
- [ ] Tooltip help icons (?) appear next to each control label
- [ ] Hovering tooltip icon shows educational explanation
- [ ] Attention Method dropdown appears for models with multiple methods (e.g., DINOv2: CLS Attention, Attention Rollout)
- [ ] Attention Method dropdown is hidden for single-method models (e.g., SigLIP shows only Mean Attention)
- [ ] Changing attention method updates the attention visualization and metrics
- [ ] Attention Head dropdown appears for per-head-compatible configurations and defaults to "All (Fused)"
- [ ] Attention Head dropdown offers fused plus all head indices for the selected ViT model
- [ ] Selecting a specific head updates the viewer badge to show `Head N`
- [ ] Switching from DINOv2 `CLS` to `Rollout` hides the Attention Head dropdown and resets head state back to fused when returning to `CLS`
- [ ] Switching to ResNet-50 hides the Attention Head dropdown

#### Q3 Tab Controls
- [ ] Q3 tab shows the dedicated `Q3 Controls` card instead of the default control panel
- [ ] Q3 controls expose only the primary Q3 models and the Q3 variant selector
- [ ] Q3 controls show layer, ranking metric, head, and Show Bounding Boxes controls
- [ ] Q3 mode switch appears above the viewer and supports `Head Attention` and `Feature Similarity`
- [ ] Q3 scope or backfill messaging appears when the selected variant lacks per-head cache support

#### Similarity Heatmap Controls (Left Column - below Core Controls)
- [ ] "Similarity Heatmap" section header is visible
- [ ] Heatmap Style dropdown shows options: Smooth Gradient, Squares, Circles
- [ ] Opacity slider allows adjusting heatmap transparency (20%-90%)
- [ ] Changing heatmap style updates the similarity visualization when bbox selected
- [ ] Tooltip help icons (?) appear next to Heatmap Style and Opacity controls

#### Layer Animation Slider
- [ ] Play/Pause button is visible
- [ ] Navigation buttons: |< (first), < (prev), > (next), >| (last)
- [ ] Play button auto-cycles through layers 0-11
- [ ] Animation STOPS at last layer (does NOT loop back to 0)
- [ ] Clicking Play when at last layer resets to layer 0 and starts playing
- [ ] Pause button stops the animation
- [ ] "Early layers" and "Late layers" labels are visible

#### Metrics Chart Panel
- [ ] Metrics panel shows the interpretation copy relating the heatmap and chart
- [ ] Active-layer indicator updates between focused and playing states
- [ ] Metric toggles render IoU, Coverage, MSE, and KL Divergence enabled by default
- [ ] Metric toggle labels make directionality explicit (for example, "IoU Score (Higher better)")
- [ ] Chart renders layer-by-layer lines instead of the old static metric cards
- [ ] Chart highlight follows the current layer when the user scrubs or clicks first/prev/next/last
- [ ] While playing, the reveal status switches to "Revealing layers 0-N" and advances with the animation
- [ ] Percentile semantics note explains that IoU changes while Coverage, MSE, and KL stay fixed
- [ ] Selecting a bbox switches the metrics panel to bbox mode with bbox-specific context
- [ ] Deselecting bbox reverts the chart to union-of-all-bboxes metrics

#### Bounding Box Interaction
- [ ] Bounding boxes are shown by default (toggle defaults ON)
- [ ] Bbox overlays render on the image
- [ ] Clicking a bbox in the annotations list selects it
- [ ] Selected bbox is highlighted (green) in both image and annotations list
- [ ] Feature similarity heatmap loads when bbox is selected
- [ ] Per-bbox metrics update in the Metrics card when bbox is selected
- [ ] Clicking outside or re-clicking deselects the bbox
- [ ] When a specific attention head is selected, bbox highlighting still works but the attention overlay stays in per-head mode instead of switching to the similarity heatmap takeover

#### Navigation Links
- [ ] No page-local "Compare Models" CTA appears beneath the image-detail metrics panel

---

### Phase 4: Compare Page

#### Layout
- [ ] Compare page loads at "/compare"
- [ ] Page layout supports side-by-side comparison
- [ ] Selection controls are visible for image and comparison type

#### Selection State
- [ ] Can select an image for comparison
- [ ] Comparison Type dropdown shows "Model vs Model" and "Frozen vs Fine-tuned"
- [ ] Model selectors appear for the model-vs-model flow once an image is selected

#### Comparison Features
- [ ] Both images render side-by-side
- [ ] Attention patterns can be compared visually
- [ ] Method-context banner explains whether the selected method is shared or defaults per model
- [ ] Inline comparison metrics show IoU, MSE, and KL for both selected models
- [ ] Switching to a lower-depth model pair keeps the comparison working without request errors
- [ ] SigLIP heatmaps render correctly (not 404 errors)
- [ ] ResNet-50 heatmaps render correctly (not 404 errors)

---

### Phase 5: Dashboard Page

#### Initial Load (CRITICAL - check for blank screen!)
- [ ] Dashboard page loads at "/dashboard"
- [ ] Page is NOT blank - verify actual content renders
- [ ] "Dashboard" heading is visible
- [ ] Loading states show while data fetches (not indefinite spinner)
- [ ] Page tabs render with `Overview` and `Q3`
- [ ] Metric selector shows IoU, Coverage, MSE, KL, and EMD options
- [ ] Threshold-free info banner appears for Coverage and the continuous metrics when selected

#### Model Leaderboard (Left Sidebar)
- [ ] Leaderboard card is visible with heading
- [ ] Ranked list shows all 7 models (dinov2, dinov3, mae, clip, siglip, siglip2, resnet50)
- [ ] Top 3 models have distinct highlighted rank badges
- [ ] Each row shows: model name, best layer, selected metric score
- [ ] Clicking a model row selects it

#### Layer Progression Chart (Main Area)
- [ ] Line chart is visible (not blank/missing)
- [ ] X-axis shows layers (L0-L11)
- [ ] Y-axis updates to match the selected metric scale and directionality
- [ ] Multiple colored lines render (one per model)
- [ ] Legend identifies each model

#### IoU by Architectural Style (Bottom Left)
- [ ] Bar chart is visible (not blank/missing)
- [ ] Bars display IoU values per style
- [ ] Style names are readable
- [ ] Card remains explicitly labeled as IoU-only
- [ ] "No style data available" message shows if no data

#### Feature Type Breakdown (Bottom Right)
- [ ] "Feature Type Breakdown" card is visible
- [ ] Search box for filtering features is present
- [ ] Sort controls (IoU, Count, Name) are visible and functional
- [ ] Feature list shows feature name, the selected metric score (color-coded), and bbox count
- [ ] Score color coding follows the selected metric's directionality instead of assuming higher-is-better
- [ ] "Show more" button appears when more features available
- [ ] Clicking sort button changes list ordering

#### Quick Actions Card
- [ ] "Browse Images" link navigates to gallery
- [ ] "Compare Models" link navigates to compare page
- [ ] Pre-computation notice is visible (yellow alert)

#### Percentile Threshold Control
- [ ] Dropdown at top shows percentile options
- [ ] Changing percentile updates leaderboard
- [ ] Changing percentile updates charts
- [ ] In MSE/KL mode, percentile changes keep the threshold-free leaderboard ordering stable

#### Q3 Per-Head Specialization
- [ ] "Q3 Per-Head Specialization" section is visible
- [ ] Q3 local controls show model, variant, layer, metric, and percentile selectors
- [ ] Frozen-to-adapted delta panel renders or shows a clear unavailable state
- [ ] Single-variant Head Ranking table renders head rows with score and rank summary columns
- [ ] Head × Feature heatmap renders feature rows and head columns with hover and selection readouts
- [ ] Selecting a ranking row or heatmap cell opens the inline exemplar picker
- [ ] Switching the Q3 metric updates the explainer text and ranking direction
- [ ] Unsupported models (for example, ResNet-50) show a clear empty-state explanation instead of a blank panel

---

### Phase 6: Desktop Layout Verification

#### Standard Desktop (1280px width)
- [ ] Full multi-column layouts display correctly
- [ ] Gallery shows proper column grid
- [ ] Image detail main tab shows View Settings + viewer/annotations + Metrics without crowding
- [ ] Image detail Q3 tab keeps the viewer visually dominant while annotations stay directly beneath it
- [ ] Dashboard shows leaderboard + charts side-by-side
- [ ] Maximum content width is respected
- [ ] Adequate whitespace and spacing

#### Wide Desktop (1920px width)
- [ ] Layout remains centered and readable
- [ ] No excessive stretching of content
- [ ] Charts and visualizations scale appropriately

---

### Phase 7: Error Handling

#### Network Errors
- [ ] Stopping backend shows appropriate error state (not blank page)
- [ ] Error messages are user-friendly
- [ ] Retry or refresh guidance is provided

#### Invalid Routes
- [ ] Navigating to /invalid-route shows the custom 404 page with recovery links
- [ ] Invalid image ID (/image/nonexistent) handled gracefully

#### Broken Links (CRITICAL - catch links to undefined routes!)
- [ ] All navigation links in the app lead to valid, rendering pages
- [ ] No internal links result in blank screens
- [ ] Check console for React Router warnings about unmatched routes

#### Edge Cases
- [ ] Empty states display when no data (not blank)
- [ ] Missing API data shows fallback UI (not crash)
- [ ] Very long text doesn't break layout
- [ ] Special characters render correctly

---

### Phase 8: Performance

- [ ] Initial page load feels responsive (< 3 seconds)
- [ ] Navigation between pages is instant
- [ ] Images lazy-load appropriately
- [ ] No visible layout shifts during load
- [ ] No memory leaks during extended use

---

## Quick Smoke Test Checklist

For rapid testing, verify these critical paths:

1. [ ] App loads at localhost:5173
2. [ ] Gallery page shows images (grid is populated)
3. [ ] Can click image to view detail page
4. [ ] Image detail shows attention visualization overlay
5. [ ] Control panel controls work (model, layer, percentile dropdowns)
6. [ ] Tooltip help icons (?) appear next to control labels
7. [ ] Attention Method dropdown appears for DINOv2 (CLS Attention / Attention Rollout)
8. [ ] Layer Play button stops at last layer (doesn't loop)
9. [ ] Heatmap Style dropdown (Smooth/Squares/Circles) visible in Similarity section
10. [ ] Metrics panel shows IoU, Coverage, MSE, KL, and EMD toggles plus bbox/union mode switching
11. [ ] Compare page loads, preserves method context appropriately, and shows IoU/MSE/KL
12. [ ] **Dashboard page renders content (NOT blank screen!)**
13. [ ] Dashboard metric selector switches between IoU, Coverage, MSE, KL, and EMD
14. [ ] Dashboard shows leaderboard, charts, and Feature Type Breakdown
15. [ ] Dashboard Q3 tab shows the delta panel, ranking table, heatmap, and exemplar drill-down flow
16. [ ] Image detail page shows the Attention Head selector for supported model/method combinations
17. [ ] Feature Type Breakdown shows searchable feature list with metric-aware scores
18. [ ] Invalid routes show the custom 404 page instead of a blank screen
19. [ ] No console errors throughout
20. [ ] No blank pages anywhere in the app

---

## Reporting Issues

When documenting bugs, include:
1. **Page/Route** where issue occurred
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Screenshot or snapshot** (use `mcp__playwright__browser_take_screenshot`)
6. **Console messages** (use `mcp__playwright__browser_console_messages`)

---

## Notes

- Playwright's mouse simulation may not perfectly replicate human interaction
- Use `browser_run_code` for complex JavaScript interactions
- Always take a `browser_snapshot` before clicking to get accurate element refs
- The `browser_snapshot` tool is preferred over screenshots for accessibility testing

---

## Checklist Summary

When instructed to perform Playwright testing, follow this workflow:

```
1. Start dev servers (./dev.sh)
2. Navigate to http://localhost:5173
3. Work through phases sequentially:

   PHASE 1: Navigation & Layout
   └── Initial Load (4 items)
   └── Navigation Links (4 items)

   PHASE 2: Gallery Page
   └── Layout (3 items)
   └── Image Display (4 items)
   └── Interaction (4 items)
   └── API Integration (2 items)

   PHASE 3: Image Detail Page
   └── Navigation (3 items)
   └── Layout (4 items) - tab-aware main/Q3 layouts
   └── Annotations Panel (5 items) - below the viewer
   └── Attention Viewer (5 items) - center column
   └── Control Panel (17 items) - main-tab settings, tooltips, and per-head controls
   └── Q3 Tab Controls (5 items) - scoped Q3 controls and mode switch
   └── Similarity Heatmap Controls (5 items) - style dropdown, opacity
   └── Layer Animation Slider (7 items) - includes stop-at-end behavior
   └── Metrics Chart Panel (10 items) - directionality, playback reveal, bbox mode
   └── Bounding Box Interaction (8 items) - includes per-bbox metrics and per-head overlay behavior
   └── Navigation Links (1 item)

   PHASE 4: Compare Page
   └── Layout (3 items)
   └── Selection State (3 items)
   └── Comparison Features (7 items) - includes method context plus SigLIP/ResNet-50 verification

   PHASE 5: Dashboard Page - CRITICAL (check for blank screen!)
   └── Initial Load (7 items)
   └── Model Leaderboard (5 items)
   └── Layer Progression Chart (5 items)
   └── IoU by Architectural Style (5 items)
   └── Feature Type Breakdown (7 items) - new component
   └── Quick Actions Card (3 items)
   └── Percentile Threshold Control (4 items)
   └── Q3 Per-Head Specialization (8 items)

   PHASE 6: Desktop Layout Verification
   └── Standard Desktop 1280px (6 items)
   └── Wide Desktop 1920px (3 items)

   PHASE 7: Error Handling
   └── Network Errors (3 items)
   └── Invalid Routes (2 items)
   └── Broken Links - CRITICAL (3 items)
   └── Edge Cases (4 items)

   PHASE 8: Performance (5 items)

4. For each item: test → document result (✅/❌) → if failed, STOP and fix
5. Provide final summary report with all results
```

**Total checklist items**: ~148 items
