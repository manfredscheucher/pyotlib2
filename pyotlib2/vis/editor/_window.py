"""EditorWindow: interactive point set editor, modelled after qtvis2.

Architecture
------------
- EditorState (_state.py): pure-Python, integer coordinates, OT logic
- EditorScene:  QGraphicsScene — full lines, points, hull overlay
- EditorView:   QGraphicsView  — zoom (wheel + +/-), pan (middle-drag)
- InfoPanel:    coordinates + L-matrix display
- PropertiesPanel: live property values
- PropertyDialog: "Select properties" popup
- ControlPanel: toggles and action buttons
- EditorWindow: main window wiring everything together

Coordinate mapping
------------------
The model (EditorState) keeps the original integer coordinates of the
loaded point set (math convention, y up).  The only mapping is
scene = (x, -y) (see _to_scene / _to_model), which is exact for integers;
fitting and zooming are done by the QGraphicsView transform.  Points,
labels and pens are drawn in pixel units (ItemIgnoresTransformations /
cosmetic pens), so the drawing works for any coordinate range.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QPointF, QLineF, QRectF, Signal, QThread
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QPolygonF, QKeySequence, QAction, QFont,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem,
    QGraphicsPolygonItem, QGraphicsLineItem, QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QStatusBar, QFileDialog, QLabel, QGraphicsItem, QWidget,
    QHBoxLayout, QVBoxLayout, QTextEdit, QCheckBox, QGroupBox,
    QSplitter, QPushButton, QFrame, QComboBox, QDialog,
    QListWidget, QListWidgetItem, QDialogButtonBox, QScrollArea,
    QAbstractItemView, QSizePolicy, QMessageBox,
)

from pyotlib2.vis.editor._state import (
    EditorState, GRID_MAX, ALL_PROPERTIES, DEFAULT_PROPERTIES,
)

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

POINT_RADIUS     = 6            # in pixels (independent of zoom / coordinates)
COLOR_POINT      = QColor("#2c3e50")
COLOR_POINT_SEL  = QColor("#e74c3c")
COLOR_POINT_COL  = QColor("#e74c3c")   # collinear → red
COLOR_POINT_HOVER = QColor("#e67e22")
COLOR_HULL_FILL  = QColor(52, 152, 219, 25)
COLOR_HULL_LINE  = QColor("#3498db")
COLOR_ONION      = [
    QColor(52, 152, 219, 25),
    QColor(46, 204, 113, 25),
    QColor(155, 89, 182, 25),
    QColor(230, 126, 34, 25),
]
COLOR_LINE_NORMAL  = QColor(180, 180, 180, 100)
COLOR_LINE_HIDDEN  = QColor(180, 180, 180, 20)
COLOR_AXIS         = QColor(200, 200, 200, 180)

# ---------------------------------------------------------------------------
# DraggablePoint
# ---------------------------------------------------------------------------

def _to_scene(x: int, y: int) -> QPointF:
    """Model (math, y up) → scene (y down).  Exact for integers."""
    return QPointF(x, -y)


def _to_model(pos: QPointF) -> tuple[int, int]:
    """Scene → nearest integer model coordinates."""
    return int(round(pos.x())), int(round(-pos.y()))


def _cosmetic_pen(color: QColor, width: float) -> QPen:
    """Pen whose width is in pixels, independent of the view transform."""
    pen = QPen(color, width)
    pen.setCosmetic(True)
    return pen


class DraggablePoint(QGraphicsEllipseItem):
    def __init__(self, index: int, x: int, y: int, scene: "EditorScene"):
        r = POINT_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.index = index
        self._scene = scene
        self._selected = False
        self._collinear = False
        self._syncing = False
        self.setPos(_to_scene(x, y))
        # drawn in pixels; the label is a child and inherits this
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self._refresh_color()
        self.setPen(QPen(Qt.NoPen))
        self.setZValue(10)

        lbl = QGraphicsSimpleTextItem(str(index), self)
        lbl.setBrush(QBrush(QColor("#7f8c8d")))
        lbl.setFont(QFont("Arial", 9))
        lbl.setPos(r + 2, -r - lbl.boundingRect().height())

    def sync_pos(self, x: int, y: int) -> None:
        """Programmatic move to model (x, y), bypassing the drag logic."""
        self._syncing = True
        self.setPos(_to_scene(x, y))
        self._syncing = False

    def _refresh_color(self):
        if self._collinear:
            self.setBrush(QBrush(COLOR_POINT_COL))
        elif self._selected:
            self.setBrush(QBrush(COLOR_POINT_SEL))
        else:
            self.setBrush(QBrush(COLOR_POINT))

    def set_selected(self, val: bool):
        self._selected = val
        self._refresh_color()

    def set_collinear(self, val: bool):
        if val != self._collinear:
            self._collinear = val
            self._refresh_color()

    def hoverEnterEvent(self, event):
        if not self._selected and not self._collinear:
            self.setBrush(QBrush(COLOR_POINT_HOVER))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._refresh_color()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._scene.toggle_selected(self.index)
            event.accept()
            return
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if self._syncing:
            return super().itemChange(change, value)
        if change == QGraphicsItem.ItemPositionChange:
            # snap the proposed position to integer model coordinates,
            # clamped to the scene rect
            rect = self._scene.sceneRect()
            pos = QPointF(min(max(value.x(), rect.left()), rect.right()),
                          min(max(value.y(), rect.top()), rect.bottom()))
            ix, iy = _to_model(pos)
            # OT lock: veto the move, the point stays where it is
            if self._scene.ot_lock and self._scene.state.would_change_ot(self.index, ix, iy):
                return self.pos()
            return _to_scene(ix, iy)
        if change == QGraphicsItem.ItemPositionHasChanged:
            ix, iy = _to_model(self.pos())
            self._scene.state.move_point(self.index, ix, iy)
            self._scene.refresh(moved=self.index)
        return super().itemChange(change, value)


# ---------------------------------------------------------------------------
# EditorScene
# ---------------------------------------------------------------------------

class EditorScene(QGraphicsScene):
    """Scene: full lines through all point pairs + hull overlay + points."""

    updated = Signal()   # point coordinates changed

    def __init__(self, state: EditorState, parent=None):
        super().__init__(parent)
        self.state = state
        self.ot_lock = False

        self._hide_incident = False
        self._highlight_mode = "hull"   # "none" | "hull" | "onion"

        self._point_items: list[DraggablePoint] = []
        self._line_items: list[tuple[int, int, QGraphicsLineItem]] = []
        self._hull_items: list[QGraphicsPolygonItem] = []
        self._border = QGraphicsRectItem()

        self._selected: Optional[int] = None

        self._build_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def move_point(self, index: int, x: int, y: int) -> None:
        """Programmatic move of one point (model coordinates)."""
        self.state.move_point(index, x, y)
        self._point_items[index].sync_pos(*self.state.get_coords()[index])
        self.refresh(moved=index)

    def set_points(self, pts: list[tuple[int, int]]) -> None:
        """Replace all coordinates unchanged (load / optimisation result)
        and refit the scene rect to them."""
        self.state.set_coords(pts)
        self.fit_scene_rect()
        self.refresh()

    def fit_scene_rect(self) -> None:
        """Scene rect = bounding box of the points plus a margin."""
        coords = self.state.get_coords()
        xs = [x for x, _ in coords] or [0]
        ys = [-y for _, y in coords] or [0]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        margin = 0.1 * (max(w, h) or 1)
        rect = QRectF(min(xs) - margin, min(ys) - margin, w + 2 * margin, h + 2 * margin)
        self.setSceneRect(rect)
        self._border.setRect(rect)

    def toggle_selected(self, index: int) -> None:
        if self._selected == index:
            self.clear_selection()
            return
        if self._selected is not None:
            self._point_items[self._selected].set_selected(False)
        self._selected = index
        self._point_items[index].set_selected(True)
        self._refresh_line_pens()

    def clear_selection(self) -> None:
        if self._selected is not None:
            self._point_items[self._selected].set_selected(False)
            self._selected = None
            self._refresh_line_pens()

    def set_hide_incident(self, val: bool) -> None:
        self._hide_incident = val
        self._refresh_line_pens()

    def set_highlight_mode(self, mode: str) -> None:
        self._highlight_mode = mode
        self._refresh_hull()

    def refresh(self, moved: Optional[int] = None) -> None:
        """Redraw after a coordinate change.  *moved*: index of the only
        point that moved (its item is already in place), else all."""
        coords = self.state.get_coords()
        if moved is None:
            for i, (x, y) in enumerate(coords):
                self._point_items[i].sync_pos(x, y)
        # collinear highlighting
        collinear_pts = set()
        for a, b, c in self.state.collinear_triples:
            collinear_pts |= {a, b, c}
        for i, pt in enumerate(self._point_items):
            pt.set_collinear(i in collinear_pts)
        self._refresh_line_geometry(moved)
        self._refresh_hull()
        self.updated.emit()

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build_all(self):
        self._border.setPen(_cosmetic_pen(COLOR_AXIS, 1.5))
        self._border.setZValue(0)
        self.addItem(self._border)
        self.fit_scene_rect()

        n = self.state.n
        coords = self.state.get_coords()

        # lines (full viewport lines, one per pair)
        for i in range(n):
            for j in range(i + 1, n):
                line = QGraphicsLineItem()
                line.setZValue(1)
                self.addItem(line)
                self._line_items.append((i, j, line))

        # points (with their labels)
        for i, (x, y) in enumerate(coords):
            pt = DraggablePoint(i, x, y, self)
            self.addItem(pt)
            self._point_items.append(pt)

        self._refresh_line_pens()
        self.refresh()

    def _refresh_line_geometry(self, moved: Optional[int] = None):
        """Recompute the full lines through each pair of points (only the
        lines through *moved* if given)."""
        coords = self.state.get_coords()
        rect = self.sceneRect()
        for i, j, line in self._line_items:
            if moved is not None and moved != i and moved != j:
                continue
            x0, y0 = coords[i]
            x1, y1 = coords[j]
            # extend line to scene rect boundary
            p1, p2 = _extend_line_to_rect(x0, -y0, x1, -y1, rect)
            line.setLine(p1.x(), p1.y(), p2.x(), p2.y())

    def _refresh_line_pens(self):
        """Line colors (depend on selection / hide-incident only)."""
        pen_normal = _cosmetic_pen(COLOR_LINE_NORMAL, 1)
        pen_hidden = _cosmetic_pen(COLOR_LINE_HIDDEN, 1)
        for i, j, line in self._line_items:
            is_incident = (self._selected is not None and
                           (i == self._selected or j == self._selected))
            line.setPen(pen_hidden if self._hide_incident and is_incident else pen_normal)

    def _refresh_hull(self):
        for item in self._hull_items:
            self.removeItem(item)
        self._hull_items.clear()

        if self._highlight_mode == "none":
            return

        coords = self.state.get_coords()

        if self._highlight_mode == "hull":
            hull_ids = self.state.get_hull_indices()
            if len(hull_ids) >= 3:
                self._draw_polygon(hull_ids, coords, COLOR_HULL_FILL, COLOR_HULL_LINE)

        elif self._highlight_mode == "onion":
            layers = self.state.get_onion_layers()
            for li, layer in enumerate(layers):
                if len(layer) >= 3:
                    color = COLOR_ONION[li % len(COLOR_ONION)]
                    line_color = color.lighter(150)
                    line_color.setAlpha(180)
                    self._draw_polygon(layer, coords, color, line_color)

    def _draw_polygon(self, ids, coords, fill, stroke):
        if len(ids) < 3:
            return
        cx = sum(coords[i][0] for i in ids) / len(ids)
        cy = sum(coords[i][1] for i in ids) / len(ids)
        sorted_ids = sorted(ids, key=lambda i: math.atan2(
            coords[i][1] - cy, coords[i][0] - cx))
        poly = QPolygonF([_to_scene(*coords[i]) for i in sorted_ids])
        item = QGraphicsPolygonItem(poly)
        item.setBrush(QBrush(fill))
        item.setPen(_cosmetic_pen(stroke, 1.5))
        item.setZValue(0)
        self.addItem(item)
        self._hull_items.append(item)


# ---------------------------------------------------------------------------
# EditorView
# ---------------------------------------------------------------------------

class EditorView(QGraphicsView):
    def __init__(self, scene: EditorScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setBackgroundBrush(QBrush(QColor("#f8f9fa")))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._panning = False
        self._pan_start = None

        # fit scene on startup
        self.setMinimumSize(500, 500)

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_view()

    def fit_view(self):
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.scale(1.2, 1.2)
        elif event.key() == Qt.Key_Minus:
            self.scale(1 / 1.2, 1 / 1.2)
        elif event.key() == Qt.Key_Home or event.key() == Qt.Key_0 or (
                event.key() == Qt.Key_0 and event.modifiers() & Qt.ControlModifier):
            self.fit_view()
        elif event.key() == Qt.Key_Escape:
            self.scene().clear_selection()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# PropertyDialog
# ---------------------------------------------------------------------------

class PropertyDialog(QDialog):
    """Modal dialog for selecting which properties to display."""

    def __init__(self, active_keys: list[str], show_times: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select properties")
        self.setMinimumWidth(320)
        self.setMinimumHeight(400)

        layout = QVBoxLayout(self)

        lbl = QLabel("Select properties to display (drag to reorder):")
        layout.addWidget(lbl)

        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)

        # add all properties; check the active ones
        name_to_key = {name: key for name, key in ALL_PROPERTIES}
        key_to_name = {key: name for name, key in ALL_PROPERTIES}

        # first: active properties in order
        added = set()
        for key in active_keys:
            name = key_to_name.get(key, key)
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._list.addItem(item)
            added.add(key)
        # then: remaining
        for name, key in ALL_PROPERTIES:
            if key not in added:
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, key)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self._list.addItem(item)

        layout.addWidget(self._list)

        self._cb_times = QCheckBox("Show computing times")
        self._cb_times.setChecked(show_times)
        layout.addWidget(self._cb_times)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_active_keys(self) -> list[str]:
        result = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.data(Qt.UserRole))
        return result

    def get_show_times(self) -> bool:
        return self._cb_times.isChecked()


# ---------------------------------------------------------------------------
# PropertiesPanel
# ---------------------------------------------------------------------------

class PropertiesPanel(QWidget):
    def __init__(self, state: EditorState, parent=None):
        super().__init__(parent)
        self._state = state
        self._active_keys: list[str] = list(DEFAULT_PROPERTIES)
        self._show_times = False
        self._rows: list[tuple[str, QLabel, QLabel]] = []  # (key, name_lbl, val_lbl)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        grp = QGroupBox("Properties")
        self._grp_layout = QVBoxLayout(grp)
        self._grp_layout.setSpacing(2)
        layout.addWidget(grp)

        self._btn_select = QPushButton("Select properties…")
        self._btn_select.clicked.connect(self._open_dialog)
        layout.addWidget(self._btn_select)

        layout.addStretch()
        self._rebuild_rows()

    def refresh(self):
        for key, name_lbl, val_lbl in self._rows:
            val, elapsed = self._state.compute_property(key)   # cached per OT
            if self._show_times and isinstance(val, int):
                val_lbl.setText(f"{val}  ({elapsed:.3f}s)")
            else:
                val_lbl.setText(str(val))

    def _rebuild_rows(self):
        # clear
        for key, name_lbl, val_lbl in self._rows:
            name_lbl.deleteLater()
            val_lbl.deleteLater()
        self._rows.clear()
        # remove old widgets from layout
        while self._grp_layout.count():
            item = self._grp_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # clear sublayout
                pass

        key_to_name = {key: name for name, key in ALL_PROPERTIES}
        for key in self._active_keys:
            name = key_to_name.get(key, key)
            row = QHBoxLayout()
            name_lbl = QLabel(name + ":")
            name_lbl.setMinimumWidth(100)
            val_lbl = QLabel("—")
            val_lbl.setFont(QFont("Courier", 10))
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setMinimumWidth(80)
            row.addWidget(name_lbl)
            row.addWidget(val_lbl)
            self._grp_layout.addLayout(row)
            self._rows.append((key, name_lbl, val_lbl))

    def _open_dialog(self):
        dlg = PropertyDialog(self._active_keys, self._show_times, self)
        if dlg.exec() == QDialog.Accepted:
            self._active_keys = dlg.get_active_keys()
            self._show_times = dlg.get_show_times()
            self._rebuild_rows()
            self.refresh()


# ---------------------------------------------------------------------------
# InfoPanel
# ---------------------------------------------------------------------------

class InfoPanel(QWidget):
    def __init__(self, state: EditorState, parent=None):
        super().__init__(parent)
        self._state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        grp_coords = QGroupBox("Coordinates")
        cv = QVBoxLayout(grp_coords)
        self._coords_text = QTextEdit()
        self._coords_text.setReadOnly(True)
        self._coords_text.setFont(QFont("Courier", 9))
        self._coords_text.setMaximumHeight(160)
        cv.addWidget(self._coords_text)
        layout.addWidget(grp_coords)

        grp_lambda = QGroupBox("Small lambda")
        lv = QVBoxLayout(grp_lambda)
        self._lambda_text = QTextEdit()
        self._lambda_text.setReadOnly(True)
        self._lambda_text.setFont(QFont("Courier", 9))
        self._lambda_text.setMaximumHeight(180)
        lv.addWidget(self._lambda_text)
        self._checksum_label = QLabel("checksum: —")
        self._checksum_label.setFont(QFont("Courier", 9))
        self._checksum_label.setStyleSheet("color: #7f8c8d;")
        lv.addWidget(self._checksum_label)
        layout.addWidget(grp_lambda)

        layout.addStretch()
        self._shown_version: Optional[int] = None

    def refresh(self):
        coords = self._state.get_coords()
        lines = [f"P{i:2d}: {x:6d}  {y:6d}" for i, (x, y) in enumerate(coords)]
        self._coords_text.setText("\n".join(lines))

        # the L matrix only depends on the order type
        if self._shown_version == self._state.ot_version:
            return
        self._shown_version = self._state.ot_version
        L = self._state.get_l_matrix()
        if L is not None:
            try:
                n = self._state.n
                vals = [[int(L[i, j]) for j in range(n)] for i in range(n)]
                w = max(len(str(v)) for row in vals for v in row)
                rows = [" ".join(str(v).rjust(w) for v in row) for row in vals]
                self._lambda_text.setText("\n".join(rows))
                # 8-hex-digit checksum of the L matrix
                raw = bytes(int(L[i, j]) for i in range(n) for j in range(n))
                digest = hashlib.sha256(raw).hexdigest()[:8]
                self._checksum_label.setText(f"checksum: {digest}")
            except Exception:
                self._lambda_text.setText("(error computing L)")
                self._checksum_label.setText("checksum: —")
        else:
            self._lambda_text.setText("(collinear — no valid OT)")
            self._checksum_label.setText("checksum: —")


# ---------------------------------------------------------------------------
# _run_one_step — pure function, no Qt deps
# ---------------------------------------------------------------------------

def _run_one_step(mode: str, prop: str, direction: str, sl, coords: list) -> list | None:
    """Run one optimization step. Returns new coords list if improved, else None.

    Args:
        mode: "minimize" | "beautify" | "property"
        prop: property key (used only when mode=="property")
        direction: "minimize" | "maximize" (used only when mode=="property")
        sl: SmallLambda with current OT (realization will be set by this function)
        coords: current list of (x, y) integer tuples
    """
    import copy

    if mode == "minimize":
        from pyotlib2.cli.commands import minimize_coords
        sl_copy = copy.copy(sl)
        sl_copy.realization = list(coords)
        new_sl = minimize_coords(sl_copy, trials=1)
        new_coords = new_sl.realization
        if new_coords is None:
            return None
        cur_max = max(abs(v) for p in coords for v in p) if coords else 0
        new_max = max(abs(v) for p in new_coords for v in p) if new_coords else 0
        return list(new_coords) if new_max < cur_max else None

    if mode == "beautify":
        from pyotlib2.cli.commands import beautify_coords
        sl_copy = copy.copy(sl)
        sl_copy.realization = list(coords)
        new_sl = beautify_coords(sl_copy, max_iter=5)
        new_coords = new_sl.realization
        if new_coords is None or new_coords == coords:
            return None
        return list(new_coords)

    if mode == "property":
        from pyotlib2.cli.commands import walk_points, _count_property
        import copy
        sl_copy = copy.copy(sl)
        sl_copy.realization = list(coords)
        cur_val = _count_property(sl_copy, prop)
        for new_sl in walk_points(
            sl_copy, prop,
            random_walk=True,
            trials=1,
            max_steps=1,
            stop_if_improved=True,
            verbose=False,
            maximize=direction == "maximize",
        ):
            if new_sl.realization:
                new_val = _count_property(new_sl, prop)
                improved = new_val < cur_val if direction == "minimize" else new_val > cur_val
                if improved:
                    return list(new_sl.realization)
        return None

    return None


# ---------------------------------------------------------------------------
# OptimizationWorker — background thread
# ---------------------------------------------------------------------------

class OptimizationWorker(QThread):
    """Runs optimization steps in a background thread."""

    improvement_found = Signal(list)   # emits new coords list[(x,y)]
    failed = Signal(str)               # emits the error message; thread ends

    def __init__(self, mode: str, prop: str, direction: str, sl, coords: list, parent=None):
        super().__init__(parent)
        self._mode = mode
        self._prop = prop
        self._direction = direction
        self._sl = sl
        self._coords = list(coords)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import copy
        from pyotlib2.core.point_set import PointSet

        while not self._stop:
            try:
                result = _run_one_step(
                    self._mode, self._prop, self._direction,
                    copy.copy(self._sl), self._coords,
                )
            except Exception as e:
                # report to the UI instead of dying silently
                self.failed.emit(f"{type(e).__name__}: {e}")
                return
            if result is not None:
                self._coords = result
                # rebuild sl from new coords
                ps = PointSet(len(result), result)
                self._sl = ps.to_small_lambda(lazy=False)
                self.improvement_found.emit(result)
            else:
                # no improvement — yield briefly so the thread stays responsive
                self.msleep(50)


# ---------------------------------------------------------------------------
# OptimizationPanel
# ---------------------------------------------------------------------------

class OptimizationPanel(QGroupBox):
    """Unified optimization group box with mode selector and step/auto buttons."""

    step_requested = Signal(str, str, str)      # (mode, prop, direction)
    auto_toggled   = Signal(bool, str, str, str) # (active, mode, prop, direction)

    # mode combo items → (label, key)
    _MODES = [
        ("Coordinate Minimization", "minimize"),
        ("Beautification",          "beautify"),
        ("Property Min/Max",        "property"),
    ]

    def __init__(self, parent=None):
        super().__init__("Optimization", parent)

        from pyotlib2.vis.editor._state import ALL_PROPERTIES

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Mode row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        for label, _ in self._MODES:
            self._mode_combo.addItem(label)
        mode_row.addWidget(self._mode_combo)
        layout.addLayout(mode_row)

        # Property row (only visible in "property" mode)
        self._prop_row = QWidget()
        prop_layout = QVBoxLayout(self._prop_row)
        prop_layout.setContentsMargins(0, 0, 0, 0)
        prop_layout.setSpacing(2)

        pr = QHBoxLayout()
        pr.addWidget(QLabel("Property:"))
        self._prop_combo = QComboBox()
        for name, key in ALL_PROPERTIES:
            self._prop_combo.addItem(name, key)
        pr.addWidget(self._prop_combo)
        prop_layout.addLayout(pr)

        dr = QHBoxLayout()
        dr.addWidget(QLabel("Direction:"))
        self._dir_combo = QComboBox()
        self._dir_combo.addItems(["Minimize", "Maximize"])
        dr.addWidget(self._dir_combo)
        prop_layout.addLayout(dr)

        layout.addWidget(self._prop_row)

        # Buttons row
        btn_row = QHBoxLayout()
        self._btn_step = QPushButton("1 Step")
        self._btn_auto = QPushButton("Auto")
        self._btn_auto.setCheckable(True)
        btn_row.addWidget(self._btn_step)
        btn_row.addWidget(self._btn_auto)
        layout.addLayout(btn_row)

        # Wire up
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._btn_step.clicked.connect(self._on_step)
        self._btn_auto.toggled.connect(self._on_auto_toggled)

        # Initial state
        self._on_mode_changed(0)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_auto_active(self, active: bool):
        """Programmatically set Auto button state without emitting toggled."""
        self._btn_auto.blockSignals(True)
        self._btn_auto.setChecked(active)
        self._btn_auto.blockSignals(False)

    def set_buttons_enabled(self, enabled: bool):
        """Enable/disable step+mode combos (but not auto button)."""
        self._btn_step.setEnabled(enabled)
        self._mode_combo.setEnabled(enabled)
        self._prop_combo.setEnabled(enabled)
        self._dir_combo.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _current_mode(self) -> str:
        return self._MODES[self._mode_combo.currentIndex()][1]

    def _current_prop(self) -> str:
        return self._prop_combo.currentData() or "crossings"

    def _current_direction(self) -> str:
        return "minimize" if self._dir_combo.currentIndex() == 0 else "maximize"

    def _on_mode_changed(self, idx):
        is_property = (self._MODES[idx][1] == "property")
        self._prop_row.setVisible(is_property)

    def _on_step(self):
        self.step_requested.emit(
            self._current_mode(), self._current_prop(), self._current_direction()
        )

    def _on_auto_toggled(self, active: bool):
        self.auto_toggled.emit(
            active,
            self._current_mode(), self._current_prop(), self._current_direction()
        )


# ---------------------------------------------------------------------------
# ControlPanel
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    def __init__(self, scene: EditorScene, parent=None):
        super().__init__(parent)
        self._scene = scene

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.setMaximumWidth(230)

        # --- Navigation group (only shown when multiple OTs loaded) ---
        self._grp_nav = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(self._grp_nav)
        nav_layout.setContentsMargins(4, 4, 4, 4)
        nav_layout.setSpacing(4)
        self._btn_prev = QPushButton("◀")
        self._btn_prev.setMaximumWidth(36)
        self._lbl_nav = QLabel("OT 1 of 1")
        self._lbl_nav.setAlignment(Qt.AlignCenter)
        self._lbl_nav.setFont(QFont("Arial", 10, QFont.Bold))
        self._btn_next = QPushButton("▶")
        self._btn_next.setMaximumWidth(36)
        nav_layout.addWidget(self._btn_prev)
        nav_layout.addWidget(self._lbl_nav, stretch=1)
        nav_layout.addWidget(self._btn_next)
        self._grp_nav.setVisible(False)
        layout.addWidget(self._grp_nav)

        # --- Display group ---
        grp_disp = QGroupBox("Display")
        gl = QVBoxLayout(grp_disp)

        self._cb_hide_incident = QCheckBox("Hide incident lines")
        self._cb_hide_incident.setChecked(False)
        self._cb_hide_incident.stateChanged.connect(
            lambda v: scene.set_hide_incident(bool(v))
        )
        gl.addWidget(self._cb_hide_incident)

        hl_row = QHBoxLayout()
        hl_row.addWidget(QLabel("Highlight:"))
        self._highlight_combo = QComboBox()
        self._highlight_combo.addItems(["Convex hull", "Onion layers", "None"])
        self._highlight_combo.currentIndexChanged.connect(self._on_highlight_changed)
        hl_row.addWidget(self._highlight_combo)
        gl.addLayout(hl_row)

        layout.addWidget(grp_disp)

        # --- OT group ---
        grp_ot = QGroupBox("Order type")
        ol = QVBoxLayout(grp_ot)

        self._cb_lock_ot = QCheckBox("Fix OT (snap back on change)")
        self._cb_lock_ot.setChecked(False)
        ol.addWidget(self._cb_lock_ot)

        layout.addWidget(grp_ot)

        # --- Optimization group ---
        self._opt_panel = OptimizationPanel()
        layout.addWidget(self._opt_panel)

        # --- View ---
        self._btn_fit = QPushButton("Fit view")
        layout.addWidget(self._btn_fit)

        # --- Save ---
        self._btn_save = QPushButton("Save coordinates…")
        layout.addWidget(self._btn_save)

        layout.addStretch()

    def _on_highlight_changed(self, idx):
        modes = ["hull", "onion", "none"]
        self._scene.set_highlight_mode(modes[idx])

    def on_lock_ot(self, fn):
        self._cb_lock_ot.stateChanged.connect(fn)

    def on_fit(self, fn):
        self._btn_fit.clicked.connect(fn)

    def on_save(self, fn):
        self._btn_save.clicked.connect(fn)

    def on_prev(self, fn):
        self._btn_prev.clicked.connect(fn)

    def on_next(self, fn):
        self._btn_next.clicked.connect(fn)

    def update_nav(self, index: int, total: int):
        self._lbl_nav.setText(f"OT {index + 1} of {total}")
        self._grp_nav.setVisible(total > 1)


# ---------------------------------------------------------------------------
# EditorWindow
# ---------------------------------------------------------------------------

class EditorWindow(QMainWindow):
    def __init__(
        self,
        filepath: Optional[str] = None,
        *,
        n: Optional[int] = None,
        fmt: Optional[str] = None,
    ):
        super().__init__()
        self.setWindowTitle("pyotlib2 editor")

        # Multi-OT navigation state
        self._all_ots: list = _load_order_types(filepath, n=n, fmt=fmt)
        self._ot_index: int = 0
        self._dirty: bool = False
        self._loading: bool = False

        # model keeps the original coordinates (and point colors, if any)
        if self._all_ots:
            pts = list(self._all_ots[0].realization)
            self._coloring = self._all_ots[0].coloring
        else:
            pts = _load_points(filepath, n=n, fmt=fmt)
            self._coloring = None

        self.state = EditorState(
            pts,
            on_ot_changed=self._on_ot_status,
            on_collinear_changed=self._on_collinear,
        )
        self._scene = EditorScene(self.state)
        self._scene.updated.connect(self._on_scene_updated)

        # Background optimization worker (None when inactive)
        self._worker: Optional[OptimizationWorker] = None

        # --- layout ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        self._ctrl = ControlPanel(self._scene)
        self._ctrl.on_lock_ot(self._on_lock_ot_toggled)
        self._ctrl.on_save(self._save)
        self._ctrl.on_prev(self._on_prev)
        self._ctrl.on_next(self._on_next)
        if self._all_ots:
            self._ctrl.update_nav(self._ot_index, len(self._all_ots))
        # Wire optimization panel signals
        opt = self._ctrl._opt_panel
        opt.step_requested.connect(self._opt_step)
        opt.auto_toggled.connect(self._opt_auto_toggle)
        main_layout.addWidget(self._ctrl)

        self._view = EditorView(self._scene)
        self._ctrl.on_fit(self._view.fit_view)
        main_layout.addWidget(self._view, stretch=1)

        right = QWidget()
        right.setMaximumWidth(270)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self._info = InfoPanel(self.state)
        right_layout.addWidget(self._info)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        right_layout.addWidget(line)

        self._props = PropertiesPanel(self.state)
        right_layout.addWidget(self._props)

        main_layout.addWidget(right)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "Left-drag: move point   Right-click: select   Esc: deselect   "
            "Wheel: zoom   Middle-drag: pan"
        )

        self._build_menu()
        self._refresh_panels()

    # ------------------------------------------------------------------
    # Programmatic API (for tests)
    # ------------------------------------------------------------------

    def move_point(self, index: int, x: int, y: int) -> None:
        self._scene.move_point(index, x, y)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_scene_updated(self):
        if not self._loading:
            self._dirty = True
        self._refresh_panels()

    def _on_ot_status(self, changed: bool):
        self._info.refresh()

    def _on_collinear(self, triples):
        if triples:
            pts = sorted(set(i for t in triples for i in t))
            self._status.showMessage(
                f"WARNING: collinear points {pts} — cannot save in this configuration!",
                5000,
            )

    def _refresh_panels(self):
        self._info.refresh()
        self._props.refresh()

    def _on_lock_ot_toggled(self, state):
        active = bool(state)
        self._scene.ot_lock = active
        if active:
            # snapshot current OT as the reference to compare against
            self.state.lock_current_ot()
            self._status.showMessage("OT locked — current configuration is the reference.", 3000)
        self._refresh_panels()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _confirm_discard(self) -> bool:
        """Return True if OK to discard current state (no changes or user confirmed)."""
        if not self._dirty:
            return True
        msg = QMessageBox(self)
        msg.setWindowTitle("Unsaved changes")
        msg.setText("The point set has been modified. Save the changes?")
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Save)
        result = msg.exec()
        if result == QMessageBox.Save:
            self._save()
            return not self._dirty   # False if Save dialog was cancelled
        return result == QMessageBox.Discard

    def _load_ot(self, index: int):
        """Load OT at index from _all_ots into the editor."""
        self._ot_index = index
        sl = self._all_ots[index]
        self._coloring = sl.coloring
        self._loading = True
        self._scene.set_points(sl.realization)
        self._loading = False
        self._dirty = False
        self._view.fit_view()
        self._ctrl.update_nav(self._ot_index, len(self._all_ots))
        self._refresh_panels()

    def _on_prev(self):
        if not self._all_ots:
            return
        if not self._confirm_discard():
            return
        self._load_ot((self._ot_index - 1) % len(self._all_ots))

    def _on_next(self):
        if not self._all_ots:
            return
        if not self._confirm_discard():
            return
        self._load_ot((self._ot_index + 1) % len(self._all_ots))

    def _set_points_from_realization(self, coords):
        """Show a new realization unchanged; only the view is refitted."""
        self._scene.set_points(coords)
        self._view.fit_view()

    def _get_sl_and_coords(self):
        """Return (sl, coords) or (None, None) if collinear."""
        sl = self.state.get_small_lambda()
        if sl is None:
            return None, None
        coords = list(self.state.get_coords())
        sl.realization = coords
        return sl, coords

    def _set_canvas_locked(self, locked: bool):
        """Disable/enable point dragging and non-Auto buttons."""
        for pt in self._scene._point_items:
            pt.setFlag(QGraphicsItem.ItemIsMovable, not locked)
        self._ctrl._opt_panel.set_buttons_enabled(not locked)
        self._ctrl._cb_lock_ot.setEnabled(not locked)
        self._ctrl._btn_save.setEnabled(not locked)

    # ------------------------------------------------------------------
    # Optimization: single step
    # ------------------------------------------------------------------

    def _opt_step(self, mode: str, prop: str, direction: str):
        import copy
        sl, coords = self._get_sl_and_coords()
        if sl is None:
            self._status.showMessage("Cannot optimize: collinear configuration.", 4000)
            return
        try:
            result = _run_one_step(mode, prop, direction, copy.copy(sl), coords)
            if result is not None:
                self._set_points_from_realization(result)
                self._status.showMessage("Step: improvement found.", 3000)
            else:
                self._status.showMessage("Step: no improvement found.", 3000)
        except Exception as e:
            self._status.showMessage(f"Step failed: {e}", 5000)

    # ------------------------------------------------------------------
    # Optimization: auto (background thread)
    # ------------------------------------------------------------------

    def _opt_auto_toggle(self, active: bool, mode: str, prop: str, direction: str):
        if active:
            sl, coords = self._get_sl_and_coords()
            if sl is None:
                self._status.showMessage("Cannot optimize: collinear configuration.", 4000)
                # un-check the auto button
                self._ctrl._opt_panel.set_auto_active(False)
                return
            import copy
            self._worker = OptimizationWorker(mode, prop, direction, copy.copy(sl), coords)
            self._worker.improvement_found.connect(self._on_improvement)
            self._worker.failed.connect(self._on_worker_failed)
            self._worker.finished.connect(self._on_worker_finished)
            self._set_canvas_locked(True)
            self._status.showMessage("Auto optimization running…")
            self._worker.start()
        else:
            if self._worker is not None:
                self._worker.stop()
                self._worker.wait()
                self._worker = None
            self._set_canvas_locked(False)
            self._status.showMessage("Auto stopped.", 3000)

    def _on_improvement(self, coords: list):
        """Called from main thread when worker finds an improvement."""
        self._set_points_from_realization(coords)
        self._refresh_panels()

    def _on_worker_failed(self, message: str):
        """Worker raised: tell the user (it stops; _on_worker_finished unlocks)."""
        self._status.showMessage(f"Auto optimization failed: {message}", 8000)
        QMessageBox.warning(self, "Optimization failed", message)

    def _on_worker_finished(self):
        """Called when worker thread finishes (e.g. after stop)."""
        self._set_canvas_locked(False)
        self._ctrl._opt_panel.set_auto_active(False)

    def closeEvent(self, event):
        """Stop worker cleanly on window close; ask about unsaved changes."""
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()
        if not self._confirm_discard():
            event.ignore()
            return
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self):
        if self.state.has_collinear:
            self._status.showMessage("Cannot save: collinear points!", 5000)
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save coordinates", "", _SAVE_FILTERS,
        )
        if not path:
            return
        try:
            _write_points(path, self.state.get_coords(), self._coloring)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot save", str(e))
            return
        except Exception as e:
            self._status.showMessage(f"Save failed: {e}", 6000)
            return
        self._dirty = False
        self._status.showMessage(f"Saved: {path}", 4000)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        act_save = QAction("&Save…", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self._save)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = mb.addMenu("&View")
        act_fit = QAction("&Fit to window", self)
        act_fit.setShortcut(QKeySequence("Ctrl+0"))   # Cmd+0 on macOS
        act_fit.triggered.connect(self._view.fit_view)
        view_menu.addAction(act_fit)

        # Navigation shortcuts (only useful when multiple OTs loaded)
        act_prev = QAction("&Previous OT", self)
        act_prev.setShortcut(QKeySequence("Ctrl+Left"))
        act_prev.triggered.connect(self._on_prev)
        file_menu.addSeparator()
        file_menu.addAction(act_prev)

        act_next = QAction("&Next OT", self)
        act_next.setShortcut(QKeySequence("Ctrl+Right"))
        act_next.triggered.connect(self._on_next)
        file_menu.addAction(act_next)

        # Help menu
        help_menu = mb.addMenu("&Help")
        act_help = QAction("Features && Help…", self)
        act_help.setShortcut(QKeySequence("F1"))
        act_help.triggered.connect(self._show_help)
        help_menu.addAction(act_help)
        act_about = QAction("About pyotlib2", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("pyotlib2 — Features & Help")
        dlg.resize(620, 520)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(_HELP_HTML)
        layout.addWidget(text)
        dlg.exec()

    def _show_about(self):
        QMessageBox.about(
            self,
            "About pyotlib2",
            "<b>pyotlib2</b> — Python Order Type Library 2<br><br>"
            "Interactive exploration and computation of abstract order types, "
            "projective classes, and point set properties.",
        )


_HELP_HTML = """\
<h2>Editor Controls</h2>
<ul>
  <li><b>Left-drag</b> — move a point (snapped to integer coordinates)</li>
  <li><b>Right-click</b> on a point — select / deselect it</li>
  <li><b>Esc</b> — deselect</li>
  <li><b>Mouse wheel / + / -</b> — zoom in/out</li>
  <li><b>Middle-drag</b> — pan view</li>
  <li><b>Home / 0 / Ctrl+0</b> — fit to window</li>
  <li><b>Ctrl+S</b> — save</li>
  <li><b>Ctrl+Left/Right</b> — previous/next OT (multi-OT files)</li>
  <li><b>F1</b> — this help</li>
</ul>

<h2>Display Features</h2>
<ul>
  <li><b>Highlight</b> — convex hull, onion layers (nested convex layers) or none</li>
  <li><b>Hide incident lines</b> — fade the lines through the selected point</li>
  <li><b>Fix OT</b> — a drag that would change the order type (or create
      collinear points) is refused; the reference is the OT at the moment
      the checkbox is enabled</li>
  <li>Points involved in a collinear triple are drawn in red</li>
</ul>

<h2>Properties Panel</h2>
<p>Live-computed combinatorial properties (choose via "Select properties…";
values are recomputed only when the order type changes):</p>
<ul>
  <li>Hull size, crossings, onion layers, triangulations</li>
  <li>Convex 5-, 6-, 7-gons</li>
  <li>Empty 3- to 7-holes</li>
</ul>

<h2>Optimization Modes</h2>
<ul>
  <li><b>Coordinate Minimization</b> — reduce coordinate size, OT preserved</li>
  <li><b>Beautification</b> — gradient-descent beautification, OT preserved</li>
  <li><b>Property Min/Max</b> — random local search (walk-points) on the
      chosen property; the OT may change</li>
</ul>
<p>"1 Step" runs one iteration, "Auto" runs in the background until toggled off.</p>

<h2>File Formats</h2>
<p>Any file with coordinates can be opened (e.g. .asc, .psz, .json,
.b08/.pb08, .b16/.pb16); coordinates are kept exactly as loaded.
Save formats: .asc, .psz, .json, .b08 (coordinates 0..255),
.b16 (0..65535).  Point colors (.asc/.psz) are preserved.</p>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extend_line_to_rect(x0, y0, x1, y1, rect: QRectF):
    """Extend the line through (x0,y0)-(x1,y1) to the boundaries of rect."""
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0 and dy == 0:
        return QPointF(x0, y0), QPointF(x0, y0)

    L = rect.left()
    R = rect.right()
    T = rect.top()
    B = rect.bottom()

    pts = []
    # intersect with 4 edges
    if dx != 0:
        for bx in (L, R):
            t = (bx - x0) / dx
            iy = y0 + t * dy
            if T <= iy <= B:
                pts.append(QPointF(bx, iy))
    if dy != 0:
        for by in (T, B):
            t = (by - y0) / dy
            ix = x0 + t * dx
            if L <= ix <= R:
                pts.append(QPointF(ix, by))

    if len(pts) >= 2:
        return pts[0], pts[1]
    # fallback: return as segment
    return QPointF(x0, y0), QPointF(x1, y1)


def _load_order_types(filepath, *, n=None, fmt=None) -> list:
    """Load all OTs with realization from file. Returns list[SmallLambda]."""
    if filepath is None:
        return []
    path = Path(filepath)
    if not path.exists():
        return []
    try:
        from pyotlib2.io.readers import read_order_types
        kw = {k: v for k, v in [("n", n), ("fmt", fmt)] if v is not None}
        ots = list(read_order_types(path, **kw))
        return [sl for sl in ots if sl.realization]
    except Exception as e:
        import sys
        print(f"Warning: could not load {filepath}: {e}", file=sys.stderr)
        return []


def _load_points(filepath, *, n=None, fmt=None) -> list[tuple[int, int]]:
    """Load the first point set from file (coordinates unchanged).
    Falls back to the default points."""
    if filepath is not None:
        path = Path(filepath)
        if path.exists():
            try:
                from pyotlib2.io.readers import read_order_types
                kw = {}
                if n is not None:
                    kw["n"] = n
                if fmt is not None:
                    kw["fmt"] = fmt
                ots = list(read_order_types(path, **kw))
                if ots and ots[0].realization:
                    return [(int(x), int(y)) for x, y in ots[0].realization]
            except Exception as e:
                import sys
                print(f"Warning: could not load {filepath}: {e}", file=sys.stderr)
    return _default_points(6)


_SAVE_FILTERS = (
    "ASCII point set (*.asc);;PoSeZo point set (*.psz);;JSON (*.json);;"
    "Point binary 8-bit (*.b08);;Point binary 16-bit (*.b16)"
)

# point binary formats → bits per coordinate (unsigned)
_BINARY_BITS = {"pb04": 4, "b08": 8, "pb08": 8, "b16": 16, "pb16": 16, "pb32": 32}


def _write_points(path, coords: list[tuple[int, int]], coloring=None) -> None:
    """Write the model coordinates to *path* (format from the extension,
    default .asc).  Raises ValueError with a readable message if the
    coordinates do not fit a binary format."""
    path = Path(path)
    fmt = path.suffix.lstrip(".").lower() or "asc"
    bits = _BINARY_BITS.get(fmt)
    if bits is not None:
        lo = min(v for p in coords for v in p)
        hi = max(v for p in coords for v in p)
        top = 2 ** bits - 1
        if lo < 0 or hi > top:
            wider = ".b16 or .asc" if bits < 16 else ".asc"
            raise ValueError(
                f"Coordinates range from {lo} to {hi}, but .{fmt} only stores "
                f"0..{top}.  Save as {wider} instead, or run "
                f"Coordinate Minimization first."
            )
    from pyotlib2.core.point_set import PointSet
    from pyotlib2.io.writers import write_order_types
    sl = PointSet(len(coords), list(coords), coloring=coloring).to_small_lambda(lazy=False)
    write_order_types([sl], path, fmt=fmt)


def _default_points(n: int) -> list[tuple[int, int]]:
    """n points on a circle in the grid."""
    cx = cy = GRID_MAX // 2
    r = int(GRID_MAX * 0.38)
    return [
        (int(cx + r * math.cos(2 * math.pi * i / n - math.pi / 2)),
         int(cy + r * math.sin(2 * math.pi * i / n - math.pi / 2)))
        for i in range(n)
    ]
