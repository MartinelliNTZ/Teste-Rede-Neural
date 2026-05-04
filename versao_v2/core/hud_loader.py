from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .dark_charcoal_style import DarkCharcoalStyle


class HudCircularRingsLoader(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.progress = 0.0
        self.phase = 0
        self.message = "Processando..."
        self.rings = [
            {"radius": 82, "width": 6, "speed": 2.0, "angle": 0, "segments": 10, "seg_span": 16, "seg_gap": 18,
             "color": QColor(212, 168, 83, 240)},
            {"radius": 62, "width": 5, "speed": -3.2, "angle": 150, "segments": 2, "seg_span": 120, "seg_gap": 60,
             "color": QColor(33, 150, 243, 220)},
            {"radius": 45, "width": 4, "speed": 5.4, "angle": 45, "segments": 14, "seg_span": 10, "seg_gap": 12,
             "color": QColor(232, 200, 120, 220)},
        ]
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)
        self.hide()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(16)

    def set_progress(self, value: float, message: str = ""):
        self.progress = max(0.0, min(100.0, float(value)))
        if message:
            self.message = message
        self.update()

    def show_loader(self):
        self.raise_()
        self.show()
        self.update()

    def hide_loader(self):
        self.hide()

    def _animate(self):
        if not self.isVisible():
            return
        self.phase = (self.phase + 1) % 360
        for ring in self.rings:
            ring["angle"] = (ring["angle"] + ring["speed"]) % 360
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0, 130))

        cx = self.width() // 2
        cy = self.height() // 2 - 20
        panel = QRectF(cx - 150, cy - 130, 300, 300)
        self._draw_panel(p, panel)

        center = panel.center().toPoint()
        for ring in self.rings:
            self._draw_ring(p, center, ring)

        p.setPen(QColor(DarkCharcoalStyle.ACCENT_GOLD))
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(QRectF(cx - 120, cy + 110, 240, 24), Qt.AlignmentFlag.AlignCenter, self.message)

        p.setPen(QColor(DarkCharcoalStyle.ACCENT_GOLD))
        p.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        p.drawText(QRectF(cx - 60, cy - 0, 120, 38), Qt.AlignmentFlag.AlignCenter, f"{self.progress:.2f}%")

    def _draw_panel(self, painter: QPainter, rect: QRectF):
        grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad.setColorAt(0.0, QColor(37, 37, 38, 245))
        grad.setColorAt(1.0, QColor(18, 18, 18, 245))
        path = QPainterPath()
        path.addRoundedRect(rect, 22, 22)
        painter.setPen(QPen(QColor(62, 62, 66, 220), 1))
        painter.setBrush(grad)
        painter.drawPath(path)

    def _draw_ring(self, painter: QPainter, center, ring):
        painter.setBrush(Qt.BrushStyle.NoBrush)
        trail_pen = QPen(QColor(45, 45, 48, 200), ring["width"])
        painter.setPen(trail_pen)
        painter.drawEllipse(center, ring["radius"], ring["radius"])

        for i in range(ring["segments"]):
            start_deg = ring["angle"] + i * (ring["seg_span"] + ring["seg_gap"])
            start = int((90 - start_deg) * 16)
            span = -int(ring["seg_span"] * 16)
            pulse = ((self.phase + i * 17) % 120) / 120.0
            alpha = int(70 + 170 * pulse)
            c = ring["color"]
            color = QColor(c.red(), c.green(), c.blue(), alpha)
            painter.setPen(QPen(color, ring["width"], Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawArc(center.x() - ring["radius"], center.y() - ring["radius"], ring["radius"] * 2, ring["radius"] * 2, start, span)
