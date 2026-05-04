from qgis.PyQt.QtWidgets import QWidget
from qgis.PyQt.QtCore import Qt, QTimer, QRectF
from qgis.PyQt.QtGui import (
    QPainter,
    QColor,
    QPen,
    QFont,
    QLinearGradient,
    QPainterPath,
)


class HudCircularRingsLoader(QWidget):
    def __init__(self):
        super().__init__()

        self.progress = 0
        self.phase = 0

        self.rings = [
            {
                "radius": 88,
                "width": 6,
                "speed": 2.2,
                "angle": 0,
                "segments": 12,
                "seg_span": 14,
                "seg_gap": 16,
                "color": QColor(0, 246, 255, 245),
            },
            {
                "radius": 68,
                "width": 5,
                "speed": -3.4,
                "angle": 160,
                "segments": 1,
                "seg_span": 180,
                "seg_gap": 180,
                "color": QColor(0, 178, 255, 235),
            },
            {
                "radius": 50,
                "width": 4,
                "speed": 5.8,
                "angle": 45,
                "segments": 20,
                "seg_span": 8,
                "seg_gap": 10,
                "color": QColor(0, 255, 200, 230),
            },
            {
                "radius": 34,
                "width": 3,
                "speed": -7.0,
                "angle": 320,
                "segments": 4,
                "seg_span": 22,
                "seg_gap": 68,
                "color": QColor(30, 255, 255, 220),
            },
        ]

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(240, 240)

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def setProgress(self, val):
        self.progress = max(0, min(100, int(val)))
        self.update()

    def animate(self):
        self.phase = (self.phase + 1) % 360
        for ring in self.rings:
            ring["angle"] = (ring["angle"] + ring["speed"]) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        panel_rect = self.rect().adjusted(14, 14, -14, -14)
        self._draw_panel(painter, panel_rect)
        center = panel_rect.center()

        for ring in self.rings:
            trail_pen = QPen(QColor(17, 28, 42, 190), ring["width"])
            painter.setPen(trail_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, ring["radius"], ring["radius"])

        for ring in self.rings:
            self._draw_segment_ring(painter, center, ring)

        self._draw_core(painter, center)

    def _draw_panel(self, painter, rect):
        radius = 24
        rectf = QRectF(rect)

        # Single soft shadow/glow.
        sombra = 1
        glow_rect = rect.adjusted(-sombra, -sombra, sombra, sombra)
        painter.setPen(QPen(QColor(0, 0, 0, 60), 9))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(glow_rect, radius + 3, radius + 3)

        # Fundo em degradê preto -> cinza escuro.
        bg_gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        bg_gradient.setColorAt(0.0, QColor(33, 36, 40, 245))
        bg_gradient.setColorAt(0.55, QColor(17, 19, 23, 245))
        bg_gradient.setColorAt(1.0, QColor(7, 8, 10, 245))

        path = QPainterPath()
        path.addRoundedRect(rectf, radius, radius)
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg_gradient)
        painter.drawPath(path)

        # Realce interno removido para evitar qualquer aparência de borda.

    def _draw_segment_ring(self, painter, center, ring):
        radius = ring["radius"]
        count = ring["segments"]
        seg_span = ring["seg_span"]
        seg_gap = ring["seg_gap"]
        base_angle = ring["angle"]
        base_color = ring["color"]

        for idx in range(count):
            start_deg = base_angle + idx * (seg_span + seg_gap)
            start = int((90 - start_deg) * 16)
            span = -int(seg_span * 16)

            pulse = ((self.phase + idx * 19) % 120) / 120.0
            alpha = int(70 + 185 * pulse)
            color = QColor(base_color.red(), base_color.green(), base_color.blue(), alpha)

            pen = QPen(color, ring["width"], Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.drawArc(
                center.x() - radius,
                center.y() - radius,
                radius * 2,
                radius * 2,
                start,
                span,
            )

    def _draw_core(self, painter, center):
        core = 22
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(5, 18, 30, 220))
        painter.drawEllipse(center, core, core)

        painter.setPen(QColor(0, 250, 255))
        painter.setFont(QFont("Consolas", 18, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self.progress}%")
