# -*- coding: utf-8 -*-
"""
Estilos Dark Charcoal Premium para a UI do Classificador Raster Neural v6
==========================================================================
Folha de estilos QSS centralizada para a aplicacao PySide6.
"""

# ═══════════════════════════════════════════════════════════════════════════
# CLASSE DE ESTILOS DARK CHARCOAL PREMIUM
# ═══════════════════════════════════════════════════════════════════════════

class DarkCharcoalStyle:
    """Centraliza toda a paleta e folha de estilos QSS da aplicacao."""

    DARK_BG        = "#121212"
    PANEL_BG       = "#1E1E1E"
    CARD_BG        = "#252526"
    INPUT_BG       = "#2D2D30"
    BORDER         = "#3E3E42"
    TEXT_PRIMARY   = "#EAEAEA"
    TEXT_SECONDARY = "#A0A0A0"
    ACCENT_GOLD    = "#D4A853"
    ACCENT_HOVER   = "#E8C878"
    SUCCESS        = "#4CAF50"
    WARNING        = "#FF9800"
    DANGER         = "#F44336"
    INFO           = "#2196F3"

    @classmethod
    def stylesheet(cls) -> str:
        c = cls
        return f"""
QMainWindow {{
    background-color: {c.DARK_BG};
}}

QWidget {{
    background-color: {c.DARK_BG};
    color: {c.TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
}}

QScrollArea {{
    border: none;
    background-color: {c.DARK_BG};
}}
QScrollBar:vertical {{
    background: {c.DARK_BG};
    width: 10px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {c.BORDER};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {c.ACCENT_GOLD};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QGroupBox {{
    background-color: {c.PANEL_BG};
    border: 1px solid {c.BORDER};
    border-radius: 12px;
    margin-top: 6px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
    font-size: 13px;
    color: {c.TEXT_PRIMARY};
}}
QGroupBox::title {{
    subcontrol-origin: padding;
    subcontrol-position: top left;
    left: 2px;
    top: 2px;
    padding: 0 2px;
    color: {c.ACCENT_GOLD};
    font-weight: 700;
    font-size: 13px;
}}

QLabel {{
    background-color: transparent;
    color: {c.TEXT_PRIMARY};
}}
QLabel#header_title {{
    font-size: 24px;
    font-weight: 700;
    color: {c.TEXT_PRIMARY};
}}
QLabel#header_subtitle {{
    font-size: 12px;
    color: {c.TEXT_SECONDARY};
}}
QLabel#section_badge {{
    background-color: {c.ACCENT_GOLD};
    color: {c.DARK_BG};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
}}

QWidget#title_bar {{
    background-color: #202124;
    border-bottom: 1px solid {c.BORDER};
}}
QLabel#window_title {{
    color: {c.TEXT_PRIMARY};
    font-size: 12px;
    font-weight: 600;
}}
QPushButton#title_btn, QPushButton#title_btn_close {{
    background-color: transparent;
    color: {c.TEXT_PRIMARY};
    border: none;
    border-radius: 4px;
    min-width: 30px;
    max-width: 30px;
    min-height: 24px;
    max-height: 24px;
    font-size: 12px;
    font-weight: 600;
}}
QPushButton#title_btn:hover {{
    background-color: {c.CARD_BG};
}}
QPushButton#title_btn_close:hover {{
    background-color: {c.DANGER};
    color: white;
}}

QLineEdit {{
    background-color: {c.INPUT_BG};
    border: 1px solid {c.BORDER};
    border-radius: 6px;
    padding: 4px 7px;
    color: {c.TEXT_PRIMARY};
    selection-background-color: {c.ACCENT_GOLD};
    selection-color: {c.DARK_BG};
}}
QLineEdit:focus {{
    border: 1px solid {c.ACCENT_GOLD};
}}

QSpinBox, QDoubleSpinBox {{
    background-color: {c.INPUT_BG};
    border: 1px solid {c.BORDER};
    border-radius: 6px;
    padding: 3px 7px;
    color: {c.TEXT_PRIMARY};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {c.ACCENT_GOLD};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 18px;
    background: {c.CARD_BG};
    border-radius: 3px;
    margin: 1px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {c.ACCENT_GOLD};
}}

QComboBox {{
    background-color: {c.INPUT_BG};
    border: 1px solid {c.BORDER};
    border-radius: 6px;
    padding: 3px 7px;
    color: {c.TEXT_PRIMARY};
    min-width: 100px;
}}
QComboBox:focus {{
    border: 1px solid {c.ACCENT_GOLD};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid {c.BORDER};
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {c.TEXT_SECONDARY};
    width: 0px;
    height: 0px;
}}
QComboBox QAbstractItemView {{
    background-color: {c.CARD_BG};
    border: 1px solid {c.BORDER};
    color: {c.TEXT_PRIMARY};
    selection-background-color: {c.ACCENT_GOLD};
    selection-color: {c.DARK_BG};
}}

QCheckBox {{
    spacing: 8px;
    background-color: transparent;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {c.BORDER};
    background-color: {c.INPUT_BG};
}}
QCheckBox::indicator:checked {{
    background-color: {c.ACCENT_GOLD};
    border: 1px solid {c.ACCENT_GOLD};
}}
QCheckBox::indicator:hover {{
    border: 1px solid {c.ACCENT_GOLD};
}}

QTableWidget {{
    background-color: {c.INPUT_BG};
    border: 1px solid {c.BORDER};
    border-radius: 8px;
    gridline-color: {c.BORDER};
    color: {c.TEXT_PRIMARY};
}}
QTableWidget::item {{
    padding: 2px;
    border-bottom: 1px solid {c.BORDER};
}}
QTableWidget::item:selected {{
    background-color: {c.ACCENT_GOLD};
    color: {c.DARK_BG};
}}
QHeaderView::section {{
    background-color: {c.CARD_BG};
    color: {c.TEXT_SECONDARY};
    padding: 2px;
    border: none;
    border-bottom: 2px solid {c.ACCENT_GOLD};
    font-weight: 600;
    font-size: 12px;
}}
QTableWidget QPushButton {{
    background-color: {c.DANGER};
    color: white;
    border-radius: 4px;
    padding: 2px 2px;
    font-size: 11px;
    min-width: 60px;
}}

QTextEdit {{
    background-color: {c.INPUT_BG};
    border: 1px solid {c.BORDER};
    border-radius: 8px;
    color: {c.TEXT_SECONDARY};
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    padding: 10px;
}}

QProgressBar {{
    border: none;
    border-radius: 6px;
    background-color: {c.INPUT_BG};
    text-align: center;
    color: {c.TEXT_PRIMARY};
    font-weight: 600;
    height: 22px;
}}
QProgressBar::chunk {{
    border-radius: 6px;
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {c.ACCENT_GOLD},
        stop:1 {c.ACCENT_HOVER}
    );
}}

QPushButton#btn_primary {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {c.ACCENT_GOLD},
        stop:1 #C49A4A
    );
    color: {c.DARK_BG};
    border: none;
    border-radius: 6px;
    padding: 2px 2px;
    font-weight: 700;
    font-size: 13px;
}}
QPushButton#btn_primary:hover {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {c.ACCENT_HOVER},
        stop:1 #D4A853
    );
}}
QPushButton#btn_primary:pressed {{
    background: #B08A3E;
}}

QPushButton#btn_secondary {{
    background-color: {c.CARD_BG};
    color: {c.TEXT_PRIMARY};
    border: 1px solid {c.BORDER};
    border-radius: 5px;
    padding: 2px 2px;
    font-weight: 600;
    font-size: 12px;
}}
QPushButton#btn_secondary:hover {{
    background-color: {c.INPUT_BG};
    border: 1px solid {c.ACCENT_GOLD};
    color: {c.ACCENT_GOLD};
}}

QPushButton#btn_action {{
    background-color: {c.INFO};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 2px 2px;
    font-weight: 600;
    font-size: 11px;
}}
QPushButton#btn_action:hover {{
    background-color: #42A5F5;
}}
QPushButton#btn_danger {{
    background-color: {c.DANGER};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 2px 2px;
    font-weight: 600;
    font-size: 11px;
}}
QPushButton#btn_danger:hover {{
    background-color: #EF5350;
}}

QFrame#separator {{
    background-color: {c.BORDER};
    max-height: 1px;
}}
QFrame#separator_v {{
    background-color: {c.BORDER};
    max-width: 1px;
}}
"""
