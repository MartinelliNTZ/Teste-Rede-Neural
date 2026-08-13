# -*- coding: utf-8 -*-
"""Tema visual Dark Premium — constantes de cores e stylesheet global."""


class Colors:
    """Paleta de cores do tema Dark Premium."""

    # Fundos
    BACKGROUND      = "#0d0d0f"
    PANEL           = "#16161a"
    BLACK_SOFT      = "#1a1a1e"

    # Bordas
    BORDER          = "#2a2a30"
    BORDER_LIGHT    = "#3a3a42"

    # Destaque (cor premium)
    GOLD            = "#d4af37"
    GOLD_LIGHT      = "#f0d98c"
    GOLD_DIM        = "#8a6d1f"

    # Status
    SUCCESS         = "#4caf50"
    WARNING         = "#ff9800"
    ERROR           = "#f44336"

    # Texto
    TEXT            = "#e0e0e0"
    TEXT_DIM        = "#9e9e9e"


class Styles:
    """Stylesheet global do sistema."""

    @staticmethod
    def get_stylesheet() -> str:
        return f"""
        QMainWindow, QWidget#centralWidget {{
            background-color: {Colors.BACKGROUND};
            color: {Colors.TEXT};
            font-family: 'Segoe UI', sans-serif;
            font-size: 13px;
        }}

        QGroupBox {{
            background-color: {Colors.PANEL};
            border: 1px solid {Colors.BORDER};
            border-radius: 6px;
            margin-top: 8px;
            padding: 2px;
            font-weight: 600;
            color: {Colors.GOLD};
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 2px;
        }}

        QLabel {{
            color: {Colors.TEXT};
        }}

        QLabel#titleLabel {{
            font-size: 22px;
            font-weight: 800;
            color: {Colors.GOLD};
        }}

        QLabel#subtitleLabel {{
            font-size: 13px;
            color: {Colors.TEXT_DIM};
        }}

        QLabel#statusLabel {{
            font-size: 13px;
            font-weight: 700;
        }}

        QPushButton {{
            background-color: {Colors.PANEL};
            color: {Colors.TEXT};
            border: 1px solid {Colors.BORDER};
            border-radius: 4px;
            padding: 2px 8px;
            font-weight: 600;
        }}

        QPushButton:hover {{
            border-color: {Colors.GOLD};
            color: {Colors.GOLD_LIGHT};
        }}

        QPushButton:pressed {{
            background-color: {Colors.BLACK_SOFT};
        }}

        QPushButton:disabled {{
            color: {Colors.TEXT_DIM};
            border-color: {Colors.BORDER};
        }}

        QPushButton#primaryButton {{
            background-color: {Colors.GOLD};
            color: #000;
            border: none;
            border-radius: 4px;
            font-weight: 700;
            font-size: 14px;
            padding: 2px 12px;
        }}

        QPushButton#primaryButton:hover {{
            background-color: {Colors.GOLD_LIGHT};
        }}

        QPushButton#primaryButton:disabled {{
            background-color: {Colors.GOLD_DIM};
            color: #555;
        }}

        QPushButton#dangerButton {{
            background-color: transparent;
            color: {Colors.ERROR};
            border: 1px solid {Colors.ERROR};
            border-radius: 4px;
        }}

        QPushButton#dangerButton:hover {{
            background-color: {Colors.ERROR};
            color: #fff;
        }}

        QPushButton#smallButton {{
            padding: 2px 6px;
            font-size: 12px;
        }}

        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {Colors.BLACK_SOFT};
            color: {Colors.TEXT};
            border: 1px solid {Colors.BORDER};
            border-radius: 3px;
            padding: 2px 4px;
            min-height: 20px;
        }}

        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border-color: {Colors.GOLD};
        }}

        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}

        QComboBox QAbstractItemView {{
            background-color: {Colors.PANEL};
            color: {Colors.TEXT};
            border: 1px solid {Colors.BORDER};
            selection-background-color: {Colors.GOLD_DIM};
        }}

        QListWidget {{
            background-color: {Colors.BLACK_SOFT};
            color: {Colors.TEXT};
            border: 1px solid {Colors.BORDER};
            border-radius: 3px;
            padding: 2px;
        }}

        QListWidget::item {{
            padding: 2px 4px;
            border-radius: 2px;
        }}

        QListWidget::item:selected {{
            background-color: {Colors.GOLD_DIM};
            color: #fff;
        }}

        QListWidget::item:hover {{
            background-color: {Colors.PANEL};
        }}

        QPlainTextEdit#consoleWidget {{
            background-color: {Colors.BLACK_SOFT};
            border: 1px solid {Colors.BORDER};
            border-radius: 4px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            color: {Colors.TEXT};
            padding: 2px;
        }}

        QProgressBar {{
            background-color: {Colors.BLACK_SOFT};
            border: 1px solid {Colors.BORDER};
            border-radius: 4px;
            text-align: center;
            color: {Colors.TEXT};
            font-weight: 600;
            min-height: 20px;
        }}

        QProgressBar::chunk {{
            background-color: {Colors.GOLD};
            border-radius: 3px;
        }}

        QFrame#headerBar {{
            background-color: {Colors.PANEL};
            border-bottom: 2px solid {Colors.GOLD_DIM};
            border-radius: 0px;
        }}

        QStatusBar {{
            background-color: {Colors.PANEL};
            color: {Colors.TEXT_DIM};
            border-top: 1px solid {Colors.BORDER};
        }}

        QStatusBar::item {{
            border: none;
        }}

        QDialog {{
            background-color: {Colors.BACKGROUND};
            color: {Colors.TEXT};
        }}

        QScrollBar:vertical {{
            background: {Colors.BLACK_SOFT};
            width: 10px;
            border-radius: 5px;
        }}

        QScrollBar::handle:vertical {{
            background: {Colors.BORDER_LIGHT};
            border-radius: 5px;
            min-height: 20px;
        }}

        QScrollBar::handle:vertical:hover {{
            background: {Colors.GOLD_DIM};
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        QScrollBar:horizontal {{
            background: {Colors.BLACK_SOFT};
            height: 10px;
            border-radius: 5px;
        }}

        QScrollBar::handle:horizontal {{
            background: {Colors.BORDER_LIGHT};
            border-radius: 5px;
            min-width: 20px;
        }}

        QScrollBar::handle:horizontal:hover {{
            background: {Colors.GOLD_DIM};
        }}

        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}

        QCheckBox {{
            color: {Colors.TEXT};
            spacing: 6px;
        }}

        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {Colors.BORDER};
            border-radius: 3px;
            background-color: {Colors.BLACK_SOFT};
        }}

        QCheckBox::indicator:checked {{
            background-color: {Colors.GOLD};
            border-color: {Colors.GOLD};
        }}

        QToolTip {{
            background-color: {Colors.PANEL};
            color: {Colors.TEXT};
            border: 1px solid {Colors.GOLD_DIM};
            padding: 2px 4px;
            border-radius: 3px;
        }}
        """