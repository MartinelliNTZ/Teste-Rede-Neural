# -*- coding: utf-8 -*-
"""
UI Profissional Dark Charcoal — Classificador Raster Neural v6
===============================================================
Interface premium em PySide6 para o pipeline main6_multcore.py.
Apenas UI (frontend); a lógica de execução será integrada posteriormente.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QGroupBox, QSplitter, QTextEdit, QProgressBar, QFrame, QSizePolicy,
    QScrollArea, QMessageBox, QDialog, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QLinearGradient, QBrush


# ═══════════════════════════════════════════════════════════════════════════
# PALETA DARK CHARCOAL PREMIUM
# ═══════════════════════════════════════════════════════════════════════════

DARK_BG       = "#121212"   # Fundo geral
PANEL_BG      = "#1E1E1E"   # Painéis
CARD_BG       = "#252526"   # Cards internos
INPUT_BG      = "#2D2D30"   # Inputs
BORDER        = "#3E3E42"   # Bordas sutis
TEXT_PRIMARY  = "#EAEAEA"   # Texto principal
TEXT_SECONDARY= "#A0A0A0"   # Texto secundário
ACCENT_GOLD   = "#D4A853"   # Dourado âmbar premium
ACCENT_HOVER  = "#E8C878"   # Dourado claro hover
SUCCESS       = "#4CAF50"   # Verde sucesso
WARNING       = "#FF9800"   # Laranja alerta
DANGER        = "#F44336"   # Vermelho erro
INFO          = "#2196F3"   # Azul info


# ═══════════════════════════════════════════════════════════════════════════
# ESTILOS QSS
# ═══════════════════════════════════════════════════════════════════════════

STYLE_SHEET = f"""
QMainWindow {{
    background-color: {DARK_BG};
}}

QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
}}

/* ── Scroll Area ── */
QScrollArea {{
    border: none;
    background-color: {DARK_BG};
}}
QScrollBar:vertical {{
    background: {DARK_BG};
    width: 10px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT_GOLD};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── GroupBox Premium ── */
QGroupBox {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 10px;
    padding-bottom: 10px;
    padding-left: 14px;
    padding-right: 14px;
    font-weight: 600;
    font-size: 13px;
    color: {TEXT_PRIMARY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: {ACCENT_GOLD};
    font-weight: 700;
    font-size: 13px;
}}

/* ── Labels ── */
QLabel {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
}}
QLabel#header_title {{
    font-size: 24px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}
QLabel#header_subtitle {{
    font-size: 12px;
    color: {TEXT_SECONDARY};
}}
QLabel#section_badge {{
    background-color: {ACCENT_GOLD};
    color: {DARK_BG};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
}}

/* ── LineEdit ── */
QLineEdit {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_GOLD};
    selection-color: {DARK_BG};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT_GOLD};
}}

/* ── SpinBoxes & DoubleSpinBoxes ── */
QSpinBox, QDoubleSpinBox {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    color: {TEXT_PRIMARY};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT_GOLD};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 18px;
    background: {CARD_BG};
    border-radius: 3px;
    margin: 1px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {ACCENT_GOLD};
}}

/* ── ComboBox ── */
QComboBox {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 10px;
    color: {TEXT_PRIMARY};
    min-width: 100px;
}}
QComboBox:focus {{
    border: 1px solid {ACCENT_GOLD};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid {BORDER};
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SECONDARY};
    width: 0px;
    height: 0px;
}}
QComboBox QAbstractItemView {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_GOLD};
    selection-color: {DARK_BG};
}}

/* ── CheckBox ── */
QCheckBox {{
    spacing: 8px;
    background-color: transparent;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {BORDER};
    background-color: {INPUT_BG};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT_GOLD};
    border: 1px solid {ACCENT_GOLD};
    image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiMxMjEyMTIiIHN0cm9rZS13aWR0aD0iMyI+PHBvbHlsaW5lIHBvaW50cz0iMjAgNiA5IDE3IDQgMTIiLz48L3N2Zz4=);
}}
QCheckBox::indicator:hover {{
    border: 1px solid {ACCENT_GOLD};
}}

/* ── TableWidget ── */
QTableWidget {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    gridline-color: {BORDER};
    color: {TEXT_PRIMARY};
}}
QTableWidget::item {{
    padding: 6px;
    border-bottom: 1px solid {BORDER};
}}
QTableWidget::item:selected {{
    background-color: {ACCENT_GOLD};
    color: {DARK_BG};
}}
QHeaderView::section {{
    background-color: {CARD_BG};
    color: {TEXT_SECONDARY};
    padding: 8px;
    border: none;
    border-bottom: 2px solid {ACCENT_GOLD};
    font-weight: 600;
    font-size: 12px;
}}
QTableWidget QPushButton {{
    background-color: {DANGER};
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    min-width: 60px;
}}

/* ── TextEdit (Log) ── */
QTextEdit {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    color: {TEXT_SECONDARY};
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    padding: 8px;
}}

/* ── ProgressBar ── */
QProgressBar {{
    border: none;
    border-radius: 6px;
    background-color: {INPUT_BG};
    text-align: center;
    color: {TEXT_PRIMARY};
    font-weight: 600;
    height: 22px;
}}
QProgressBar::chunk {{
    border-radius: 6px;
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT_GOLD},
        stop:1 {ACCENT_HOVER}
    );
}}

/* ── Botões Principais ── */
QPushButton#btn_primary {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT_GOLD},
        stop:1 #C49A4A
    );
    color: {DARK_BG};
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 700;
    font-size: 14px;
}}
QPushButton#btn_primary:hover {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT_HOVER},
        stop:1 #D4A853
    );
}}
QPushButton#btn_primary:pressed {{
    background: #B08A3E;
}}

/* ── Botões Secundários ── */
QPushButton#btn_secondary {{
    background-color: {CARD_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
}}
QPushButton#btn_secondary:hover {{
    background-color: {INPUT_BG};
    border: 1px solid {ACCENT_GOLD};
    color: {ACCENT_GOLD};
}}

/* ── Botões de Ação Pequenos ── */
QPushButton#btn_action {{
    background-color: {INFO};
    color: white;
    border: none;
    border-radius: 5px;
    padding: 5px 14px;
    font-weight: 600;
    font-size: 12px;
}}
QPushButton#btn_action:hover {{
    background-color: #42A5F5;
}}
QPushButton#btn_danger {{
    background-color: {DANGER};
    color: white;
    border: none;
    border-radius: 5px;
    padding: 5px 14px;
    font-weight: 600;
    font-size: 12px;
}}
QPushButton#btn_danger:hover {{
    background-color: #EF5350;
}}

/* ── Separadores ── */
QFrame#separator {{
    background-color: {BORDER};
    max-height: 1px;
}}
QFrame#separator_v {{
    background-color: {BORDER};
    max-width: 1px;
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# WIDGETS AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

class Badge(QLabel):
    """Badge estilizado tipo tag premium."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("section_badge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class Separator(QFrame):
    """Linha separadora horizontal sutil."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("separator")
        self.setFrameShape(QFrame.Shape.HLine)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(1)


class PathBrowseRow(QWidget):
    """Linha com label, campo de texto e botão browse."""
    def __init__(self, label_text: str, default_path: str = "", file_mode=True,
                 file_filter="Todos (*.*)", parent=None):
        super().__init__(parent)
        self.file_mode = file_mode
        self.file_filter = file_filter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.label = QLabel(label_text)
        self.label.setFixedWidth(140)
        self.label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: 500;")

        self.edit = QLineEdit(default_path)
        self.edit.setPlaceholderText("Caminho do arquivo...")

        self.btn = QPushButton("⋯")
        self.btn.setObjectName("btn_secondary")
        self.btn.setFixedWidth(36)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.clicked.connect(self._browse)

        layout.addWidget(self.label)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.btn)

    def _browse(self):
        if self.file_mode:
            path, _ = QFileDialog.getOpenFileName(
                self, "Selecionar arquivo", "", self.file_filter
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "Selecionar pasta")
        if path:
            self.edit.setText(path)

    def path(self) -> str:
        return self.edit.text().strip()


# ═══════════════════════════════════════════════════════════════════════════
# JANELA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classificador Raster Neural — v6 Premium")
        self.setMinimumSize(1280, 860)
        self.resize(1440, 900)
        self._build_ui()

    def _build_ui(self):
        # ── Widget central com scroll ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        central = QWidget()
        scroll.setWidget(central)
        self.setCentralWidget(scroll)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(20)

        # ═══════════════════════════════════════════════════════════════════
        # HEADER PREMIUM
        # ═══════════════════════════════════════════════════════════════════
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(16)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)

        self.lbl_title = QLabel("Classificador Raster Neural")
        self.lbl_title.setObjectName("header_title")

        self.lbl_subtitle = QLabel(
            "Pipeline de classificação supervisionada com redes neurais profundas "
            "— extração espectral, treinamento multicore e exportação GeoTIFF."
        )
        self.lbl_subtitle.setObjectName("header_subtitle")
        self.lbl_subtitle.setWordWrap(True)

        title_col.addWidget(self.lbl_title)
        title_col.addWidget(self.lbl_subtitle)

        header_layout.addLayout(title_col, 1)

        # Status badge
        self.badge_status = Badge("PRONTA")
        self.badge_status.setStyleSheet(f"""
            QLabel {{
                background-color: {SUCCESS};
                color: {DARK_BG};
                border-radius: 6px;
                padding: 4px 14px;
                font-weight: 700;
                font-size: 11px;
            }}
        """)
        header_layout.addWidget(self.badge_status, alignment=Qt.AlignmentFlag.AlignVCenter)

        main_layout.addWidget(header)
        main_layout.addWidget(Separator())

        # ═══════════════════════════════════════════════════════════════════
        # CORPO — Splitter esquerda (config) / direita (log + preview)
        # ═══════════════════════════════════════════════════════════════════
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Painel Esquerdo: Configurações ──
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(18)

        # ━━ Grupo: Imagens ━━
        grp_imagens = QGroupBox("Imagens & Saída")
        lay_img = QVBoxLayout(grp_imagens)
        lay_img.setSpacing(10)

        self.row_img_treino   = PathBrowseRow("Imagem Treino",   "dados/imagemTreino.tif",
                                               file_filter="GeoTIFF (*.tif *.tiff)")
        self.row_img_classif  = PathBrowseRow("Imagem Classif.", "dados/imagemCompleta.tif",
                                               file_filter="GeoTIFF (*.tif *.tiff)")
        self.row_img_saida    = PathBrowseRow("Saída GeoTIFF",   "resultado/mapa_classificado_ui.tif",
                                               file_filter="GeoTIFF (*.tif *.tiff)")

        lay_img.addWidget(self.row_img_treino)
        lay_img.addWidget(self.row_img_classif)
        lay_img.addWidget(self.row_img_saida)
        left_layout.addWidget(grp_imagens)

        # ━━ Grupo: Amostras (Shapefiles) ━━
        grp_amostras = QGroupBox("Amostras — Shapefiles por Classe")
        lay_amostras = QVBoxLayout(grp_amostras)
        lay_amostras.setSpacing(10)

        self.table_shp = QTableWidget(0, 3)
        self.table_shp.setHorizontalHeaderLabels(["Caminho do Shapefile", "ID Classe", "Ação"])
        self.table_shp.horizontalHeader().setStretchLastSection(False)
        self.table_shp.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_shp.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table_shp.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table_shp.setColumnWidth(1, 80)
        self.table_shp.setColumnWidth(2, 90)
        self.table_shp.setMinimumHeight(140)

        # Dados padrão do config
        default_shps = [
            ("dados/solo.shp", 0),
            ("dados/floresta.shp", 1),
            ("dados/palhada.shp", 2),
            ("dados/daninhas.shp", 3),
        ]
        for p, c in default_shps:
            self._add_shp_row(p, c)

        btn_add_shp = QPushButton("+ Adicionar Shapefile")
        btn_add_shp.setObjectName("btn_action")
        btn_add_shp.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_add_shp.clicked.connect(self._on_add_shp)

        lay_amostras.addWidget(self.table_shp)
        lay_amostras.addWidget(btn_add_shp, alignment=Qt.AlignmentFlag.AlignLeft)
        left_layout.addWidget(grp_amostras)

        # ━━ Grupo: Arquitetura da Rede ━━
        grp_rede = QGroupBox("Arquitetura da Rede Neural")
        lay_rede = QVBoxLayout(grp_rede)
        lay_rede.setSpacing(10)

        # Camadas ocultas
        row_camadas = QHBoxLayout()
        row_camadas.setSpacing(12)
        row_camadas.addWidget(QLabel("Camadas Ocultas:"))
        self.edit_camadas = QLineEdit("128, 64, 32")
        self.edit_camadas.setPlaceholderText("ex: 256, 128, 64")
        row_camadas.addWidget(self.edit_camadas, 1)

        # Ativação
        row_camadas.addWidget(QLabel("Ativação:"))
        self.combo_ativacao = QComboBox()
        self.combo_ativacao.addItems(["relu", "elu", "tanh", "sigmoid", "linear"])
        self.combo_ativacao.setCurrentText("relu")
        row_camadas.addWidget(self.combo_ativacao)

        lay_rede.addLayout(row_camadas)

        # Dropout
        row_dropout = QHBoxLayout()
        row_dropout.setSpacing(12)
        row_dropout.addWidget(QLabel("Dropout:"))
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 0.9)
        self.spin_dropout.setSingleStep(0.05)
        self.spin_dropout.setDecimals(2)
        self.spin_dropout.setValue(0.1)
        self.spin_dropout.setSuffix("  (0 = desativado)")
        row_dropout.addWidget(self.spin_dropout)
        row_dropout.addStretch()
        lay_rede.addLayout(row_dropout)
        left_layout.addWidget(grp_rede)

        # ━━ Grupo: Treinamento ━━
        grp_treino = QGroupBox("Hiperparâmetros de Treinamento")
        lay_treino = QVBoxLayout(grp_treino)
        lay_treino.setSpacing(10)

        grid_treino = QHBoxLayout()
        grid_treino.setSpacing(18)

        # Épocas
        col = QVBoxLayout()
        col.addWidget(QLabel("Épocas"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(150)
        col.addWidget(self.spin_epochs)
        grid_treino.addLayout(col)

        # Batch treino
        col = QVBoxLayout()
        col.addWidget(QLabel("Batch Treino"))
        self.spin_batch_train = QSpinBox()
        self.spin_batch_train.setRange(1, 8192)
        self.spin_batch_train.setValue(64)
        col.addWidget(self.spin_batch_train)
        grid_treino.addLayout(col)

        # Batch predição
        col = QVBoxLayout()
        col.addWidget(QLabel("Batch Predição"))
        self.spin_batch_pred = QSpinBox()
        self.spin_batch_pred.setRange(1, 65536)
        self.spin_batch_pred.setValue(4096)
        col.addWidget(self.spin_batch_pred)
        grid_treino.addLayout(col)

        # Test size
        col = QVBoxLayout()
        col.addWidget(QLabel("Test Size (%)"))
        self.spin_test_size = QDoubleSpinBox()
        self.spin_test_size.setRange(0.01, 0.99)
        self.spin_test_size.setSingleStep(0.01)
        self.spin_test_size.setDecimals(2)
        self.spin_test_size.setValue(0.30)
        col.addWidget(self.spin_test_size)
        grid_treino.addLayout(col)

        # Random state
        col = QVBoxLayout()
        col.addWidget(QLabel("Random State"))
        self.spin_random = QSpinBox()
        self.spin_random.setRange(0, 999999)
        self.spin_random.setValue(42)
        col.addWidget(self.spin_random)
        grid_treino.addLayout(col)

        lay_treino.addLayout(grid_treino)
        left_layout.addWidget(grp_treino)

        # ━━ Grupo: Hardware & Máscara ━━
        grp_hw = QGroupBox("Hardware & Pré-processamento")
        lay_hw = QVBoxLayout(grp_hw)
        lay_hw.setSpacing(10)

        row_hw1 = QHBoxLayout()
        row_hw1.setSpacing(16)

        # RAM limite
        col = QVBoxLayout()
        col.addWidget(QLabel("Limite RAM (%)"))
        self.spin_ram = QSpinBox()
        self.spin_ram.setRange(10, 95)
        self.spin_ram.setValue(70)
        self.spin_ram.setSuffix(" %")
        col.addWidget(self.spin_ram)
        row_hw1.addLayout(col)

        # Checkbox máscara
        self.chk_mascara = QCheckBox("Usar máscara (última banda = alpha)")
        self.chk_mascara.setChecked(True)
        row_hw1.addWidget(self.chk_mascara, alignment=Qt.AlignmentFlag.AlignBottom)

        # Valor mínimo alpha
        col = QVBoxLayout()
        col.addWidget(QLabel("Valor Mín. Alpha"))
        self.spin_alpha = QSpinBox()
        self.spin_alpha.setRange(0, 255)
        self.spin_alpha.setValue(250)
        col.addWidget(self.spin_alpha)
        row_hw1.addLayout(col)

        lay_hw.addLayout(row_hw1)
        left_layout.addWidget(grp_hw)

        # ━━ Grupo: Persistência ━━
        grp_modelo = QGroupBox("Persistência do Modelo")
        lay_modelo = QVBoxLayout(grp_modelo)
        lay_modelo.setSpacing(10)

        self.chk_salvar_modelo = QCheckBox("Salvar modelo treinado em disco (.keras)")
        self.chk_salvar_modelo.setChecked(True)
        lay_modelo.addWidget(self.chk_salvar_modelo)

        self.row_modelo_path = PathBrowseRow("Caminho do Modelo", "resultado/modelo_ui.keras",
                                              file_filter="Keras Model (*.keras)")
        lay_modelo.addWidget(self.row_modelo_path)
        left_layout.addWidget(grp_modelo)

        left_layout.addStretch()

        # ── Painel Direito: Log, Preview, Ações ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(18)

        # ━━ Preview / Status Card ━━
        card_preview = QFrame()
        card_preview.setStyleSheet(f"""
            QFrame {{
                background-color: {PANEL_BG};
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
        """)
        card_lay = QVBoxLayout(card_preview)
        card_lay.setContentsMargins(18, 18, 18, 18)
        card_lay.setSpacing(12)

        prev_title = QLabel("Resumo da Configuração")
        prev_title.setStyleSheet(f"color: {ACCENT_GOLD}; font-weight: 700; font-size: 14px;")
        card_lay.addWidget(prev_title)

        self.lbl_resumo = QTextEdit()
        self.lbl_resumo.setReadOnly(True)
        self.lbl_resumo.setMaximumHeight(140)
        self.lbl_resumo.setStyleSheet(f"""
            QTextEdit {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER};
                border-radius: 8px;
                color: {TEXT_SECONDARY};
                font-size: 12px;
                padding: 10px;
            }}
        """)
        self._update_resumo()
        card_lay.addWidget(self.lbl_resumo)

        right_layout.addWidget(card_preview)

        # ━━ Console / Log ━━
        grp_log = QGroupBox("Console de Execução")
        lay_log = QVBoxLayout(grp_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("Aguardando início do pipeline...")
        lay_log.addWidget(self.txt_log)
        right_layout.addWidget(grp_log, 1)

        # ━━ Progresso ━━
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat(" %p% — aguardando... ")
        right_layout.addWidget(self.progress)

        # ━━ Botões de Ação ━━
        action_bar = QWidget()
        action_lay = QHBoxLayout(action_bar)
        action_lay.setContentsMargins(0, 0, 0, 0)
        action_lay.setSpacing(12)

        self.btn_load_cfg = QPushButton("📂 Carregar Config")
        self.btn_load_cfg.setObjectName("btn_secondary")
        self.btn_load_cfg.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_save_cfg = QPushButton("💾 Salvar Config")
        self.btn_save_cfg.setObjectName("btn_secondary")
        self.btn_save_cfg.setCursor(Qt.CursorShape.PointingHandCursor)

        action_lay.addWidget(self.btn_load_cfg)
        action_lay.addWidget(self.btn_save_cfg)
        action_lay.addStretch()

        self.btn_executar = QPushButton("▶  EXECUTAR PIPELINE")
        self.btn_executar.setObjectName("btn_primary")
        self.btn_executar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_executar.setMinimumWidth(220)
        self.btn_executar.setMinimumHeight(46)
        action_lay.addWidget(self.btn_executar)

        right_layout.addWidget(action_bar)

        # Monta splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 55)
        splitter.setStretchFactor(1, 45)
        splitter.setHandleWidth(2)
        main_layout.addWidget(splitter, 1)

        # Conecta slots dummy (apenas UI, sem lógica ainda)
        self.btn_executar.clicked.connect(self._on_executar)
        self.btn_load_cfg.clicked.connect(self._on_load_cfg)
        self.btn_save_cfg.clicked.connect(self._on_save_cfg)

        # Atualiza resumo quando valores mudam
        for w in [
            self.row_img_treino.edit, self.row_img_classif.edit,
            self.row_img_saida.edit, self.edit_camadas,
            self.combo_ativacao, self.spin_dropout,
            self.spin_epochs, self.spin_batch_train,
            self.spin_batch_pred, self.spin_test_size,
            self.spin_ram, self.chk_mascara, self.chk_salvar_modelo
        ]:
            if hasattr(w, "textChanged"):
                w.textChanged.connect(self._update_resumo)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self._update_resumo)
            elif hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._update_resumo)
            elif hasattr(w, "stateChanged"):
                w.stateChanged.connect(self._update_resumo)

    # ── Helpers UI ──
    def _add_shp_row(self, path: str, classe: int):
        row = self.table_shp.rowCount()
        self.table_shp.insertRow(row)

        item_path = QTableWidgetItem(path)
        item_path.setFlags(item_path.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table_shp.setItem(row, 0, item_path)

        spin_cls = QSpinBox()
        spin_cls.setRange(0, 999)
        spin_cls.setValue(classe)
        spin_cls.setStyleSheet("background-color: transparent; border: none;")
        self.table_shp.setCellWidget(row, 1, spin_cls)

        btn_rem = QPushButton("Remover")
        btn_rem.setObjectName("btn_danger")
        btn_rem.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_rem.clicked.connect(lambda _, r=row: self._remove_shp_row(r))
        self.table_shp.setCellWidget(row, 2, btn_rem)

    def _remove_shp_row(self, row: int):
        self.table_shp.removeRow(row)
        # Reconecta botões para índices atualizados (simplificado)
        for r in range(self.table_shp.rowCount()):
            btn = self.table_shp.cellWidget(r, 2)
            if btn:
                btn.clicked.disconnect()
                btn.clicked.connect(lambda _, nr=r: self._remove_shp_row(nr))
        self._update_resumo()

    def _on_add_shp(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Adicionar Shapefile", "", "Shapefile (*.shp)"
        )
        if path:
            # Auto-incrementa classe baseado no maior existente
            max_cls = -1
            for r in range(self.table_shp.rowCount()):
                w = self.table_shp.cellWidget(r, 1)
                if isinstance(w, QSpinBox):
                    max_cls = max(max_cls, w.value())
            self._add_shp_row(path, max_cls + 1)
            self._update_resumo()

    def _update_resumo(self):
        treino = self.row_img_treino.path() or "—"
        classif = self.row_img_classif.path() or "—"
        saida = self.row_img_saida.path() or "—"
        camadas = self.edit_camadas.text() or "—"
        ativ = self.combo_ativacao.currentText()
        drop = self.spin_dropout.value()
        ep = self.spin_epochs.value()
        bt = self.spin_batch_train.value()
        bp = self.spin_batch_pred.value()
        ram = self.spin_ram.value()
        mask = "Sim" if self.chk_mascara.isChecked() else "Não"

        resumo = (
            f"<b>Imagem Treino:</b> {treino}<br>"
            f"<b>Imagem Classif.:</b> {classif}<br>"
            f"<b>Saída:</b> {saida}<br>"
            f"<b>Rede:</b> [{camadas}] — ativação {ativ}, dropout {drop}<br>"
            f"<b>Treino:</b> {ep} épocas | batch {bt} / pred {bp}<br>"
            f"<b>RAM limite:</b> {ram}% | Máscara: {mask}"
        )
        self.lbl_resumo.setHtml(resumo)

    # ── Slots stub (sem lógica de execução ainda) ──
    def _on_executar(self):
        self.txt_log.append("▶ Pipeline iniciado... [STUB — integrar lógica posteriormente]")
        self.progress.setValue(10)
        self.badge_status.setText("EXECUTANDO")
        self.badge_status.setStyleSheet(f"""
            QLabel {{
                background-color: {WARNING};
                color: {DARK_BG};
                border-radius: 6px;
                padding: 4px 14px;
                font-weight: 700;
                font-size: 11px;
            }}
        """)

    def _on_load_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Carregar Configuração", "", "JSON (*.json)"
        )
        if path:
            self.txt_log.append(f"📂 Config carregada: {path}  [STUB]")

    def _on_save_cfg(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Configuração", "config_ui.json", "JSON (*.json)"
        )
        if path:
            self.txt_log.append(f"💾 Config salva: {path}  [STUB]")


# ═══════════════════════════════════════════════════════════════════════════
# PONTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)

    # Fonte premium
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
