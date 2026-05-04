# -*- coding: utf-8 -*-
"""
UI Profissional Dark Charcoal — Classificador Raster Neural v6
===============================================================
Interface premium em PySide6 para o pipeline main6_multcore.py.
Apenas UI (frontend); a lógica de execucao sera integrada posteriormente.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QGroupBox, QSplitter, QTextEdit, QProgressBar, QFrame, QSizePolicy,
    QScrollArea
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QFont
from core.dark_charcoal_style import DarkCharcoalStyle
from core.hud_loader import HudCircularRingsLoader
from core.main_controller import MainController


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
    """Linha com label, campo de texto e botao browse."""
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
        self.label.setStyleSheet(
            f"color: {DarkCharcoalStyle.TEXT_SECONDARY}; font-weight: 500;"
        )

        self.edit = QLineEdit(default_path)
        self.edit.setPlaceholderText("Caminho do arquivo...")

        self.btn = QPushButton("...")
        self.btn.setObjectName("btn_secondary")
        self.btn.setFixedWidth(32)
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
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setMinimumSize(1280, 860)
        self.resize(1440, 900)
        self._drag_active = False
        self._drag_offset = QPoint()
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        title_bar = QWidget()
        title_bar.setObjectName("title_bar")
        title_bar.setFixedHeight(38)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 0, 6, 0)
        title_layout.setSpacing(8)

        self.lbl_window_title = QLabel(self.windowTitle())
        self.lbl_window_title.setObjectName("window_title")
        title_layout.addWidget(self.lbl_window_title)
        title_layout.addStretch()

        self.btn_min = QPushButton("—")
        self.btn_min.setObjectName("title_btn")
        self.btn_min.clicked.connect(self.showMinimized)
        title_layout.addWidget(self.btn_min)

        self.btn_max = QPushButton("□")
        self.btn_max.setObjectName("title_btn")
        self.btn_max.clicked.connect(self._toggle_maximize_restore)
        title_layout.addWidget(self.btn_max)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("title_btn_close")
        self.btn_close.clicked.connect(self.close)
        title_layout.addWidget(self.btn_close)

        root_layout.addWidget(title_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        central = QWidget()
        scroll.setWidget(central)
        root_layout.addWidget(scroll, 1)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(22, 16, 22, 16)
        main_layout.setSpacing(14)

        # HEADER PREMIUM
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)

        #self.lbl_title = QLabel("Classificador Raster Neural")
        #self.lbl_title.setObjectName("header_title")

        self.lbl_subtitle = QLabel(
            "Pipeline de classificacao supervisionada com redes neurais profundas "
            "— extracao espectral, treinamento multicore e exportacao GeoTIFF."
        )
        self.lbl_subtitle.setObjectName("header_subtitle")
        self.lbl_subtitle.setWordWrap(True)

       # title_col.addWidget(self.lbl_title)
        title_col.addWidget(self.lbl_subtitle)

        header_layout.addLayout(title_col, 1)

        self.badge_status = Badge("PRONTA")
        self.badge_status.setStyleSheet(
            "QLabel {"
            f"  background-color: {DarkCharcoalStyle.SUCCESS};"
            f"  color: {DarkCharcoalStyle.DARK_BG};"
            "  border-radius: 6px;"
            "  padding: 4px 14px;"
            "  font-weight: 700;"
            "  font-size: 11px;"
            "}"
        )
        header_layout.addWidget(self.badge_status, alignment=Qt.AlignmentFlag.AlignVCenter)

        main_layout.addWidget(header)
        main_layout.addWidget(Separator())

        # CORPO — Splitter esquerda / direita
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Painel Esquerdo
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Grupo: Imagens
        grp_imagens = QGroupBox("Imagens & Saida")
        lay_img = QVBoxLayout(grp_imagens)
        lay_img.setSpacing(8)

        self.row_img_treino = PathBrowseRow(
            "Imagem Treino", "dados/imagemTreino.tif",
            file_filter="GeoTIFF (*.tif *.tiff)"
        )
        self.row_img_classif = PathBrowseRow(
            "Imagem Classif.", "dados/imagemCompleta.tif",
            file_filter="GeoTIFF (*.tif *.tiff)"
        )
        self.row_img_saida = PathBrowseRow(
            "Saida GeoTIFF", "resultado/mapa_classificado_ui.tif",
            file_filter="GeoTIFF (*.tif *.tiff)"
        )

        lay_img.addWidget(self.row_img_treino)
        lay_img.addWidget(self.row_img_classif)
        lay_img.addWidget(self.row_img_saida)

        left_layout.addWidget(grp_imagens)

        # Grupo: Amostras
        grp_amostras = QGroupBox("Amostras — Shapefiles por Classe")
        lay_amostras = QVBoxLayout(grp_amostras)
        lay_amostras.setSpacing(8)

        self.table_shp = QTableWidget(0, 4)
        self.table_shp.setHorizontalHeaderLabels(
            ["Caminho do Shapefile", "ID Classe", "Legenda", "Acao"]
        )
        self.table_shp.horizontalHeader().setStretchLastSection(False)
        self.table_shp.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table_shp.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Fixed
        )
        self.table_shp.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Fixed
        )
        self.table_shp.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Fixed
        )
        self.table_shp.setColumnWidth(1, 80)
        self.table_shp.setColumnWidth(2, 140)
        self.table_shp.setColumnWidth(3, 90)
        self.table_shp.setMinimumHeight(140)

        btn_add_shp = QPushButton("+ Adicionar Shapefile")
        btn_add_shp.setObjectName("btn_action")
        btn_add_shp.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_shp = btn_add_shp

        lay_amostras.addWidget(self.table_shp)
        lay_amostras.addWidget(btn_add_shp, alignment=Qt.AlignmentFlag.AlignLeft)
        left_layout.addWidget(grp_amostras)

        # Grupo: Arquitetura da Rede
        grp_rede = QGroupBox("Arquitetura da Rede Neural")
        lay_rede = QVBoxLayout(grp_rede)
        lay_rede.setSpacing(8)

        row_camadas = QHBoxLayout()
        row_camadas.setSpacing(10)
        row_camadas.addWidget(QLabel("Camadas Ocultas:"))
        self.edit_camadas = QLineEdit("128, 64, 32")
        self.edit_camadas.setPlaceholderText("ex: 256, 128, 64")
        row_camadas.addWidget(self.edit_camadas, 1)

        row_camadas.addWidget(QLabel("Ativacao:"))
        self.combo_ativacao = QComboBox()
        self.combo_ativacao.addItems(["relu", "elu", "tanh", "sigmoid", "linear"])
        self.combo_ativacao.setCurrentText("relu")
        row_camadas.addWidget(self.combo_ativacao)

        lay_rede.addLayout(row_camadas)

        row_dropout = QHBoxLayout()
        row_dropout.setSpacing(10)
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

        # Grupo: Treinamento
        grp_treino = QGroupBox("Hiperparametros de Treinamento")
        lay_treino = QVBoxLayout(grp_treino)
        lay_treino.setSpacing(8)

        grid_treino = QHBoxLayout()
        grid_treino.setSpacing(12)

        col = QVBoxLayout()
        col.addWidget(QLabel("Epocas"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(150)
        col.addWidget(self.spin_epochs)
        grid_treino.addLayout(col)

        col = QVBoxLayout()
        col.addWidget(QLabel("Batch Treino"))
        self.spin_batch_train = QSpinBox()
        self.spin_batch_train.setRange(1, 8192)
        self.spin_batch_train.setValue(64)
        col.addWidget(self.spin_batch_train)
        grid_treino.addLayout(col)

        col = QVBoxLayout()
        col.addWidget(QLabel("Batch Predicao"))
        self.spin_batch_pred = QSpinBox()
        self.spin_batch_pred.setRange(1, 65536)
        self.spin_batch_pred.setValue(4096)
        col.addWidget(self.spin_batch_pred)
        grid_treino.addLayout(col)

        col = QVBoxLayout()
        col.addWidget(QLabel("Test Size (%)"))
        self.spin_test_size = QDoubleSpinBox()
        self.spin_test_size.setRange(0.01, 0.99)
        self.spin_test_size.setSingleStep(0.01)
        self.spin_test_size.setDecimals(2)
        self.spin_test_size.setValue(0.30)
        col.addWidget(self.spin_test_size)
        grid_treino.addLayout(col)

        col = QVBoxLayout()
        col.addWidget(QLabel("Random State"))
        self.spin_random = QSpinBox()
        self.spin_random.setRange(0, 999999)
        self.spin_random.setValue(42)
        col.addWidget(self.spin_random)
        grid_treino.addLayout(col)

        lay_treino.addLayout(grid_treino)
        left_layout.addWidget(grp_treino)

        # Grupo: Hardware & Mascara
        grp_hw = QGroupBox("Hardware & Pre-processamento")
        lay_hw = QVBoxLayout(grp_hw)
        lay_hw.setSpacing(8)

        row_hw1 = QHBoxLayout()
        row_hw1.setSpacing(12)

        col = QVBoxLayout()
        col.addWidget(QLabel("Limite RAM (%)"))
        self.spin_ram = QSpinBox()
        self.spin_ram.setRange(10, 95)
        self.spin_ram.setValue(70)
        self.spin_ram.setSuffix(" %")
        col.addWidget(self.spin_ram)
        row_hw1.addLayout(col)

        self.chk_mascara = QCheckBox("Usar mascara (ultima banda = alpha)")
        self.chk_mascara.setChecked(True)
        row_hw1.addWidget(self.chk_mascara, alignment=Qt.AlignmentFlag.AlignBottom)

        col = QVBoxLayout()
        col.addWidget(QLabel("Valor Min. Alpha"))
        self.spin_alpha = QSpinBox()
        self.spin_alpha.setRange(0, 255)
        self.spin_alpha.setValue(250)
        col.addWidget(self.spin_alpha)
        row_hw1.addLayout(col)

        lay_hw.addLayout(row_hw1)
        left_layout.addWidget(grp_hw)

        # Grupo: Persistencia
        grp_modelo = QGroupBox("Persistencia do Modelo")
        lay_modelo = QVBoxLayout(grp_modelo)
        lay_modelo.setSpacing(8)

        row_model_action = QHBoxLayout()
        row_model_action.setSpacing(10)
        row_model_action.addWidget(QLabel("Acao do Modelo:"))
        self.combo_model_action = QComboBox()
        self.combo_model_action.addItems([
            "Treinar modelo novo",
            "Treinar modelo existente",
            "Usar modelo existente"
        ])
        self.combo_model_action.setCurrentText("Treinar modelo novo")
        row_model_action.addWidget(self.combo_model_action, 1)
        lay_modelo.addLayout(row_model_action)

        self.row_modelo_existente = PathBrowseRow(
            "Modelo Existente", "",
            file_filter="Keras Model (*.keras)"
        )
        self.row_modelo_existente.setVisible(False)
        lay_modelo.addWidget(self.row_modelo_existente)
        
        self.btn_listar_modelos = QPushButton("Listar Modelos")
        self.btn_listar_modelos.setObjectName("btn_secondary")
        self.btn_listar_modelos.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_listar_modelos.setVisible(False)
        lay_modelo.addWidget(self.btn_listar_modelos, alignment=Qt.AlignmentFlag.AlignLeft)

        self.chk_salvar_modelo = QCheckBox("Salvar modelo treinado em disco (.keras)")
        self.chk_salvar_modelo.setChecked(True)
        lay_modelo.addWidget(self.chk_salvar_modelo)

        self.row_modelo_path = PathBrowseRow(
            "Caminho do Modelo", "resultado/modelo_ui.keras",
            file_filter="Keras Model (*.keras)"
        )
        lay_modelo.addWidget(self.row_modelo_path)
        left_layout.addWidget(grp_modelo)

        left_layout.addStretch()

        # Painel Direito
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setContentsMargins(0, 12, 0, 0)
        right_layout.setSpacing(6)

        card_preview = QFrame()
        card_preview.setStyleSheet(
            "QFrame {"
            f"  background-color: {DarkCharcoalStyle.PANEL_BG};"
            f"  border: 1px solid {DarkCharcoalStyle.BORDER};"
            "  border-radius: 12px;"
            "}"
        )
        card_lay = QVBoxLayout(card_preview)
        card_lay.setContentsMargins(12, 12, 12, 12)
        card_lay.setSpacing(8)

        prev_title = QLabel("Resumo da Configuracao")
        prev_title.setStyleSheet(
            f"color: {DarkCharcoalStyle.ACCENT_GOLD}; font-weight: 700; font-size: 14px;"
        )
        card_lay.addWidget(prev_title)

        self.lbl_resumo = QTextEdit()
        self.lbl_resumo.setReadOnly(True)
        self.lbl_resumo.setMaximumHeight(140)
        self.lbl_resumo.setStyleSheet(
            "QTextEdit {"
            f"  background-color: {DarkCharcoalStyle.CARD_BG};"
            f"  border: 1px solid {DarkCharcoalStyle.BORDER};"
            "  border-radius: 8px;"
            f"  color: {DarkCharcoalStyle.TEXT_SECONDARY};"
            "  font-size: 12px;"
            "  padding: 10px;"
            "}"
        )
        card_lay.addWidget(self.lbl_resumo)

        right_layout.addWidget(card_preview)

        grp_log = QGroupBox("Console de Execucao")
        lay_log = QVBoxLayout(grp_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("Aguardando inicio do pipeline...")
        lay_log.addWidget(self.txt_log)
        right_layout.addWidget(grp_log, 1)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat(" %p% — aguardando... ")
        right_layout.addWidget(self.progress)

        action_bar = QWidget()
        action_lay = QHBoxLayout(action_bar)
        action_lay.setContentsMargins(0, 0, 0, 0)
        action_lay.setSpacing(8)

        self.btn_load_cfg = QPushButton("Carregar Config")
        self.btn_load_cfg.setObjectName("btn_secondary")
        self.btn_load_cfg.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_save_cfg = QPushButton("Salvar Config")
        self.btn_save_cfg.setObjectName("btn_secondary")
        self.btn_save_cfg.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_reset_cfg = QPushButton("Restaurar Padrao")
        self.btn_reset_cfg.setObjectName("btn_secondary")
        self.btn_reset_cfg.setCursor(Qt.CursorShape.PointingHandCursor)

        action_lay.addWidget(self.btn_load_cfg)
        action_lay.addWidget(self.btn_save_cfg)
        action_lay.addWidget(self.btn_reset_cfg)
        action_lay.addStretch()

        self.btn_executar = QPushButton("EXECUTAR PIPELINE")
        self.btn_executar.setObjectName("btn_primary")
        self.btn_executar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_executar.setMinimumWidth(200)
        self.btn_executar.setMinimumHeight(38)
        action_lay.addWidget(self.btn_executar)

        right_layout.addWidget(action_bar)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 55)
        splitter.setStretchFactor(1, 45)
        splitter.setHandleWidth(2)
        main_layout.addWidget(splitter, 1)

        self.controller = MainController(self)
        self.loader_overlay = HudCircularRingsLoader(self)
        self.loader_overlay.setGeometry(self.rect())

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
        for r in range(self.table_shp.rowCount()):
            btn = self.table_shp.cellWidget(r, 2)
            if btn:
                try:
                    btn.clicked.disconnect()
                except Exception:
                    pass
                btn.clicked.connect(lambda _, nr=r: self._remove_shp_row(nr))

    def _toggle_maximize_restore(self):
        if self.isMaximized():
            self.showNormal()
            if hasattr(self, "btn_max"):
                self.btn_max.setText("□")
        else:
            self.showMaximized()
            if hasattr(self, "btn_max"):
                self.btn_max.setText("❐")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "loader_overlay"):
            self.loader_overlay.setGeometry(self.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= 38:
            self._drag_active = True
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_active and not self.isMaximized():
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_active = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= 38:
            self._toggle_maximize_restore()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
# PONTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DarkCharcoalStyle.stylesheet())

    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
