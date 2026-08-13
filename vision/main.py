# -*- coding: utf-8 -*-
"""
Classificador de Uso do Solo — Random Forest
============================================
UI PySide6 com painéis de configuração, console, barra de progresso
com ETA/tempo decorrido/restante, e preferências persistidas.

Uso:
    python main.py
"""

import os
import sys
import time
import traceback
from datetime import datetime, timedelta

from PySide6.QtCore import Qt, QObject, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QDoubleSpinBox,
    QListWidget, QAbstractItemView, QPlainTextEdit, QProgressBar,
    QFrame, QSplitter, QStatusBar, QDialog, QCheckBox, QFileDialog, QSpinBox,
    QComboBox, QListWidgetItem,
)

from core.styles import Colors, Styles
from core.preferences import Preferences
from core.classifier_manager import ClassifierConfig, ClassifierManager

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DO APP
# ─────────────────────────────────────────────────────────────────────────────

APP_VERSAO = "1.0.0"
APP_DATA_ATUALIZACAO = "13/08/2026"
APP_TO = "Palmas - TO"
APP_EMPRESA = "Linhas Brasil"

MAX_SHAPES = 8


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE OFUSCADA DO AUTOR (Npb) — XOR + hexadecimal
# ─────────────────────────────────────────────────────────────────────────────

def _gerar_hex_ofuscado(nome: str, chave: str) -> str:
    """Gera o valor hexadecimal do nome XOR chave (use para novo autor)."""
    data = nome.encode("utf-8")
    key_bytes = chave.encode("utf-8")
    return bytes(
        b ^ key_bytes[i % len(key_bytes)]
        for i, b in enumerate(data)
    ).hex()


class Npb:
    """Classe interna (ofuscada) que revela o autor do programa."""

    _LFGT = _gerar_hex_ofuscado("Linhas Brasil", "Lb")
    _LGTR = "Lb"

    @staticmethod
    def _decrypt(hex_value: str, key: str) -> str:
        data = bytes.fromhex(hex_value)
        key_bytes = key.encode("utf-8")
        return bytes(
            b ^ key_bytes[i % len(key_bytes)]
            for i, b in enumerate(data)
        ).decode("utf-8")

    @classmethod
    def npbt(cls) -> str:
        """Retorna o nome real, já descriptografado."""
        return cls._decrypt(cls._LFGT, cls._LGTR)


# ─────────────────────────────────────────────────────────────────────────────
# DIÁLOGO SOBRE
# ─────────────────────────────────────────────────────────────────────────────

class SobreDialog(QDialog):
    """Diálogo 'Sobre' com informações da aplicação."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Informações")
        self.setFixedWidth(440)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(28, 28, 28, 28)

        title = QLabel("ℹ️ Sobre")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        app_name = QLabel("Classificador de Uso do Solo")
        app_name.setObjectName("subtitleLabel")
        app_name.setAlignment(Qt.AlignCenter)
        layout.addWidget(app_name)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {Colors.BORDER};")
        layout.addWidget(sep)

        grid = QGridLayout()
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(16)

        linhas = [
            ("Versão", APP_VERSAO),
            ("Atualização", APP_DATA_ATUALIZACAO),
            ("Local", APP_TO),
            ("Empresa", APP_EMPRESA),
            ("Autor", Npb.npbt()),
        ]

        for i, (rotulo, valor) in enumerate(linhas):
            lbl_rotulo = QLabel(rotulo)
            lbl_rotulo.setStyleSheet(
                f"color: {Colors.GOLD}; font-weight: 700;"
            )
            lbl_valor = QLabel(valor)
            lbl_valor.setStyleSheet(f"color: {Colors.TEXT};")
            lbl_valor.setWordWrap(True)
            grid.addWidget(lbl_rotulo, i, 0, Qt.AlignTop)
            grid.addWidget(lbl_valor, i, 1)

        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_ok = QPushButton("OK")
        btn_ok.setObjectName("primaryButton")
        btn_ok.setFixedWidth(100)
        btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(btn_ok)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE — REDIRECIONAMENTO DE STDOUT/STDERR
# ─────────────────────────────────────────────────────────────────────────────

class StreamRedirector(QObject):
    """Objeto file-like que redireciona sys.stdout/sys.stderr para um sinal."""

    text_written = Signal(str)

    def write(self, text: str):
        if text:
            payload = str(text)
            if not payload.endswith("\n"):
                payload += "\n"
            self.text_written.emit(payload)

    def flush(self):
        pass


class ConsoleBridge(QObject):
    """Canal seguro para encaminhar mensagens da thread de trabalho à UI."""

    message_written = Signal(str)

    def write(self, text: str):
        if text:
            payload = str(text)
            if not payload.endswith("\n"):
                payload += "\n"
            self.message_written.emit(payload)

    def flush(self):
        pass


class ConsoleWidget(QPlainTextEdit):
    """Console de exibição com buffer de linhas e auto-scroll."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("consoleWidget")
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._pending = ""

    def append_line(self, text: str):
        self.appendPlainText(text)
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def flush_pending(self):
        if self._pending:
            self.append_line(self._pending)
            self._pending = ""

    def write_stream(self, text: str):
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self.append_line(line)
        if len(self._pending) > 4096:
            self.flush_pending()

    def copy_all(self):
        self.selectAll()
        self.copy()
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)

    def clear_console(self):
        self.clear()
        self._pending = ""


# ─────────────────────────────────────────────────────────────────────────────
# HEADER BAR
# ─────────────────────────────────────────────────────────────────────────────

class HeaderBar(QFrame):
    """Barra de cabeçalho com título, spinner, botão info e status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("headerBar")
        self.setFixedHeight(78)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.spinner = QLabel("⚙")
        self.spinner.setStyleSheet(f"font-size: 26px; color: {Colors.GOLD};")
        self.spinner.setFixedWidth(34)
        layout.addWidget(self.spinner, 0, Qt.AlignCenter)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)

        self.title_label = QLabel("CLASSIFICADOR DE USO DO SOLO")
        self.title_label.setObjectName("titleLabel")
        title_box.addWidget(self.title_label)

        self.subtitle_label = QLabel("Random Forest · Processamento em segundo plano")
        self.subtitle_label.setObjectName("subtitleLabel")
        title_box.addWidget(self.subtitle_label)

        layout.addLayout(title_box, 1)

        self.btn_info = QPushButton("ⓘ")
        self.btn_info.setToolTip("Informações")
        self.btn_info.setFixedSize(32, 32)
        self.btn_info.setCursor(Qt.PointingHandCursor)
        self.btn_info.setStyleSheet(
            "QPushButton {"
            f"  background-color: {Colors.PANEL};"
            f"  color: {Colors.GOLD};"
            f"  border: 1px solid {Colors.BORDER};"
            "  border-radius: 16px;"
            "  font-size: 16px;"
            "  font-weight: 700;"
            "  padding: 0px;"
            "}"
            "QPushButton:hover {"
            f"  border-color: {Colors.GOLD};"
            f"  background-color: {Colors.BLACK_SOFT};"
            "}"
        )
        layout.addWidget(self.btn_info, 0, Qt.AlignVCenter)

        self.status_label = QLabel("● PRONTO")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setStyleSheet(f"color: {Colors.SUCCESS};")
        layout.addWidget(self.status_label, 0, Qt.AlignVCenter)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_timer.start(1800)
        self._pulse_state = False

    def _pulse(self):
        self._pulse_state = not self._pulse_state
        color = Colors.GOLD_LIGHT if self._pulse_state else Colors.GOLD
        self.title_label.setStyleSheet(f"color: {color};")

    def set_status(self, text: str, color: str):
        self.status_label.setText(f"● {text}")
        self.status_label.setStyleSheet(f"color: {color};")


# ─────────────────────────────────────────────────────────────────────────────
# PAINEL — ARQUIVOS (TIFF entrada + TIFF saída)
# ─────────────────────────────────────────────────────────────────────────────

class FilesPanel(QGroupBox):
    """Painel de seleção de arquivos TIFF (entrada e saída)."""

    def __init__(self, parent=None):
        super().__init__("📁 Arquivos TIFF", parent)
        layout = QGridLayout(self)
        layout.setSpacing(2)

        layout.addWidget(QLabel("🖼️ Entrada (TIFF):"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Selecione o TIFF de entrada…")
        layout.addWidget(self.input_edit, 0, 1)
        btn_in = QPushButton("📂")
        btn_in.setObjectName("smallButton")
        btn_in.setFixedWidth(36)
        btn_in.setToolTip("Selecionar TIFF de entrada")
        btn_in.clicked.connect(self._browse_input)
        layout.addWidget(btn_in, 0, 2)

        layout.addWidget(QLabel("💾 Saída (TIFF):"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Selecione onde salvar o TIFF classificado…")
        layout.addWidget(self.output_edit, 1, 1)
        btn_out = QPushButton("📂")
        btn_out.setObjectName("smallButton")
        btn_out.setFixedWidth(36)
        btn_out.setToolTip("Selecionar arquivo de saída")
        btn_out.clicked.connect(self._browse_output)
        layout.addWidget(btn_out, 1, 2)

    def _browse_input(self):
        path = QFileDialog.getOpenFileName(
            self, "Selecionar TIFF de entrada",
            self.input_edit.text() or os.path.abspath("."),
            "TIFF (*.tif *.tiff);;Todos (*)",
        )[0]
        if path:
            self.input_edit.setText(path)
            # Sugere saída automática se vazia
            if not self.output_edit.text().strip():
                base, ext = os.path.splitext(path)
                self.output_edit.setText(base + "_classificado.tif")

    def _browse_output(self):
        path = QFileDialog.getSaveFileName(
            self, "Salvar TIFF classificado como",
            self.output_edit.text() or os.path.abspath("."),
            "TIFF (*.tif *.tiff)",
        )[0]
        if path:
            if not path.lower().endswith((".tif", ".tiff")):
                path += ".tif"
            self.output_edit.setText(path)

    def get_files(self):
        return {
            "input_tiff": self.input_edit.text().strip(),
            "output_tiff": self.output_edit.text().strip(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PAINEL — SHAPES DINÂMICOS (até 8, reordenáveis)
# ─────────────────────────────────────────────────────────────────────────────

class ShapesPanel(QGroupBox):
    """Painel de shapefiles dinâmicos — até 8, com reordenação."""

    def __init__(self, parent=None):
        super().__init__("🗺️ Shapefiles de Treino (até 8)", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(2)

        # Lista de shapes
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setMaximumHeight(180)
        layout.addWidget(self.list_widget, 1)

        # Botões de ação
        btns = QHBoxLayout()
        btns.setSpacing(2)

        self.btn_add = QPushButton("➕ Adicionar")
        self.btn_add.setObjectName("smallButton")
        self.btn_add.clicked.connect(self._add_shape)
        btns.addWidget(self.btn_add)

        self.btn_add_many = QPushButton("📚 Selecionar vários")
        self.btn_add_many.setObjectName("smallButton")
        self.btn_add_many.setToolTip("Seleciona vários shapefiles de uma vez")
        self.btn_add_many.clicked.connect(self._add_many_shapes)
        btns.addWidget(self.btn_add_many)

        self.btn_remove = QPushButton("🗑️ Remover")
        self.btn_remove.setObjectName("smallButton")
        self.btn_remove.clicked.connect(self._remove_shape)
        btns.addWidget(self.btn_remove)

        self.btn_up = QPushButton("⬆️")
        self.btn_up.setObjectName("smallButton")
        self.btn_up.setToolTip("Mover para cima")
        self.btn_up.clicked.connect(self._move_up)
        btns.addWidget(self.btn_up)

        self.btn_down = QPushButton("⬇️")
        self.btn_down.setObjectName("smallButton")
        self.btn_down.setToolTip("Mover para baixo")
        self.btn_down.clicked.connect(self._move_down)
        btns.addWidget(self.btn_down)

        self.btn_clear = QPushButton("🧹 Limpar")
        self.btn_clear.setObjectName("smallButton")
        self.btn_clear.clicked.connect(self._clear_shapes)
        btns.addWidget(self.btn_clear)

        btns.addStretch(1)
        layout.addLayout(btns)

        # Dica
        hint = QLabel("💡 Clique 2x em um item para renomear a classe.")
        hint.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 11px;")
        layout.addWidget(hint)

        self.list_widget.itemDoubleClicked.connect(self._rename_shape)

    # ── Internos ────────────────────────────────────────────────────────────

    def _add_shape(self):
        if self.list_widget.count() >= MAX_SHAPES:
            return
        path = QFileDialog.getOpenFileName(
            self, "Selecionar shapefile", os.path.abspath("."),
            "Shapefiles (*.shp);;Todos (*)",
        )[0]
        if path:
            self._append_item(path)

    def _add_many_shapes(self):
        """Seleciona vários shapefiles de uma vez."""
        remaining = MAX_SHAPES - self.list_widget.count()
        if remaining <= 0:
            return
        paths = QFileDialog.getOpenFileNames(
            self, "Selecionar shapefiles (vários)", os.path.abspath("."),
            "Shapefiles (*.shp);;Todos (*)",
        )[0]
        for path in paths[:remaining]:
            self._append_item(path)

    def _append_item(self, path: str):
        name = os.path.splitext(os.path.basename(path))[0]
        item = QListWidgetItem(f"{name}  →  {path}")
        item.setData(Qt.UserRole, {"name": name, "path": path})
        self.list_widget.addItem(item)

    def _remove_shape(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)

    def _move_up(self):
        row = self.list_widget.currentRow()
        if row > 0:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row - 1, item)
            self.list_widget.setCurrentRow(row - 1)

    def _move_down(self):
        row = self.list_widget.currentRow()
        if 0 <= row < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row + 1, item)
            self.list_widget.setCurrentRow(row + 1)

    def _clear_shapes(self):
        self.list_widget.clear()

    def _rename_shape(self, item: QListWidgetItem):
        """Renomeia a classe do shapefile (duplo clique)."""
        from PySide6.QtWidgets import QInputDialog
        data = item.data(Qt.UserRole)
        if not data:
            return
        novo_nome, ok = QInputDialog.getText(
            self, "Renomear classe",
            "Nome da classe:",
            text=data["name"],
        )
        if ok and novo_nome.strip():
            data["name"] = novo_nome.strip()
            item.setData(Qt.UserRole, data)
            item.setText(f"{data['name']}  →  {data['path']}")

    # ── API ─────────────────────────────────────────────────────────────────

    def get_shapes_list(self) -> list:
        """Retorna lista de dicts [{"name": ..., "path": ...}] na ordem atual."""
        result = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(Qt.UserRole)
            if data and data.get("path"):
                result.append({
                    "name": data.get("name", f"classe_{i+1}"),
                    "path": data["path"],
                })
        return result

    def set_shapes_list(self, shapes_list: list):
        """Preenche a lista a partir de uma lista de dicts."""
        self.list_widget.clear()
        for item_data in shapes_list:
            name = item_data.get("name", "classe")
            path = item_data.get("path", "")
            if path:
                item = QListWidgetItem(f"{name}  →  {path}")
                item.setData(Qt.UserRole, {"name": name, "path": path})
                self.list_widget.addItem(item)


# ─────────────────────────────────────────────────────────────────────────────
# PAINEL — PARÂMETROS DE TREINO
# ─────────────────────────────────────────────────────────────────────────────

class TrainPanel(QGroupBox):
    """Painel de parâmetros de treino do Random Forest."""

    def __init__(self, parent=None):
        super().__init__("🎓 Parâmetros de Treino", parent)
        layout = QGridLayout(self)
        layout.setSpacing(2)

        layout.addWidget(QLabel("Amostras por classe:"), 0, 0)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1000, 10_000_000)
        self.samples_spin.setSingleStep(1000)
        self.samples_spin.setValue(60000)
        self.samples_spin.setToolTip("Pixels de treino por classe (balanceado)")
        layout.addWidget(self.samples_spin, 0, 1)

        layout.addWidget(QLabel("Árvores (RF):"), 1, 0)
        self.trees_spin = QSpinBox()
        self.trees_spin.setRange(10, 2000)
        self.trees_spin.setSingleStep(10)
        self.trees_spin.setValue(200)
        self.trees_spin.setToolTip("Mais árvores = mais lento, porém mais preciso")
        layout.addWidget(self.trees_spin, 1, 1)

        layout.addWidget(QLabel("Núcleos CPU:"), 2, 0)
        self.jobs_combo = QComboBox()
        self.jobs_combo.addItem("Todos (-1)", -1)
        self.jobs_combo.addItem("1", 1)
        self.jobs_combo.addItem("2", 2)
        self.jobs_combo.addItem("4", 4)
        self.jobs_combo.addItem("8", 8)
        self.jobs_combo.addItem("16", 16)
        layout.addWidget(self.jobs_combo, 2, 1)

    def get_train(self):
        return {
            "samples_per_class": self.samples_spin.value(),
            "rf_n_trees": self.trees_spin.value(),
            "rf_jobs": self.jobs_combo.currentData(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PAINEL — PARÂMETROS DE PREDIÇÃO E VETORIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

class ProcessPanel(QGroupBox):
    """Painel de parâmetros de predição e vetorização."""

    def __init__(self, parent=None):
        super().__init__("🔧 Predição & Vetorização", parent)
        layout = QGridLayout(self)
        layout.setSpacing(2)

        layout.addWidget(QLabel("Tamanho do tile (px):"), 0, 0)
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(256, 8192)
        self.tile_spin.setSingleStep(256)
        self.tile_spin.setValue(2048)
        self.tile_spin.setToolTip("Pixels por tile — RAM ↔ velocidade")
        layout.addWidget(self.tile_spin, 0, 1)

        layout.addWidget(QLabel("Confiança mínima:"), 1, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 1.00)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.45)
        self.conf_spin.setToolTip("Abaixo disso → sem classe (0)")
        layout.addWidget(self.conf_spin, 1, 1)

        layout.addWidget(QLabel("Área mínima (m²):"), 2, 0)
        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.1, 10_000.0)
        self.min_area_spin.setSingleStep(0.5)
        self.min_area_spin.setDecimals(1)
        self.min_area_spin.setValue(5.0)
        self.min_area_spin.setToolTip("Polígonos e buracos menores são eliminados")
        layout.addWidget(self.min_area_spin, 2, 1)

        layout.addWidget(QLabel("Iter. morfológicas:"), 3, 0)
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(0, 10)
        self.smooth_spin.setValue(2)
        self.smooth_spin.setToolTip("Abertura/fechamento morfológico")
        layout.addWidget(self.smooth_spin, 3, 1)

        layout.addWidget(QLabel("Área de buracos (m²):"), 4, 0)
        self.hole_spin = QDoubleSpinBox()
        self.hole_spin.setRange(0.1, 10_000.0)
        self.hole_spin.setSingleStep(0.5)
        self.hole_spin.setDecimals(1)
        self.hole_spin.setValue(5.0)
        self.hole_spin.setToolTip("Buracos internos menores que isso são preenchidos")
        layout.addWidget(self.hole_spin, 4, 1)

        layout.addWidget(QLabel("Buffer ponto (m):"), 5, 0)
        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(0.0, 100.0)
        self.buffer_spin.setSingleStep(0.1)
        self.buffer_spin.setDecimals(1)
        self.buffer_spin.setValue(0.1)
        self.buffer_spin.setToolTip("Buffer aplicado a shapefiles do tipo Ponto")
        layout.addWidget(self.buffer_spin, 5, 1)

        layout.addWidget(QLabel("Forçar re-treino:"), 6, 0)
        self.retrain_check = QCheckBox("Treinar modelo novamente")
        self.retrain_check.setToolTip("Ignora o modelo salvo e treina do zero")
        layout.addWidget(self.retrain_check, 6, 1)

    def get_process(self):
        return {
            "tile_sz": self.tile_spin.value(),
            "conf_threshold": self.conf_spin.value(),
            "min_area_m2": self.min_area_spin.value(),
            "smooth_iter": self.smooth_spin.value(),
            "hole_area_m2": self.hole_spin.value(),
            "buffer_m": self.buffer_spin.value(),
            "force_retrain": self.retrain_check.isChecked(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# WORKER — PROCESSAMENTO EM SEGUNDO PLANO
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingWorker(QObject):
    """Worker que executa o pipeline em uma QThread."""

    started = Signal()
    finished = Signal(bool, str)
    progress = Signal(float, str)
    log_message = Signal(str)

    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self._config = config
        self._manager = None
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True
        if self._manager is not None:
            self._manager.request_stop()

    def run(self):
        self.started.emit()
        try:
            self._manager = ClassifierManager(
                self._config,
                log_callback=self.log_message.emit,
                progress_callback=lambda pct, msg: self.progress.emit(float(pct), str(msg)),
            )
            self._manager.run()
            if self._stop_requested:
                self.finished.emit(False, "Processamento interrompido pelo usuário.")
            else:
                self.finished.emit(True, "Pipeline concluído com sucesso.")
        except InterruptedError:
            self.finished.emit(False, "Processamento interrompido pelo usuário.")
        except Exception as exc:
            tb = traceback.format_exc()
            self.log_message.emit(f"❌ ERRO: {exc}\n{tb}\n")
            self.finished.emit(False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# JANELA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Janela principal do classificador."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classificador de Uso do Solo — Random Forest")
        self.resize(1280, 780)
        self.setMinimumSize(1024, 640)

        self.prefs = Preferences("classificador")
        self._worker = None
        self._worker_thread = None
        self._start_time = 0.0

        self._setup_ui()
        self._setup_console_redirect()
        self._setup_signals()
        self._carregar_preferencias()

        self.statusBar().showMessage("Pronto", 3000)

    # ── UI ──────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        self.header = HeaderBar(self)
        self.header.btn_info.clicked.connect(self._on_info_clicked)
        root.addWidget(self.header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p%")
        root.addWidget(self.progress_bar)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # Coluna esquerda — configurações
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        self.files_panel = FilesPanel(left)
        left_layout.addWidget(self.files_panel)

        self.shapes_panel = ShapesPanel(left)
        left_layout.addWidget(self.shapes_panel)

        self.train_panel = TrainPanel(left)
        left_layout.addWidget(self.train_panel)

        self.process_panel = ProcessPanel(left)
        left_layout.addWidget(self.process_panel)

        btn_layout = QHBoxLayout()
        self.btn_process = QPushButton("🚀 PROCESSAR")
        self.btn_process.setObjectName("primaryButton")
        self.btn_process.setMinimumHeight(40)
        self.btn_process.clicked.connect(self._on_process_clicked)
        btn_layout.addWidget(self.btn_process, 2)

        self.btn_stop = QPushButton("⏹️ INTERROMPER")
        self.btn_stop.setObjectName("dangerButton")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        btn_layout.addWidget(self.btn_stop, 1)

        left_layout.addLayout(btn_layout)
        left_layout.addStretch(1)

        splitter.addWidget(left)

        # Coluna direita — console
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self.console = ConsoleWidget(right)
        right_layout.addWidget(self.console, 1)

        console_btns = QHBoxLayout()
        btn_clear = QPushButton("🗑️ Limpar")
        btn_clear.clicked.connect(self.console.clear_console)
        console_btns.addWidget(btn_clear)

        btn_copy = QPushButton("📋 Copiar")
        btn_copy.clicked.connect(self.console.copy_all)
        console_btns.addWidget(btn_copy)

        btn_test = QPushButton("🧪 Testar")
        btn_test.setToolTip("Testa o redirecionamento do console")
        btn_test.clicked.connect(self._on_test_console)
        console_btns.addWidget(btn_test)

        console_btns.addStretch(1)
        right_layout.addLayout(console_btns)

        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 740])

    # ── Console redirect ────────────────────────────────────────────────────

    def _setup_console_redirect(self):
        self._console_bridge = ConsoleBridge()
        self._stdout_redirector = StreamRedirector()
        self._stderr_redirector = StreamRedirector()

        self._console_bridge.message_written.connect(
            self.console.write_stream, Qt.QueuedConnection
        )
        self._stdout_redirector.text_written.connect(self._console_bridge.write)
        self._stderr_redirector.text_written.connect(self._console_bridge.write)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self._stdout_redirector
        sys.stderr = self._stderr_redirector

    # ── Signals ─────────────────────────────────────────────────────────────

    def _setup_signals(self):
        """Conecta todos os widgets de preferência ao salvamento."""
        for widget in [
            self.files_panel.input_edit,
            self.files_panel.output_edit,
        ]:
            widget.textChanged.connect(self._on_pref_changed)

        for spin in [
            self.train_panel.samples_spin,
            self.train_panel.trees_spin,
            self.process_panel.tile_spin,
            self.process_panel.smooth_spin,
        ]:
            spin.valueChanged.connect(self._on_pref_changed)

        for dspin in [
            self.process_panel.conf_spin,
            self.process_panel.min_area_spin,
            self.process_panel.hole_spin,
            self.process_panel.buffer_spin,
        ]:
            dspin.valueChanged.connect(self._on_pref_changed)

        self.train_panel.jobs_combo.currentIndexChanged.connect(self._on_pref_changed)
        self.process_panel.retrain_check.toggled.connect(self._on_pref_changed)

        # Shapes — salva quando a lista muda
        self.shapes_panel.list_widget.model().rowsInserted.connect(self._on_pref_changed)
        self.shapes_panel.list_widget.model().rowsRemoved.connect(self._on_pref_changed)
        self.shapes_panel.list_widget.model().dataChanged.connect(self._on_pref_changed)

    # ── Preferências ────────────────────────────────────────────────────────

    def _on_pref_changed(self):
        self.prefs.set_muitos({
            **self.files_panel.get_files(),
            "shapes_list": self.shapes_panel.get_shapes_list(),
            **self.train_panel.get_train(),
            **self.process_panel.get_process(),
        })

    def _carregar_preferencias(self):
        widgets = [
            self.files_panel.input_edit,
            self.files_panel.output_edit,
            self.train_panel.samples_spin,
            self.train_panel.trees_spin,
            self.train_panel.jobs_combo,
            self.process_panel.tile_spin,
            self.process_panel.conf_spin,
            self.process_panel.min_area_spin,
            self.process_panel.smooth_spin,
            self.process_panel.hole_spin,
            self.process_panel.buffer_spin,
            self.process_panel.retrain_check,
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.files_panel.input_edit.setText(self.prefs.get("input_tiff", ""))
            self.files_panel.output_edit.setText(self.prefs.get("output_tiff", ""))

            self.train_panel.samples_spin.setValue(self.prefs.get("samples_per_class", 60000))
            self.train_panel.trees_spin.setValue(self.prefs.get("rf_n_trees", 200))
            jobs = self.prefs.get("rf_jobs", -1)
            idx = self.train_panel.jobs_combo.findData(jobs)
            if idx >= 0:
                self.train_panel.jobs_combo.setCurrentIndex(idx)

            self.process_panel.tile_spin.setValue(self.prefs.get("tile_sz", 2048))
            self.process_panel.conf_spin.setValue(self.prefs.get("conf_threshold", 0.45))
            self.process_panel.min_area_spin.setValue(self.prefs.get("min_area_m2", 5.0))
            self.process_panel.smooth_spin.setValue(self.prefs.get("smooth_iter", 2))
            self.process_panel.hole_spin.setValue(self.prefs.get("hole_area_m2", 5.0))
            self.process_panel.buffer_spin.setValue(self.prefs.get("buffer_m", 0.1))
            self.process_panel.retrain_check.setChecked(self.prefs.get("force_retrain", False))

            # Shapes
            self.shapes_panel.set_shapes_list(self.prefs.get("shapes_list", []))
        finally:
            for w in widgets:
                w.blockSignals(False)

        self.prefs.set_muitos({
            **self.files_panel.get_files(),
            "shapes_list": self.shapes_panel.get_shapes_list(),
            **self.train_panel.get_train(),
            **self.process_panel.get_process(),
        })

    # ── Handlers ────────────────────────────────────────────────────────────

    def _on_info_clicked(self):
        dialog = SobreDialog(self)
        dialog.exec()

    def _on_test_console(self):
        self.console.write_stream(
            f"[{datetime.now().strftime('%H:%M:%S')}] 🧪 Teste de console OK\n"
        )

    def _build_config(self) -> ClassifierConfig:
        """Monta a configuração a partir dos painéis."""
        params = {
            **self.files_panel.get_files(),
            "shapes_list": self.shapes_panel.get_shapes_list(),
            **self.train_panel.get_train(),
            **self.process_panel.get_process(),
        }
        return ClassifierConfig(**params)

    def _on_process_clicked(self):
        params = {
            **self.files_panel.get_files(),
            "shapes_list": self.shapes_panel.get_shapes_list(),
            **self.train_panel.get_train(),
            **self.process_panel.get_process(),
        }

        # Validação
        if not params["input_tiff"] or not os.path.exists(params["input_tiff"]):
            msg = "❌ TIFF de entrada inválido ou inexistente."
            self.console.write_stream(f"[{self._ts()}] {msg}\n")
            self.header.set_status("ERRO", Colors.ERROR)
            self.statusBar().showMessage(msg, 4000)
            return

        if not params["output_tiff"]:
            msg = "❌ Informe o arquivo TIFF de saída."
            self.console.write_stream(f"[{self._ts()}] {msg}\n")
            self.header.set_status("ERRO", Colors.ERROR)
            self.statusBar().showMessage(msg, 4000)
            return

        if not params["shapes_list"]:
            msg = "❌ Adicione pelo menos 1 shapefile de treino."
            self.console.write_stream(f"[{self._ts()}] {msg}\n")
            self.header.set_status("ERRO", Colors.ERROR)
            self.statusBar().showMessage(msg, 4000)
            return

        config = ClassifierConfig(**params)

        self._setup_worker(config)
        self._start_time = time.time()

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(True)
        self.btn_process.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.header.set_status("PROCESSANDO", Colors.GOLD)

        print("╔══════════════════════════════════════════════════════════╗")
        print("║  INICIANDO PROCESSAMENTO                                  ║")
        print("╚══════════════════════════════════════════════════════════╝")

        self._worker_thread.start()

    def _on_stop_clicked(self):
        if self._worker is not None:
            self._worker.request_stop()
            self.btn_stop.setEnabled(False)
            print(f"[{self._ts()}] ⏹️ Interrupção solicitada…")
            self.header.set_status("INTERROMPENDO", Colors.WARNING)

    # ── Worker lifecycle ────────────────────────────────────────────────────

    def _setup_worker(self, config: ClassifierConfig):
        """Cria worker + thread e faz todo o wiring de sinais."""
        self._worker = ProcessingWorker(config)
        self._worker_thread = QThread(self)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.started.connect(self._on_worker_started)
        self._worker.log_message.connect(
            self.console.write_stream, Qt.QueuedConnection
        )
        self._worker.progress.connect(
            self._on_worker_progress, Qt.QueuedConnection
        )
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _format_time_hms(seconds: float) -> str:
        total = int(round(max(0.0, seconds)))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _on_worker_started(self):
        self.header.set_status("PROCESSANDO", Colors.GOLD)

    def _on_worker_progress(self, pct: float, msg: str):
        elapsed = time.time() - self._start_time
        self.progress_bar.setValue(int(round(pct)))

        if pct >= 100.0:
            format_text = f"{msg} · Tempo decorrido: {self._format_time_hms(elapsed)}"
        elif pct > 0:
            remaining = elapsed * (100.0 - pct) / pct
            eta_dt = datetime.now() + timedelta(seconds=remaining)
            format_text = (
                f"{msg} · Progresso: {pct:.1f}% | "
                f"Tempo decorrido: {self._format_time_hms(elapsed)} | "
                f"Restante: {self._format_time_hms(remaining)} | "
                f"Término: {eta_dt.strftime('%H:%M:%S')}"
            )
        else:
            format_text = f"{msg} | Tempo decorrido: {self._format_time_hms(elapsed)}"

        self.progress_bar.setFormat(format_text)

    def _on_worker_finished(self, success: bool, message: str):
        elapsed = time.time() - self._start_time
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setFormat(
            f"{'✅' if success else '❌'} {message} | "
            f"Tempo total: {self._format_time_hms(elapsed)}"
        )
        self.progress_bar.setVisible(True)

        if success:
            self.header.set_status("CONCLUÍDO", Colors.SUCCESS)
            self.statusBar().showMessage("✅ " + message, 5000)
        else:
            self.header.set_status("ERRO", Colors.ERROR)
            self.statusBar().showMessage("❌ " + message, 5000)

        print("╔══════════════════════════════════════════════════════════╗")
        print(f"║  {'✅ PROCESSAMENTO CONCLUÍDO' if success else '❌ PROCESSAMENTO FALHOU'}")
        print(f"║  Mensagem: {message}")
        print("╚══════════════════════════════════════════════════════════╝")

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Cleanup: interrompe thread e restaura stdout/stderr."""
        if self._worker is not None:
            self._worker.request_stop()
            self._worker_thread.quit()
            self._worker_thread.wait(5000)

        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(Styles.get_stylesheet())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()