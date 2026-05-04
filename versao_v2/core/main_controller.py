# -*- coding: utf-8 -*-
"""
Controlador Principal para a UI do Classificador Raster Neural v6
=================================================================
Logica de controle separada da view (MainWindow).
"""

import os

# Supressao de warnings do TensorFlow - deve ser configurado antes dos imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MAX_LOG_LEVEL"] = "3"

import warnings
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QTableWidgetItem, QLineEdit, QSpinBox, QPushButton, QFileDialog, QInputDialog, QMessageBox

from core.Preferences import Preferences
from core.dark_charcoal_style import DarkCharcoalStyle
from core.classifier_pipeline import ClassifierPipeline
from core.pipeline_config import PipelineConfig, PipelineConfigError

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")


class PipelineWorker(QThread):
    log = Signal(str)
    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        try:
            pipeline = ClassifierPipeline(
                config=self.config,
                logger=self.log.emit,
                progress_callback=self._emit_progress,
            )
            pipeline.execute()
            self.finished.emit("Pipeline concluido com sucesso")
        except Exception as exc:
            self.error.emit(str(exc))

    def _emit_progress(self, percent: int, message: str) -> None:
        self.progress.emit(percent, message)


class MainController:
    def __init__(self, view):
        self.view = view
        self.preferences = Preferences(Path("config") / "preferences.json")
        self.worker = None

        self._connect_signals()
        self._init_defaults()
        self.loadpreferences()
        self._update_resumo()

    def _connect_signals(self):
        self.view.btn_executar.clicked.connect(self._on_executar)
        self.view.btn_load_cfg.clicked.connect(self._on_load_cfg)
        self.view.btn_save_cfg.clicked.connect(self._on_save_cfg)
        self.view.btn_add_shp.clicked.connect(self._on_add_shp)
        self.view.combo_model_action.currentTextChanged.connect(self._on_model_action_changed)
        self.view.btn_listar_modelos.clicked.connect(self._on_listar_modelos)

        widgets_bind = [
            self.view.row_img_treino.edit,
            self.view.row_img_classif.edit,
            self.view.row_img_saida.edit,
            self.view.edit_camadas,
            self.view.combo_ativacao,
            self.view.spin_dropout,
            self.view.spin_epochs,
            self.view.spin_batch_train,
            self.view.spin_batch_pred,
            self.view.spin_test_size,
            self.view.spin_ram,
            self.view.chk_mascara,
            self.view.chk_salvar_modelo,
            self.view.combo_model_action,
            self.view.spin_random,
            self.view.spin_alpha,
            self.view.row_modelo_path.edit,
            self.view.row_modelo_existente.edit,
        ]
        for w in widgets_bind:
            if hasattr(w, "textChanged"):
                w.textChanged.connect(self._update_resumo)
                w.textChanged.connect(self.savepreferences)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self._update_resumo)
                w.currentTextChanged.connect(self.savepreferences)
            elif hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._update_resumo)
                w.valueChanged.connect(self.savepreferences)
            elif hasattr(w, "stateChanged"):
                w.stateChanged.connect(self._update_resumo)
                w.stateChanged.connect(self.savepreferences)

    def _init_defaults(self):
        default_shps = [
            ("dados/solo.shp", 0, "Solo"),
            ("dados/floresta.shp", 1, "Floresta"),
            ("dados/palhada.shp", 2, "Palhada"),
            ("dados/daninhas.shp", 3, "Daninhas"),
        ]
        for p, c, legenda in default_shps:
            self._add_shp_row(p, c, legenda)

    def _add_shp_row(self, path: str, classe: int, legenda: str = ""):
        row = self.view.table_shp.rowCount()
        self.view.table_shp.insertRow(row)

        item_path = QTableWidgetItem(path)
        item_path.setFlags(item_path.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.view.table_shp.setItem(row, 0, item_path)

        spin_cls = QSpinBox()
        spin_cls.setRange(0, 999)
        spin_cls.setValue(classe)
        spin_cls.setStyleSheet("background-color: transparent; border: none;")
        spin_cls.valueChanged.connect(self.savepreferences)
        self.view.table_shp.setCellWidget(row, 1, spin_cls)

        edit_legenda = QLineEdit(legenda)
        edit_legenda.setPlaceholderText("Legenda da classe...")
        edit_legenda.textChanged.connect(self.savepreferences)
        self.view.table_shp.setCellWidget(row, 2, edit_legenda)

        btn_rem = QPushButton("Remover")
        btn_rem.setObjectName("btn_danger")
        btn_rem.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_rem.clicked.connect(lambda _, r=row: self._remove_shp_row(r))
        self.view.table_shp.setCellWidget(row, 3, btn_rem)

    def _remove_shp_row(self, row: int):
        self.view.table_shp.removeRow(row)
        for r in range(self.view.table_shp.rowCount()):
            btn = self.view.table_shp.cellWidget(r, 3)
            if btn:
                try:
                    btn.clicked.disconnect()
                except Exception:
                    pass
                btn.clicked.connect(lambda _, nr=r: self._remove_shp_row(nr))
        self._update_resumo()
        self.savepreferences()

    def _on_add_shp(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Adicionar Shapefile", "", "Shapefile (*.shp)"
        )
        if path:
            max_cls = -1
            for r in range(self.view.table_shp.rowCount()):
                w = self.view.table_shp.cellWidget(r, 1)
                if isinstance(w, QSpinBox):
                    max_cls = max(max_cls, w.value())
            default_legend = Path(path).stem
            self._add_shp_row(path, max_cls + 1, default_legend)
            self._update_resumo()
            self.savepreferences()

    def _on_model_action_changed(self):
        action = self.view.combo_model_action.currentText()
        show_existing = action in ["Treinar modelo existente", "Usar modelo existente"]
        self.view.row_modelo_existente.setVisible(show_existing)
        self.view.btn_listar_modelos.setVisible(show_existing)
        self._update_resumo()

    def _on_listar_modelos(self):
        model_root = Path("models")
        if not model_root.exists():
            QMessageBox.information(self.view, "Modelos", "Pasta 'models' nao encontrada.")
            return

        model_files = [p for p in model_root.rglob("*.keras") if p.is_file()]
        if not model_files:
            QMessageBox.information(self.view, "Modelos", "Nenhum modelo .keras encontrado em 'models'.")
            return

        model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        options = []
        for p in model_files:
            modified_at = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            options.append(f"{p} | modificado: {modified_at}")

        selected, ok = QInputDialog.getItem(
            self.view,
            "Selecionar Modelo",
            "Modelos (mais recentes primeiro):",
            options,
            0,
            False,
        )
        if not ok or not selected:
            return

        selected_path = selected.split(" | modificado: ")[0]
        self.view.row_modelo_existente.edit.setText(selected_path)
        self._append_log(f"> Modelo selecionado: {selected_path}")
        self.savepreferences()

    def _update_resumo(self):
        treino = self.view.row_img_treino.path() or "-"
        classif = self.view.row_img_classif.path() or "-"
        saida = self.view.row_img_saida.path() or "-"
        camadas = self.view.edit_camadas.text() or "-"
        ativ = self.view.combo_ativacao.currentText()
        drop = self.view.spin_dropout.value()
        ep = self.view.spin_epochs.value()
        bt = self.view.spin_batch_train.value()
        bp = self.view.spin_batch_pred.value()
        ram = self.view.spin_ram.value()
        mask = "Sim" if self.view.chk_mascara.isChecked() else "Nao"

        model_action = self.view.combo_model_action.currentText()
        resumo = (
            f"<b>Imagem Treino:</b> {treino}<br>"
            f"<b>Imagem Classif.:</b> {classif}<br>"
            f"<b>Saida:</b> {saida}<br>"
            f"<b>Acao Modelo:</b> {model_action}<br>"
        )
        if self.view.row_modelo_existente.isVisible():
            existing_model = self.view.row_modelo_existente.path() or "-"
            resumo += f"<b>Modelo Existente:</b> {existing_model}<br>"
        resumo += (
            f"<b>Rede:</b> [{camadas}] - ativacao {ativ}, dropout {drop}<br>"
            f"<b>Treino:</b> {ep} epocas | batch {bt} / pred {bp}<br>"
            f"<b>RAM limite:</b> {ram}% | Mascara: {mask}"
        )
        self.view.lbl_resumo.setHtml(resumo)

    def _on_executar(self):
        if self.worker is not None and self.worker.isRunning():
            self._append_log("> O pipeline ja esta em execucao")
            return

        pipeline_data = self.get_pipeline_config()
        try:
            config = PipelineConfig.from_dict(pipeline_data)
        except PipelineConfigError as exc:
            self._append_log(f"> Configuracao invalida: {exc}")
            return

        self.savepreferences()
        self._append_log("> Pipeline iniciado")
        self._set_running_state(True)
        self.worker = PipelineWorker(config)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._on_progress_update)
        self.worker.finished.connect(self._on_pipeline_finished)
        self.worker.error.connect(self._on_pipeline_error)
        self.worker.start()

    def _on_load_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Carregar Configuracao", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            config = PipelineConfig.load(Path(path))
            self._populate_fields(config)
            self._append_log(f"> Configuracao carregada: {path}")
            self.savepreferences()
        except Exception as exc:
            self._append_log(f"> Falha ao carregar configuracao: {exc}")

    def _on_save_cfg(self):
        path, _ = QFileDialog.getSaveFileName(
            self.view, "Salvar Configuracao", "config_ui.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            config = PipelineConfig.from_dict(self.get_pipeline_config())
            config.save(Path(path))
            self._append_log(f"> Configuracao salva: {path}")
        except Exception as exc:
            self._append_log(f"> Falha ao salvar configuracao: {exc}")

    def _append_log(self, message: str) -> None:
        self.view.txt_log.append(str(message))

    def _set_running_state(self, running: bool) -> None:
        self.view.btn_executar.setEnabled(not running)
        self.view.btn_load_cfg.setEnabled(not running)
        self.view.btn_save_cfg.setEnabled(not running)
        if running:
            self.view.badge_status.setText("EXECUTANDO")
            self.view.badge_status.setStyleSheet(
                "QLabel {"
                f"  background-color: {DarkCharcoalStyle.WARNING};"
                f"  color: {DarkCharcoalStyle.DARK_BG};"
                "  border-radius: 6px;"
                "  padding: 4px 14px;"
                "  font-weight: 700;"
                "  font-size: 11px;"
                "}"
            )
        else:
            self.view.badge_status.setText("PRONTA")
            self.view.badge_status.setStyleSheet(
                "QLabel {"
                f"  background-color: {DarkCharcoalStyle.SUCCESS};"
                f"  color: {DarkCharcoalStyle.DARK_BG};"
                "  border-radius: 6px;"
                "  padding: 4px 14px;"
                "  font-weight: 700;"
                "  font-size: 11px;"
                "}"
            )

    def _on_progress_update(self, percent: int, message: str) -> None:
        self.view.progress.setValue(min(max(percent, 0), 100))
        self.view.progress.setFormat(f" {percent}% - {message} ")

    def _on_pipeline_finished(self, message: str) -> None:
        self._append_log(f"> {message}")
        self._set_running_state(False)
        self.view.progress.setValue(100)
        self.view.progress.setFormat(" 100% - concluido ")

    def _on_pipeline_error(self, message: str) -> None:
        self._append_log(f"> ERRO: {message}")
        self._set_running_state(False)
        self.view.badge_status.setText("ERRO")
        self.view.badge_status.setStyleSheet(
            "QLabel {"
            f"  background-color: {DarkCharcoalStyle.DANGER};"
            f"  color: {DarkCharcoalStyle.DARK_BG};"
            "  border-radius: 6px;"
            "  padding: 4px 14px;"
            "  font-weight: 700;"
            "  font-size: 11px;"
            "}"
        )

    def _populate_fields(self, config: PipelineConfig) -> None:
        self.view.row_img_treino.edit.setText(str(config.training_image))
        self.view.row_img_classif.edit.setText(str(config.classification_image))
        self.view.row_img_saida.edit.setText(str(config.output_path))
        self.view.edit_camadas.setText(", ".join(str(layer) for layer in config.hidden_layers))
        self.view.combo_ativacao.setCurrentText(config.activation)
        self.view.spin_dropout.setValue(config.dropout_rate)
        self.view.spin_epochs.setValue(config.epochs)
        self.view.spin_batch_train.setValue(config.batch_size_train)
        self.view.spin_batch_pred.setValue(config.batch_size_pred)
        self.view.spin_test_size.setValue(config.test_size)
        self.view.spin_random.setValue(config.random_state)
        self.view.spin_ram.setValue(config.ram_limit_pct)
        self.view.chk_mascara.setChecked(config.use_mask)
        self.view.spin_alpha.setValue(config.alpha_threshold)
        self.view.chk_salvar_modelo.setChecked(config.save_model)
        self.view.row_modelo_path.edit.setText(str(config.model_path))
        self.view.combo_model_action.setCurrentText(config.model_action)
        self.view.row_modelo_existente.edit.setText(str(config.existing_model_path or ""))

        self.view.table_shp.setRowCount(0)
        for entry in config.shapefiles:
            self._add_shp_row(str(entry.path), int(entry.class_id), str(entry.legend or ""))

        self._on_model_action_changed()
        self.savepreferences()

    def loadpreferences(self) -> None:
        self.preferences.loadpreferences()

        self.view.row_img_treino.edit.setText(str(self.preferences.get("training_image", self.view.row_img_treino.path())))
        self.view.row_img_classif.edit.setText(str(self.preferences.get("classification_image", self.view.row_img_classif.path())))
        self.view.row_img_saida.edit.setText(str(self.preferences.get("output", self.view.row_img_saida.path())))
        self.view.edit_camadas.setText(str(self.preferences.get("hidden_layers", self.view.edit_camadas.text())))
        self.view.combo_ativacao.setCurrentText(str(self.preferences.get("activation", self.view.combo_ativacao.currentText())))
        self.view.spin_dropout.setValue(float(self.preferences.get("dropout_rate", self.view.spin_dropout.value())))
        self.view.spin_epochs.setValue(int(self.preferences.get("epochs", self.view.spin_epochs.value())))
        self.view.spin_batch_train.setValue(int(self.preferences.get("batch_size_train", self.view.spin_batch_train.value())))
        self.view.spin_batch_pred.setValue(int(self.preferences.get("batch_size_pred", self.view.spin_batch_pred.value())))
        self.view.spin_test_size.setValue(float(self.preferences.get("test_size", self.view.spin_test_size.value())))
        self.view.spin_random.setValue(int(self.preferences.get("random_state", self.view.spin_random.value())))
        self.view.spin_ram.setValue(int(self.preferences.get("ram_limit_pct", self.view.spin_ram.value())))
        self.view.chk_mascara.setChecked(bool(self.preferences.get("use_mask", self.view.chk_mascara.isChecked())))
        self.view.spin_alpha.setValue(int(self.preferences.get("alpha_threshold", self.view.spin_alpha.value())))
        self.view.chk_salvar_modelo.setChecked(bool(self.preferences.get("save_model", self.view.chk_salvar_modelo.isChecked())))
        self.view.row_modelo_path.edit.setText(str(self.preferences.get("model_path", self.view.row_modelo_path.path())))
        self.view.combo_model_action.setCurrentText(str(self.preferences.get("model_action", self.view.combo_model_action.currentText())))
        self.view.row_modelo_existente.edit.setText(str(self.preferences.get("existing_model_path", self.view.row_modelo_existente.path())))

        shapefiles = self.preferences.get("shapefiles", [])
        if isinstance(shapefiles, list) and shapefiles:
            self.view.table_shp.setRowCount(0)
            for item in shapefiles:
                if isinstance(item, dict) and "path" in item and "class_id" in item:
                    self._add_shp_row(str(item["path"]), int(item["class_id"]), str(item.get("legend", "")))

        self._on_model_action_changed()
        self._update_resumo()

    def savepreferences(self) -> None:
        self.preferences.savepreferences(self.get_pipeline_config())

    def get_shapefile_entries(self):
        entries = []
        for row in range(self.view.table_shp.rowCount()):
            path_item = self.view.table_shp.item(row, 0)
            cls_widget = self.view.table_shp.cellWidget(row, 1)
            legend_widget = self.view.table_shp.cellWidget(row, 2)
            if path_item and isinstance(cls_widget, QSpinBox):
                legend = ""
                if isinstance(legend_widget, QLineEdit):
                    legend = legend_widget.text().strip()
                entries.append({
                    "path": path_item.text(),
                    "class_id": cls_widget.value(),
                    "legend": legend,
                })
        return entries

    def get_output_path(self):
        return self.view.row_img_saida.path()

    def get_model_action(self):
        return self.view.combo_model_action.currentText()

    def get_pipeline_config(self):
        return {
            "shapefiles": self.get_shapefile_entries(),
            "output": self.get_output_path(),
            "training_image": self.view.row_img_treino.path(),
            "classification_image": self.view.row_img_classif.path(),
            "model_action": self.get_model_action(),
            "save_model": self.view.chk_salvar_modelo.isChecked(),
            "model_path": self.view.row_modelo_path.path(),
            "existing_model_path": self.view.row_modelo_existente.path() if self.view.row_modelo_existente.isVisible() else None,
            "test_size": self.view.spin_test_size.value(),
            "random_state": self.view.spin_random.value(),
            "epochs": self.view.spin_epochs.value(),
            "batch_size_train": self.view.spin_batch_train.value(),
            "batch_size_pred": self.view.spin_batch_pred.value(),
            "hidden_layers": self.view.edit_camadas.text(),
            "activation": self.view.combo_ativacao.currentText(),
            "dropout_rate": self.view.spin_dropout.value(),
            "use_mask": self.view.chk_mascara.isChecked(),
            "alpha_threshold": self.view.spin_alpha.value(),
            "ram_limit_pct": self.view.spin_ram.value(),
        }
