# -*- coding: utf-8 -*-
"""
Controlador Principal para a UI do Classificador Raster Neural v6
=================================================================
Logica de controle separada da view (MainWindow).
"""

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidgetItem, QLineEdit, QSpinBox, QPushButton, QFileDialog
from core.dark_charcoal_style import DarkCharcoalStyle


class MainController:
    def __init__(self, view):
        self.view = view
        self._connect_signals()
        self._init_defaults()
        self._update_resumo()

    def _connect_signals(self):
        self.view.btn_executar.clicked.connect(self._on_executar)
        self.view.btn_load_cfg.clicked.connect(self._on_load_cfg)
        self.view.btn_save_cfg.clicked.connect(self._on_save_cfg)
        self.view.btn_add_shp.clicked.connect(self._on_add_shp)
        self.view.combo_model_action.currentTextChanged.connect(self._on_model_action_changed)

        widgets_bind = [
            self.view.row_img_treino.edit, self.view.row_img_classif.edit,
            self.view.row_img_saida.edit, self.view.edit_camadas,
            self.view.combo_ativacao, self.view.spin_dropout,
            self.view.spin_epochs, self.view.spin_batch_train,
            self.view.spin_batch_pred, self.view.spin_test_size,
            self.view.spin_ram, self.view.chk_mascara, self.view.chk_salvar_modelo,
            self.view.combo_model_action
        ]
        for w in widgets_bind:
            if hasattr(w, "textChanged"):
                w.textChanged.connect(self._update_resumo)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self._update_resumo)
            elif hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._update_resumo)
            elif hasattr(w, "stateChanged"):
                w.stateChanged.connect(self._update_resumo)

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
        self.view.table_shp.setCellWidget(row, 1, spin_cls)

        edit_legenda = QLineEdit(legenda)
        edit_legenda.setPlaceholderText("Legenda da classe...")
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

    def _on_model_action_changed(self):
        action = self.view.combo_model_action.currentText()
        show_existing = action in ["Treinar modelo existente", "Usar modelo existente"]
        self.view.row_modelo_existente.setVisible(show_existing)
        self._update_resumo()

    def _update_resumo(self):
        treino = self.view.row_img_treino.path() or "—"
        classif = self.view.row_img_classif.path() or "—"
        saida = self.view.row_img_saida.path() or "—"
        camadas = self.view.edit_camadas.text() or "—"
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
            existing_model = self.view.row_modelo_existente.path() or "—"
            resumo += f"<b>Modelo Existente:</b> {existing_model}<br>"
        resumo += (
            f"<b>Rede:</b> [{camadas}] — ativacao {ativ}, dropout {drop}<br>"
            f"<b>Treino:</b> {ep} epocas | batch {bt} / pred {bp}<br>"
            f"<b>RAM limite:</b> {ram}% | Mascara: {mask}"
        )
        self.view.lbl_resumo.setHtml(resumo)

    def _on_executar(self):
        self.view.txt_log.append("> Pipeline iniciado... [STUB — integrar logica posteriormente]")
        self.view.progress.setValue(10)
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

    def _on_load_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Carregar Configuracao", "", "JSON (*.json)"
        )
        if path:
            self.view.txt_log.append(f"> Config carregada: {path}  [STUB]")

    def _on_save_cfg(self):
        path, _ = QFileDialog.getSaveFileName(
            self.view, "Salvar Configuracao", "config_ui.json", "JSON (*.json)"
        )
        if path:
            self.view.txt_log.append(f"> Config salva: {path}  [STUB]")

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
        config = {
            "shapefiles": self.get_shapefile_entries(),
            "output": self.get_output_path(),
            "training_image": self.view.row_img_treino.path(),
            "classification_image": self.view.row_img_classif.path(),
            "model_action": self.get_model_action(),
            "save_model": self.view.chk_salvar_modelo.isChecked(),
            "model_path": self.view.row_modelo_path.path(),
        }
        if self.view.combo_model_action.currentText() in ["Treinar modelo existente", "Usar modelo existente"]:
            config["existing_model_path"] = self.view.row_modelo_existente.path()
        return config
