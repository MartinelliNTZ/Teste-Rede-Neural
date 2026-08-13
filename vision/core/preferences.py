# -*- coding: utf-8 -*-
"""Sistema de preferências persistidas em JSON — agnóstico de UI."""

import os
import json


class Preferences:
    """Gerencia preferências do usuário, persistidas automaticamente em JSON."""

    ARQUIVO_PREFERENCIAS = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "preferences.json",
    )

    # ── Defaults compartilhados (globais) ──────────────────────────────────
    DEFAULTS_COMPARTILHADAS = {
        "data_dir": "1-AETHERIS_CLASSIFIER_",
        "out_dir": "1-AETHERIS_CLASSIFIER_output",
    }

    # ── Defaults da ferramenta (classificador) ─────────────────────────────
    DEFAULTS_FERRAMENTA = {
        # Classes / shapefiles
        "shape_solo": "solo.shp",
        "shape_palhada": "palhada.shp",
        "shape_vegetacao": "vegetacao.shp",
        "shape_outros": "",

        # Treino
        "samples_per_class": 60000,
        "rf_n_trees": 200,
        "rf_jobs": -1,

        # Predição
        "tile_sz": 2048,
        "conf_threshold": 0.45,

        # Vetorização
        "min_area_m2": 5.0,
        "smooth_iter": 2,
        "hole_area_m2": 5.0,
        "buffer_m": 0.1,

        # Comportamento
        "force_retrain": False,
    }

    def __init__(self, namespace: str | None = None):
        self._namespace = namespace or "classificador"
        self._dados = {}
        self._carregar()

    # ── Internos ───────────────────────────────────────────────────────────

    def _carregar(self):
        """Lê o JSON e mescla com defaults. Arquivo corrompido → defaults."""
        dados = {}
        try:
            if os.path.exists(self.ARQUIVO_PREFERENCIAS):
                with open(self.ARQUIVO_PREFERENCIAS, "r", encoding="utf-8") as f:
                    dados = json.load(f)
        except (json.JSONDecodeError, OSError):
            dados = {}

        # Migração: arquivo antigo sem estrutura de ferramentas → compartilhadas
        if "compartilhadas" not in dados and "ferramentas" not in dados:
            dados = {"compartilhadas": dados, "ferramentas": {}}

        self._dados = dados

        # Garante que a seção da ferramenta existe com todos os defaults
        ferramentas = self._dados.setdefault("ferramentas", {})
        ferramenta = ferramentas.setdefault(self._namespace, {})

        # Migração: remove chaves antigas em UPPERCASE (versão anterior)
        for chave in list(ferramenta.keys()):
            if chave != chave.lower():
                del ferramenta[chave]

        for chave, valor in self.DEFAULTS_FERRAMENTA.items():
            ferramenta.setdefault(chave, valor)

        # Garante defaults compartilhados
        compartilhadas = self._dados.setdefault("compartilhadas", {})
        for chave in list(compartilhadas.keys()):
            if chave != chave.lower():
                del compartilhadas[chave]
        for chave, valor in self.DEFAULTS_COMPARTILHADAS.items():
            compartilhadas.setdefault(chave, valor)

        self._salvar()

    def _salvar(self):
        """Persiste o JSON em disco."""
        try:
            with open(self.ARQUIVO_PREFERENCIAS, "w", encoding="utf-8") as f:
                json.dump(self._dados, f, ensure_ascii=False, indent=2)
        except OSError:
            pass  # silencioso — não deve quebrar a UI

    # ── API pública ────────────────────────────────────────────────────────

    def get(self, chave, padrao=None):
        """Prioridade: ferramenta → compartilhada → padrão."""
        ferramenta = self._dados.get("ferramentas", {}).get(self._namespace, {})
        if chave in ferramenta:
            return ferramenta[chave]
        compartilhadas = self._dados.get("compartilhadas", {})
        if chave in compartilhadas:
            return compartilhadas[chave]
        return padrao

    def set(self, chave, valor):
        """Salva um valor na seção da ferramenta e persiste."""
        ferramenta = self._dados.setdefault("ferramentas", {}).setdefault(
            self._namespace, {}
        )
        ferramenta[chave] = valor
        self._salvar()

    def set_muitos(self, valores: dict):
        """Salva vários valores de uma vez e persiste."""
        ferramenta = self._dados.setdefault("ferramentas", {}).setdefault(
            self._namespace, {}
        )
        ferramenta.update(valores)
        self._salvar()

    def reset(self):
        """Restaura defaults e salva."""
        self._dados = {
            "compartilhadas": dict(self.DEFAULTS_COMPARTILHADAS),
            "ferramentas": {
                self._namespace: dict(self.DEFAULTS_FERRAMENTA),
            },
        }
        self._salvar()

    def to_dict(self) -> dict:
        """Cópia (compartilhadas + ferramenta)."""
        ferramenta = self._dados.get("ferramentas", {}).get(self._namespace, {})
        return {
            **self._dados.get("compartilhadas", {}),
            **ferramenta,
        }