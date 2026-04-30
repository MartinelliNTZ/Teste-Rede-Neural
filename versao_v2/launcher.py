#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launcher para a UI do Classificador Raster Neural v6
==================================================
Inicia a aplicacao Qt com suppressao de warnings do TensorFlow.
"""

import os
import sys

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACOES INICIAIS — DEVEM VIR ANTES DE QUALQUER IMPORT
# ═══════════════════════════════════════════════════════════════════════════

# Suprime warnings do TensorFlow (deve ser definido antes do primeiro import)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desabilita oneDNN custom operations
os.environ["TF_CPP_MAX_LOG_LEVEL"] = "3"  # Alternativo para algumas versoes

# Suprime warnings de deprecacao de bibliotecas externas
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# ═══════════════════════════════════════════════════════════════════════
# INICIA APLICACAO
# ═══════════════════════════════════════════════════════════════════════

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    
    from ui_main import MainWindow
    from core.dark_charcoal_style import DarkCharcoalStyle
    
    app = QApplication(sys.argv)
    app.setStyleSheet(DarkCharcoalStyle.stylesheet())
    
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
    
except Exception as e:
    print(f"\n[ERRO] Falha ao iniciar: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
