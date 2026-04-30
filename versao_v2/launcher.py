#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launcher melhorado para a UI do Classificador
"""
import sys
import os

print("\n" + "="*60)
print(" CLASSIFICADOR RASTER - INTERFACE GRÁFICA")
print("="*60)

print(f"\n[INFO] Python: {sys.version.split()[0]}")
print(f"[INFO] Diretório de trabalho: {os.getcwd()}")

try:
    print("\n[1/4] Importando dependências...")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    print("      ✓ PySide6 OK")
    
    from ui_main import MainWindow
    from core.dark_charcoal_style import DarkCharcoalStyle
    print("      ✓ UI OK")
    
    print("\n[2/4] Criando aplicação...")
    app = QApplication(sys.argv)
    app.setStyleSheet(DarkCharcoalStyle.stylesheet())
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    print("      ✓ QApplication OK")
    
    print("\n[3/4] Criando janela...")
    window = MainWindow()
    print("      ✓ MainWindow OK")
    
    print("\n[4/4] Exibindo interface...")
    window.show()
    print("      ✓ Janela visível")
    
    print("\n" + "="*60)
    print(" INTERFACE CARREGADA COM SUCESSO")
    print("="*60)
    print("\n[INFO] Clique em EXECUTAR PIPELINE para iniciar o processo")
    print("[INFO] Pressione Ctrl+C para sair\n")
    
    sys.exit(app.exec())
    
except Exception as e:
    print(f"\n❌ ERRO FATAL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
