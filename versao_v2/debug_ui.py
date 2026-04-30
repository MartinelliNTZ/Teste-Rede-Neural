#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de debug para testar a UI
"""
import sys
import traceback

def main():
    try:
        from PySide6.QtWidgets import QApplication
        from ui_main import MainWindow
        
        app = QApplication(sys.argv)
        
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"[ERRO] Exceção: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
