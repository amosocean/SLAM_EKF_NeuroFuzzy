#!/usr/bin/python

"""
ZetCode PyQt6 tutorial

In this example, we create a simple
window in PyQt6.

Author: Jan Bodnar
Website: zetcode.com
"""


import sys
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtWidgets import QApplication, QWidget
import torch
from sources.qtUIs.test import Ui_Dialog
"""
一些gui界面的小实验
"""
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow,self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

if __name__ =="__main__":
    print("注册界面")
    import sys
    app=QtWidgets.QApplication(sys.argv)
    mainWindowOriginal=QtWidgets.QDialog()
    second_ui=Ui_Dialog()
    second_ui.setupUi(mainWindowOriginal)
    mainWindowOriginal.show()
    sys.exit(app.exec())



