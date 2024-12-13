from PyQt5.QtWidgets import QApplication, QLabel, QWidget

app = QApplication([])
window = QWidget()
window.setWindowTitle('PyQt5 Test')
label = QLabel('PyQt5 is working!', parent=window)
label.move(60, 15)
window.setGeometry(100, 100, 280, 80)
window.show()
app.exec_()