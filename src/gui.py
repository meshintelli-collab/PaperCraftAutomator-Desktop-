from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QLabel, QListWidget
from stage1 import Stage1Widget
from stage2 import Stage2Widget
from pyvista_viewer import PyVistaViewer

class StageMenu(QListWidget):
    def __init__(self, stages, parent=None):
        super().__init__(parent)
        self.addItems(stages)
        self.setFixedWidth(180)
        self.setCurrentRow(0)
        self.setStyleSheet('font-size: 16px;')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PaperCraft Automator Desktop')
        self.setGeometry(100, 100, 1200, 800)
        stages = [
            '1. Import & Preview',
            '2. Simplify & Merge',
            '3. Build Graph',
            '4. Manual Cuts',
            '5. Unfold',
            '6. Tabs & Labels',
            '7. Layout',
            '8. Export'
        ]
        self.menu = StageMenu(stages)
        self.menu.currentRowChanged.connect(self.change_stage)
        self.stage_stack = QStackedWidget()
        self.model_viewer = PyVistaViewer()
        self.stage1 = Stage1Widget(self.model_viewer)
        self.stage2 = Stage2Widget(self.model_viewer)
        self.stage_stack.addWidget(self.stage1)
        self.stage_stack.addWidget(self.stage2)
        # Placeholder for other stages
        for _ in range(6):
            self.stage_stack.addWidget(QLabel('Stage coming soon...'))
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.menu)
        main_layout.addWidget(self.stage_stack)
        main_layout.addWidget(self.model_viewer, stretch=1)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        # Connect buttons
        self.stage1.next_btn.clicked.connect(self.next_stage)

    def change_stage(self, idx):
        self.stage_stack.setCurrentIndex(idx)

    def next_stage(self):
        idx = self.stage_stack.currentIndex()
        if idx < self.stage_stack.count() - 1:
            self.stage_stack.setCurrentIndex(idx + 1)
            self.menu.setCurrentRow(idx + 1)
