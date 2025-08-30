from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QLabel, QListWidget, QMessageBox
from stage1 import Stage1Widget
from stage2 import Stage2Widget
from stage3 import Stage3Widget
from pyvista_viewer import PyVistaViewer

class StageMenu(QListWidget):
    def __init__(self, stages, parent=None):
        super().__init__(parent)
        self.addItems(stages)
        self.setFixedWidth(180)
        self.setCurrentRow(0)
        self.setStyleSheet('font-size: 16px;')

class MainWindow(QMainWindow):
    def get_current_mesh(self):
        # Returns the current mesh as a dict with 'vertices' and 'faces' keys
        pmesh = getattr(self.model_viewer, '_last_polymesh', None)
        if pmesh is not None:
            return {'vertices': pmesh.vertices, 'faces': pmesh.faces}
        return None
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PaperCraft Automator Desktop')
        self.setGeometry(100, 100, 1200, 800)
        stages = [
            '1. Import & Preview',
            '2. Simplify & Merge',
            '3. Build Graph',
            '4. Unfold',
            '5. Tabs & Labels',
            '6. Layout',
            '7. Export'
        ]
        self.menu = StageMenu(stages)
        self.menu.currentRowChanged.connect(self.change_stage)
        self.stage_stack = QStackedWidget()
        self.model_viewer = PyVistaViewer()
        self.stage1 = Stage1Widget(self.model_viewer)
        self.stage2 = Stage2Widget(self.model_viewer)
        # Stage 3: Build Graph & Spanning Tree
        self.stage3 = Stage3Widget(self.get_current_mesh)
        self.stage_stack.addWidget(self.stage1)
        self.stage_stack.addWidget(self.stage2)
        self.stage_stack.addWidget(self.stage3)
    # Placeholder for other stages
        for _ in range(4):
            self.stage_stack.addWidget(QLabel('Stage coming soon...'))
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.menu.setMinimumWidth(160)
        self.menu.setMaximumWidth(220)
        self.stage_stack.setMinimumWidth(260)
        self.stage_stack.setMaximumWidth(340)
        self.main_layout.addWidget(self.menu, stretch=0)
        self.main_layout.addWidget(self.stage_stack, stretch=0)
        self.viewer_container = QWidget()
        self.viewer_layout = QHBoxLayout()
        self.viewer_layout.setContentsMargins(0,0,0,0)
        self.viewer_container.setLayout(self.viewer_layout)
        # Ensure the viewer always takes at least half the window
        self.viewer_container.setMinimumWidth(600)
        self.viewer_container.setSizePolicy(self.viewer_container.sizePolicy().horizontalPolicy(), self.viewer_container.sizePolicy().verticalPolicy())
        self.main_layout.addWidget(self.viewer_container, stretch=2)
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)
        # Add the 3D viewer by default
        self.viewer_layout.addWidget(self.model_viewer)
        # Connect buttons
        self.stage1.next_btn.clicked.connect(self.next_stage)
        self.stage2.next_btn.clicked.connect(self.next_stage)
        self.stage3.next_btn.clicked.connect(self.next_stage)

    def change_stage(self, idx):
        # If leaving Stage 3 while edit mode is active, block and show popup
        if self.stage_stack.currentIndex() == 2 and self.stage3.edit_mode:
            QMessageBox.warning(self, "Finish Editing", "Please untoggle 'Edit Spanning Tree' before leaving this stage.")
            # Optionally, set focus to the edit button or flash it
            return
        self.stage_stack.setCurrentIndex(idx)
        # Swap viewer: 3D for all except Stage 3, matplotlib canvas for Stage 3
        while self.viewer_layout.count():
            item = self.viewer_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        if idx == 2:  # Stage 3
            self.viewer_layout.addWidget(self.stage3.viewer_widget)
            self.stage3.show_graph()  # Automatically show the graph
        else:
            self.viewer_layout.addWidget(self.model_viewer)

    def next_stage(self):
        idx = self.stage_stack.currentIndex()
        # If in Stage 3 and edit mode is active, block and show popup
        if idx == 2 and self.stage3.edit_mode:
            QMessageBox.warning(self, "Finish Editing", "Please untoggle 'Edit Spanning Tree' before continuing.")
            return
        if idx < self.stage_stack.count() - 1:
            self.stage_stack.setCurrentIndex(idx + 1)
            self.menu.setCurrentRow(idx + 1)
