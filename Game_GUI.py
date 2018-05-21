import sys
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel,
    QFrame, QApplication, QGridLayout
)
from PyQt5.QtGui import QGuiApplication, QColor, QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer

from GameData import GameData


class GameGui(QWidget):
    game_over = False
    button_style = 'QWidget { background-color: %s; border-radius: %spx; font: %spx; color: %s; padding: %spx;}'
    style = 'QWidget { background-color: %s; border-radius: %spx}'

    main_color = QColor(187, 173, 160)
    new_game_color = QColor(143, 122, 102)
    number_color = QColor(119, 110, 101)
    main_text_color = QColor(248, 246, 242)
    empty_cell = QColor(205, 193, 180)
    frames_color = {
        0: QColor(238, 228, 218),
        2: QColor(238, 228, 218),
        4: QColor(237, 224, 200),
        8: QColor(242, 177, 121),
        16: QColor(245, 149, 99),
        32: QColor(246, 124, 95),
        64: QColor(246, 94, 59),
        128: QColor(237, 207, 114),
        256: QColor(237, 204, 97),
        512: QColor(237, 200, 80),
        1024: QColor(237, 197, 63),
        2048: QColor(237, 194, 46)
    }
    frames_color_other = QColor(60, 58, 50)
    moves = {
        '87': 0,
        '68': 1,
        '83': 2,
        '65': 3,
        '1062': 0,
        '1042': 1,
        '1067': 2,
        '1060': 3
    }

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.initUI()

    def initUI(self):
        self.game_data = GameData()

        self.new_game_button = QPushButton('Новая Игра', self)
        self.new_game_button.setStyleSheet(self.button_style % (self.new_game_color.name(), 5, 20, self.main_text_color.name(), 10))
        self.new_game_button.move(10, 50)

        self.new_game_button.clicked.connect(self.new_game)

        self.square = QFrame(self)
        self.square.setGeometry(0, 100, 380, 380)
        self.square.setStyleSheet(self.style % (self.main_color.name(), 5))

        self.result_label = QLabel('Результат\n0', self)
        self.crt_progress_label = QLabel('За последний ход\n0', self)
        self.moves_counter = QLabel('Ходов сделано\n0', self)

        self.result_label.setStyleSheet(self.button_style % (self.main_color.name(), 5, 20, self.main_text_color.name(), 5))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.crt_progress_label.setStyleSheet(self.button_style % (self.main_color.name(), 5, 20, self.main_text_color.name(), 5))
        self.crt_progress_label.setAlignment(Qt.AlignCenter)
        self.moves_counter.setStyleSheet(self.button_style % (self.main_color.name(), 5, 20, self.main_text_color.name(), 5))
        self.moves_counter.setAlignment(Qt.AlignCenter)

        self.result_label.move(400, 100)
        self.crt_progress_label.move(400, 180)
        self.moves_counter.move(400, 260)

        self.image_w = QPixmap(r'images\w.png')
        self.image_d = QPixmap(r'images\d.png')
        self.image_s = QPixmap(r'images\s.png')
        self.image_a = QPixmap(r'images\a.png')
        self.image_w_red = QPixmap(r'images\w_red.png')
        self.image_d_red = QPixmap(r'images\d_red.png')
        self.image_s_red = QPixmap(r'images\s_red.png')
        self.image_a_red = QPixmap(r'images\a_red.png')
        self.label_w = QLabel(self)
        self.label_d = QLabel(self)
        self.label_s = QLabel(self)
        self.label_a = QLabel(self)

        self.label_w.setPixmap(self.image_w)
        self.label_d.setPixmap(self.image_d)
        self.label_s.setPixmap(self.image_s)
        self.label_a.setPixmap(self.image_a)

        self.label_w.move(474, 374)
        self.label_d.move(528, 426)
        self.label_s.move(474, 426)
        self.label_a.move(420, 426)

        self.grid = QGridLayout()
        self.grid.setSpacing(4)
        self.grid.setContentsMargins(4, 4, 4, 4)

        self.square_go = QFrame(self)
        self.square_go.setGeometry(0, 100, 380, 380)
        self.square_go.setStyleSheet('QWidget { background-color: rgba(240, 240, 240, 90); border-radius: 5px}')
        self.square_go.setVisible(False)

        self.game_over_label = QLabel('GAME OVER', self)
        self.game_over_label.setStyleSheet(self.button_style % (self.main_color.name(), 5, 20, self.main_text_color.name(), 10))
        self.game_over_label.setAlignment(Qt.AlignCenter)
        self.game_over_label.setVisible(False)
        self.game_over_label.move(130, 270)

        self.square.setLayout(self.grid)
        self.refresh_grid()

        self.timer = QTimer()

        ico = QIcon(r'images\2048_icon.ico')
        self.setWindowIcon(ico)
        self.setFixedSize(640, 480)
        self.setWindowTitle('2048')
        self.show()

    def new_game(self):
        if self.game_over:
            self.square_go.setVisible(False)
            self.game_over_label.setVisible(False)
            self.game_over = False

        self.game_data.refresh()
        self.refresh_grid()
        self.result_label.setText('Результат\n0')
        self.crt_progress_label.setText('За текущий ход\n0')
        self.moves_counter.setText('Ходов сделано\n0')
        self.result_label.update()
        self.crt_progress_label.update()
        self.moves_counter.update()
        return

    def refresh_grid(self):
        field = self.game_data.get_field()

        for i in range(4):
            for j in range(4):
                cell = field[i][j]
                num = cell.get_number()

                label = QLabel('{}'.format(num if num != 0 else ''))
                label_col = self.frames_color_other if num > 2048 else self.frames_color[num]
                text_color = self.number_color if num < 8 else self.main_text_color
                if num == 0:
                    label_col = self.empty_cell
                label.setStyleSheet(self.button_style % (label_col.name(), 3, 30, text_color.name(), 10))
                label.setAlignment(Qt.AlignCenter)
                self.grid.addWidget(label, i, j)
        self.square.update()
        return

    def keyPressEvent(self, event):
        key = str(event.key())
        try:
            move = self.moves[key]
            if move == 0:
                self.label_w.setPixmap(self.image_w_red)
                self.label_w.update()
                self.timer.timeout.connect(self.time_out_w)
                self.timer.start(50)
            elif move == 1:
                self.label_d.setPixmap(self.image_d_red)
                self.label_d.update()
                self.timer.timeout.connect(self.time_out_d)
                self.timer.start(50)
            elif move == 2:
                self.label_s.setPixmap(self.image_s_red)
                self.label_s.update()
                self.timer.timeout.connect(self.time_out_s)
                self.timer.start(50)
            else:
                self.label_a.setPixmap(self.image_a_red)
                self.label_a.update()
                self.timer.timeout.connect(self.time_out_a)
                self.timer.start(50)

            if self.game_over:
                return

            check = self.game_data.move(move)
            if not check:
                return

            self.game_data.rand_cell()
            self.result_label.setText('Результат\n{}'.format(self.game_data.progress))
            self.crt_progress_label.setText('За текущий ход\n{}'.format(self.game_data.crt_progress))
            self.moves_counter.setText('Ходов сделано\n{}'.format(self.game_data.moves))
            self.result_label.update()
            self.crt_progress_label.update()
            self.moves_counter.update()
            self.game_data.crt_progress = 0
            self.refresh_grid()

            if self.game_data.check_GameOver():
                self.game_over = True
                self.square_go.setVisible(True)
                self.game_over_label.setVisible(True)
            return
        except KeyError:
            return

    def time_out_w(self):
        self.timer.stop()
        self.label_w.setPixmap(self.image_w)
        self.label_w.update()
        return

    def time_out_d(self):
        self.timer.stop()
        self.label_d.setPixmap(self.image_d)
        self.label_d.update()
        return

    def time_out_s(self):
        self.timer.stop()
        self.label_s.setPixmap(self.image_s)
        self.label_s.update()
        return

    def time_out_a(self):
        self.timer.stop()
        self.label_a.setPixmap(self.image_a)
        self.label_a.update()
        return

    def closeEvent(self, event):
        self.hide()
        event.accept()
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GameGui()
    sys.exit(app.exec_())
