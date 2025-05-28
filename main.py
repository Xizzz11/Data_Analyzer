import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import customtkinter as ctk
import io
from PIL import Image, ImageTk
from datetime import datetime
import asyncio
import threading
import logging
from tkinter import filedialog
from typing import List, Tuple, Optional
import pytz

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("minimal_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

matplotlib.use('Agg')  # Рендеринг без окна

class DataAnalyzer:
    def __init__(self, file_path: str):
        """Инициализация анализатора данных."""
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None
        self.output_dir = Path('minimal_results') / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Инициализирован DataAnalyzer с файлом: {file_path}")

    async def load_data_async(self, chunksize: int = 10000) -> str:
        """Асинхронная загрузка данных."""
        try:
            if not self.file_path.is_file():
                logger.error(f"Файл не найден: {self.file_path}")
                return f"Файл {self.file_path} не найден! 🔥 Укажите правильный путь или создайте sample_data.csv."
            if self.file_path.suffix == '.csv':
                chunks = pd.read_csv(self.file_path, chunksize=chunksize, encoding='utf-8', low_memory=False)
                self.df = await asyncio.get_event_loop().run_in_executor(None, lambda: pd.concat(chunks, ignore_index=True))
                logger.info(f"Загружены данные: {self.df.shape}, столбцы: {list(self.df.columns)}")
                return f"Данные загружены! Размер: {self.df.shape}\nСтолбцы: {list(self.df.columns)} 🚀"
            else:
                raise ValueError("Поддерживается только формат CSV")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {str(e)}")
            return f"Ошибка загрузки: {str(e)} 💎"

    def basic_analysis(self) -> str:
        """Базовый анализ данных."""
        if self.df is None:
            logger.error("Данные не загружены")  # Исправлено: убран logger.errorCullable
            return "Данные не загружены! 🔥"
        
        output = ["💎 Анализ данных:\n"]
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        output.append(buffer.getvalue())  # Исправлено: buffer.kleinenvalue() на buffer.getvalue()
        output.append("\n📊 Статистика:\n")
        output.append(self.df.describe().to_string())
        output.append("\n🚀 Пропущенные значения:\n")
        output.append(self.df.isnull().sum().to_string())
        
        analysis_path = self.output_dir / 'minimal_analysis.txt'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output))
        logger.info(f"Анализ сохранён в: {analysis_path}")
        
        return "\n".join(output)

    def clean_data(self) -> str:
        """Очистка данных."""
        if self.df is None:
            logger.error("Данные не загружены")
            return "Данные не загружены! 🔥"

        original_shape = self.df.shape
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                self.df[column] = self.df[column].fillna(self.df[column].median())
            else:
                mode_val = self.df[column].mode()
                self.df[column] = self.df[column].fillna(mode_val[0] if not mode_val.empty else 'Неизвестно')

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR)))
            self.df = self.df[mask]

        if self.df.empty:
            logger.warning(f"После очистки данных не осталось строк: {original_shape}")
            return f"После очистки данных не осталось строк! Исходный размер: {original_shape} 🔥"
        logger.info(f"Данные очищены: {self.df.shape}")
        return f"Данные очищены! Новый размер: {self.df.shape} 💎"

    def visualize_data(self, numeric_column: Optional[str] = None, categorical_column: Optional[str] = None, theme: str = 'dark') -> Tuple[str, List[Path]]:
        """Создание минималистичных визуализаций."""
        if self.df is None:
            logger.error("Данные не загружены")
            return "Данные не загружены! 🔥", []

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        if numeric_column and numeric_column not in self.df.columns:
            logger.warning(f"Столбец {numeric_column} не найден, выбираем первый числовой")
            numeric_column = numeric_cols[0] if len(numeric_cols) > 0 else None
        if categorical_column and categorical_column not in self.df.columns:
            logger.warning(f"Столбец {categorical_column} не найден, выбираем первый категориальный")
            categorical_column = categorical_cols[0] if len(categorical_cols) > 0 else None

        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        sns.set_palette("Greys")
        bg_color = '#FFFFFF' if theme == 'light' else '#212121'
        text_color = '#424242' if theme == 'light' else '#E0E0E0'
        plot_color = '#B0BEC5'

        image_paths: List[Path] = []

        # Гистограмма
        if numeric_column:
            try:
                fig = plt.figure(figsize=(10, 6), facecolor=bg_color)
                ax = fig.add_subplot(111)
                sns.histplot(self.df[numeric_column].dropna(), bins=30, color=plot_color, edgecolor='#424242')
                ax.set_title(f'Гистограмма {numeric_column} 📊', color=text_color, fontsize=16)
                ax.set_xlabel(numeric_column, color=text_color, fontsize=12)
                ax.set_ylabel('Частота', color=text_color, fontsize=12)
                ax.set_facecolor(bg_color)
                fig.patch.set_facecolor(bg_color)
                hist_path = self.output_dir / f'hist_{numeric_column}.png'
                plt.savefig(hist_path, bbox_inches='tight', facecolor=bg_color, dpi=300)
                plt.close()
                if hist_path.exists():
                    image_paths.append(hist_path)
                    logger.info(f"Гистограмма сохранена: {hist_path}")
            except Exception as e:
                logger.error(f"Ошибка создания гистограммы: {str(e)}")
                return f"Ошибка гистограммы: {str(e)} 🔥", []

        # Столбчатая диаграмма
        if categorical_column:
            try:
                plt.figure(figsize=(10, 6), facecolor=bg_color)
                ax = plt.gca()
                ax.set_facecolor(bg_color)
                counts = self.df[categorical_column].value_counts()
                sns.barplot(x=counts.index, y=counts.values, color=plot_color, edgecolor='#424242')
                plt.title(f'Категории {categorical_column} 💎', color=text_color, fontsize=16)
                plt.xlabel(categorical_column, color=text_color, fontsize=12)
                plt.ylabel('Количество', color=text_color, fontsize=12)
                plt.xticks(rotation=45, color=text_color)
                plt.yticks(color=text_color)
                bar_path = self.output_dir / f'bar_{categorical_column}.png'
                plt.savefig(bar_path, bbox_inches='tight', facecolor=bg_color, dpi=300)
                plt.close()
                if bar_path.exists():
                    image_paths.append(bar_path)
                    logger.info(f"Столбчатая диаграмма сохранена: {bar_path}")
            except Exception as e:
                logger.error(f"Ошибка создания столбчатой диаграммы: {str(e)}")
                return f"Ошибка столбчатой диаграммы: {str(e)} 🔥", []

        # Корреляционная матрица
        if len(numeric_cols) > 1:
            try:
                fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
                ax = plt.gca()
                ax.set_facecolor(bg_color)
                sns.heatmap(
                    self.df[numeric_cols].corr(), annot=True, cmap='Greys',
                    annot_kws={'color': text_color, 'fontsize': 10},
                    cbar_kws={'label': 'Корреляция', 'ticks': [-1, 0, 1]},
                    linewidths=0.5, linecolor='#424242'
                )
                plt.title('Корреляция 📈', color=text_color, fontsize=16)
                plt.tick_params(colors=text_color)
                corr_path = self.output_dir / 'correlation_matrix.png'
                plt.savefig(corr_path, bbox_inches='tight', facecolor=bg_color, dpi=300)
                plt.close()
                if corr_path.exists():
                    image_paths.append(corr_path)
                    logger.info(f"Корреляционная матрица сохранена: {corr_path}")
            except Exception as e:
                logger.error(f"Ошибка создания корреляционной матрицы: {str(e)}")
                return f"Ошибка корреляции: {str(e)} 🔥", []

        logger.info(f"Визуализации сохранены в: {self.output_dir}")
        return f"Визуализации сохранены в: {self.output_dir} 🚀", image_paths

class DataAnalyzerApp:
    def __init__(self, root: ctk.CTk):
        """Инициализация интерфейса с минимальным начальным экраном."""
        self.root = root
        self.root.title("💎 Минимальный Анализатор Данных 💎")
        self.root.geometry("1200x800")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # Динамический путь к файлу данных
        script_dir = Path(__file__).parent
        self.analyzer = DataAnalyzer(str(script_dir / "sample_data.csv"))
        self.current_theme = "light"
        self.numeric_column: Optional[str] = None
        self.categorical_column: Optional[str] = None
        self.image_labels: List[ctk.CTkLabel] = []

        # Основной фон
        self.root.configure(fg_color="#FFFFFF")

        # Начальный экран: приветственное сообщение и кнопка "Загрузить"
        self.welcome_label = ctk.CTkLabel(
            self.root,
            text="Добро пожаловать! Загрузите файл",
            font=("Helvetica", 24, "bold"),
            text_color="#424242"
        )
        self.welcome_label.place(relx=0.5, rely=0.4, anchor=ctk.CENTER)

        button_style = {
            "width": 200,
            "height": 50,
            "font": ("Helvetica", 16, "bold"),
            "fg_color": "#B0BEC5",
            "hover_color": "#CFD8DC",
            "corner_radius": 10,
            "text_color": "#424242"
        }

        self.load_button = ctk.CTkButton(
            self.root,
            text="📂 Загрузить",
            command=self.load_data,
            **button_style
        )
        self.load_button.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)
        self.animate_button(self.load_button)

        # Создаём скрытые элементы интерфейса
        # Верхняя панель
        self.navbar = ctk.CTkFrame(self.root, height=90, fg_color="#333", corner_radius=0)
        # Не размещаем пока

        # Метка для отображения даты и времени
        self.datetime_label = ctk.CTkLabel(
            self.navbar,
            text="",
            font=("Helvetica", 14),
            text_color="#E0E0E0"
        )
        self.datetime_label.pack(side=ctk.TOP, pady=5)

        # Фрейм для кнопок
        self.button_frame_1 = ctk.CTkFrame(self.navbar, fg_color="#333")
        self.button_frame_1.pack(side=ctk.TOP, fill=ctk.X, pady=5)

        # Выпадающее меню для выбора действий (будет показано после загрузки)
        self.action_menu = ctk.CTkOptionMenu(
            self.button_frame_1,
            values=["Выберите действие", "📂 Загрузить", "👀 Просмотр", "📊 Анализ", "🧹 Очистить", "📈 Визуализация", "📄 Экспорт", "🌙 Тема"],
            command=self.execute_action,
            width=200,
            fg_color="#B0BEC5",
            button_color="#CFD8DC",
            button_hover_color="#E0E0E0",
            text_color="#424242",
            font=("Helvetica", 12, "bold")
        )

        # Текстовое поле (скрыто на старте)
        self.text_area = ctk.CTkTextbox(
            self.root,
            height=200,
            font=("Helvetica", 14),
            wrap="word",
            fg_color="#F5F5F5",
            text_color="#424242",
            corner_radius=10
        )

        # Фрейм для выбора столбцов (скрыт на старте)
        self.button_frame_2 = ctk.CTkFrame(self.root, fg_color="#FFFFFF" if self.current_theme == "light" else "#212121")

        # Панель выбора столбцов
        self.numeric_label = ctk.CTkLabel(
            self.button_frame_2, text="🔢 Числовой столбец:", font=("Helvetica", 12, "bold"), text_color="#424242"
        )
        self.numeric_label.pack(side=ctk.LEFT, padx=10)

        self.numeric_menu = ctk.CTkOptionMenu(
            self.button_frame_2,
            values=["Выберите..."],
            command=self.set_numeric_column,
            width=120,
            fg_color="#B0BEC5",
            button_color="#CFD8DC",
            button_hover_color="#E0E0E0",
            text_color="#424242",
            font=("Helvetica", 12)
        )
        self.numeric_menu.pack(side=ctk.LEFT, padx=5)

        self.categorical_label = ctk.CTkLabel(
            self.button_frame_2, text="📋 Категориальный столбец:", font=("Helvetica", 12, "bold"), text_color="#424242"
        )
        self.categorical_label.pack(side=ctk.LEFT, padx=10)

        self.categorical_menu = ctk.CTkOptionMenu(
            self.button_frame_2,
            values=["Выберите..."],
            command=self.set_categorical_column,
            width=120,
            fg_color="#B0BEC5",
            button_color="#CFD8DC",
            button_hover_color="#E0E0E0",
            text_color="#424242",
            font=("Helvetica", 12)
        )
        self.categorical_menu.pack(side=ctk.LEFT, padx=5)

        # Панель для графиков (скрыта на старте)
        self.image_frame = ctk.CTkScrollableFrame(
            self.root, fg_color="transparent", corner_radius=10
        )

        logger.info("Интерфейс готов 🔥")

    def update_datetime(self):
        """Обновление текущей даты и времени в московском часовом поясе."""
        msk = pytz.timezone('Europe/Moscow')
        current_time = datetime.now(msk)
        formatted_time = current_time.strftime("%I:%M %p MSK on %A, %B %d, %Y")
        self.datetime_label.configure(text=f"Текущая дата и время: {formatted_time}")
        self.root.after(1000, self.update_datetime)

    def animate_button(self, button: ctk.CTkButton) -> None:
        """Пульсация кнопок."""
        def pulse():
            current_color = button.cget("fg_color")
            new_color = "#CFD8DC" if current_color == "#B0BEC5" else "#B0BEC5"
            button.configure(fg_color=new_color)
            self.root.after(800, pulse)
        pulse()

    def toggle_theme(self) -> None:
        """Переключение тем."""
        if self.current_theme == "light":
            self.current_theme = "dark"
            ctk.set_appearance_mode("dark")
            self.root.configure(fg_color="#212121")
            self.text_area.configure(fg_color="#424242", text_color="#000000")
            self.navbar.configure(fg_color="#333")
            self.button_frame_1.configure(fg_color="#333")
            self.button_frame_2.configure(fg_color="#212121")
            self.action_menu.configure(fg_color="#B0BEC5", button_color="#CFD8DC", button_hover_color="#E0E0E0", text_color="#000000")
            self.numeric_label.configure(text_color="#000000")
            self.categorical_label.configure(text_color="#000000")
            self.numeric_menu.configure(fg_color="#B0BEC5", button_color="#CFD8DC", button_hover_color="#E0E0E0", text_color="#000000")
            self.categorical_menu.configure(fg_color="#B0BEC5", button_color="#CFD8DC", button_hover_color="#E0E0E0", text_color="#000000")
            self.datetime_label.configure(text_color="#000000")
            self.welcome_label.configure(text_color="#000000")
        else:
            self.current_theme = "light"
            ctk.set_appearance_mode("light")
            self.root.configure(fg_color="#FFFFFF")
            self.text_area.configure(fg_color="#F5F5F5", text_color="#424242")
            self.navbar.configure(fg_color="#333")
            self.button_frame_1.configure(fg_color="#333")
            self.button_frame_2.configure(fg_color="#FFFFFF")
            self.action_menu.configure(fg_color="#B0BEC5", button_color="#CFD8DC", button_hover_color="#E0E0E0", text_color="#424242")
            self.numeric_label.configure(text_color="#424242")
            self.categorical_label.configure(text_color="#424242")
            self.numeric_menu.configure(fg_color="#B0BEC5", button_color="#CFD8DC", button_hover_color="#E0E0E0", text_color="#424242")
            self.categorical_menu.configure(fg_color="#B0BEC5", button_color="#CFD8DC", button_hover_color="#E0E0E0", text_color="#424242")
            self.datetime_label.configure(text_color="#E0E0E0")
            self.welcome_label.configure(text_color="#424242")
        logger.info(f"Тема изменена на: {self.current_theme}")

    def log_message(self, message: str) -> None:
        """Вывод сообщения."""
        self.text_area.delete(1.0, ctk.END)
        self.text_area.insert(ctk.END, message + "\n\n")
        self.text_area.see(ctk.END)
        logger.info(f"Сообщение: {message[:50]}...")

    def clear_images(self) -> None:
        """Очистка визуализаций."""
        for label in self.image_labels:
            label.destroy()
        self.image_labels = []
        logger.info("Визуализации очищены")

    def set_numeric_column(self, column: str) -> None:
        """Установка числового столбца."""
        self.numeric_column = column if column != "Выберите..." else None
        logger.info(f"Числовой столбец: {self.numeric_column}")

    def set_categorical_column(self, column: str) -> None:
        """Установка категориального столбца."""
        self.categorical_column = column if column != "Выберите..." else None
        logger.info(f"Категориальный столбец: {self.categorical_column}")

    def load_data(self) -> None:
        """Асинхронная загрузка данных с выбором файла."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV файлы", "*.csv")])
        if file_path:
            self.analyzer.file_path = Path(file_path)
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.analyzer.load_data_async())
                loop.close()
                self.root.after(0, lambda: self._post_load_data(result))
            
            self.welcome_label.configure(text="Загрузка... 🚀")
            self.root.update()
            threading.Thread(target=run_async, daemon=True).start()
        else:
            self.welcome_label.configure(text="Загрузка отменена! 🔥")

    def execute_action(self, action: str) -> None:
        """Выполнение выбранного действия из меню."""
        if action == "Выберите действие":
            return
        elif action == "📂 Загрузить":
            self.load_data()
        elif action == "👀 Просмотр":
            self.preview_data()
        elif action == "📊 Анализ":
            self.basic_analysis()
        elif action == "🧹 Очистить":
            self.clean_data()
        elif action == "📈 Визуализация":
            self.visualize_data()
        elif action == "📄 Экспорт":
            self.export_to_pdf()
        elif action == "🌙 Тема":
            self.toggle_theme()
        # Сбрасываем выбор в меню
        self.action_menu.set("Выберите действие")

    def _post_load_data(self, result: str) -> None:
        """Обработка результата загрузки и отображение полного интерфейса."""
        # Удаляем начальный экран
        self.welcome_label.destroy()
        self.load_button.place_forget()

        # Показываем полный интерфейс
        self.navbar.pack(side=ctk.TOP, fill=ctk.X)
        self.update_datetime()  # Запускаем обновление времени
        self.text_area.place(relx=0.01, rely=0.12, relwidth=0.98, relheight=0.35)
        self.button_frame_2.place(relx=0.01, rely=0.48, relwidth=0.98, relheight=0.05)
        self.image_frame.place(relx=0.01, rely=0.54, relwidth=0.98, relheight=0.45)

        # Показываем только выпадающее меню
        self.action_menu.pack(side=ctk.LEFT, padx=5)

        # Обновляем меню столбцов
        if self.analyzer.df is not None:
            numeric_cols = list(self.analyzer.df.select_dtypes(include=[np.number]).columns)
            categorical_cols = list(self.analyzer.df.select_dtypes(include=['object', 'category']).columns)
            self.numeric_menu.configure(values=["Выберите..."] + numeric_cols)
            self.categorical_menu.configure(values=["Выберите..."] + categorical_cols)
            logger.info("Меню столбцов и кнопки отображены")

        self.log_message(result)
        self.clear_images()

    def preview_data(self) -> None:
        """Предпросмотр данных."""
        if self.analyzer.df is None:
            self.log_message("Данные не загружены! 🔥")
            return
        self.log_message("Первые 5 строк:\n" + str(self.analyzer.df.head()))
        self.clear_images()

    def basic_analysis(self) -> None:
        """Базовый анализ данных."""
        self.log_message("Анализ... 📊")
        self.root.update()
        result = self.analyzer.basic_analysis()
        self.log_message(result)
        self.clear_images()

    def clean_data(self) -> None:
        """Очистка данных."""
        self.log_message("Очистка... 🧹")
        self.root.update()
        result = self.analyzer.clean_data()
        self.log_message(result)
        self.clear_images()

    def visualize_data(self) -> None:
        """Создание и отображение визуализаций."""
        self.log_message("Генерация графиков... 📈")
        self.root.update()
        self.clear_images()
        result, image_paths = self.analyzer.visualize_data(
            numeric_column=self.numeric_column,
            categorical_column=self.categorical_column,
            theme=self.current_theme
        )
        self.log_message(result)

        if not image_paths:
            self.log_message("Графики не созданы! Проверьте данные. 🔥")
            return

        for path in image_paths:
            try:
                if not path.exists():
                    self.log_message(f"График не найден: {path}")
                    continue
                img = Image.open(path)
                img = img.resize((350, 250), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label = ctk.CTkLabel(
                    self.image_frame,
                    image=photo,
                    text="",
                    fg_color="#F5F5F5" if self.current_theme == "light" else "#424242",
                    corner_radius=10
                )
                label.image = photo
                label.pack(side=ctk.LEFT, padx=15, pady=15)
                self.image_labels.append(label)
                logger.info(f"График отображён: {path}")
            except Exception as e:
                logger.error(f"Ошибка отображения графика {path}: {str(e)}")
                self.log_message(f"Ошибка графика {path}: {str(e)}")
        
        self.log_message(f"Графики готовы! {len(self.image_labels)} шт. 🔥")

    def export_to_pdf(self) -> None:
        """Экспорт визуализаций в PDF."""
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as pdf_canvas

        self.log_message("Экспорт в PDF... 📄")
        self.root.update()

        result, image_paths = self.analyzer.visualize_data(
            numeric_column=self.numeric_column,
            categorical_column=self.categorical_column,
            theme=self.current_theme
        )
        
        if not image_paths:
            self.log_message("Нет графиков для экспорта! 🔥")
            return

        pdf_path = self.analyzer.output_dir / 'minimal_visualizations.pdf'
        c = pdf_canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4

        for i, img_path in enumerate(image_paths):
            if img_path.exists():
                img = Image.open(img_path)
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                target_width = width - 120
                target_height = target_width * aspect
                if target_height > height - 120:
                    target_height = height - 120
                    target_width = target_height / aspect
                c.drawImage(
                    str(img_path),
                    (width - target_width) / 2,
                    height - target_height - 60,
                    width=target_width,
                    height=target_height
                )
                c.showPage()
        
        c.save()
        self.log_message(f"PDF сохранён: {pdf_path} 📄")
        logger.info(f"PDF экспортирован: {pdf_path}")

def main() -> None:
    """Основная функция запуска."""
    try:
        logger.info("Запуск анализатора...")
        root = ctk.CTk()
        app = DataAnalyzerApp(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    main()