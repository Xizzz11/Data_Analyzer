import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import customtkinter as ctk
import io
from PIL import Image, ImageTk
import matplotlib
import os
matplotlib.use('Agg')

class DataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.output_dir = Path('analysis_results')
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self, chunksize=10000):
        try:
            if not Path(self.file_path).is_file():
                return f"Файл {self.file_path} не найден!"
            if self.file_path.endswith('.csv'):
                chunks = pd.read_csv(self.file_path, chunksize=chunksize, encoding='utf-8', low_memory=False)
                self.df = pd.concat(chunks)
                return f"Данные успешно загружены. Размер набора: {self.df.shape}\nСтолбцы: {list(self.df.columns)}"
            else:
                raise ValueError("Поддерживается только CSV формат")
        except Exception as e:
            return f"Ошибка при загрузке данных: {str(e)}"

    def basic_analysis(self):
        if self.df is None:
            return "Данные не загружены!"
        
        output = []
        output.append("Общая информация о данных:\n")
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        output.append(buffer.getvalue())
        
        output.append("\nСтатистическое описание:\n")
        output.append(str(self.df.describe()))
        
        output.append("\nПропущенные значения:\n")
        output.append(str(self.df.isnull().sum()))
        
        with open(self.output_dir / 'basic_analysis.txt', 'w', encoding='utf-8') as f:
            f.write("\n".join(output))
        
        return "\n".join(output)

    def clean_data(self):
        if self.df is None:
            return "Данные не загружены!"

        original_shape = self.df.shape
        for column in self.df.columns:
            if self.df[column].dtype in ['int64', 'float64']:
                self.df[column] = self.df[column].fillna(self.df[column].mean())
            else:
                self.df[column] = self.df[column].fillna(self.df[column].mode()[0])

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR)))
            self.df = self.df[mask]

        if self.df.empty:
            return f"После очистки данных не осталось строк! Исходный размер: {original_shape}"
        return f"Данные очищены. Новый размер набора: {self.df.shape}"

    def visualize_data(self, numeric_column=None, categorical_column=None, theme='light'):
        if self.df is None:
            return "Данные не загружены!", []

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        if numeric_column and numeric_column not in self.df.columns:
            numeric_column = numeric_cols[0] if len(numeric_cols) > 0 else None
        if categorical_column and categorical_column not in self.df.columns:
            categorical_column = categorical_cols[0] if len(categorical_cols) > 0 else None

        if not self.output_dir.is_dir() or not os.access(self.output_dir, os.W_OK):
            return f"Папка {self.output_dir} недоступна для записи!", []

        sns.set_style("whitegrid" if theme == 'light' else "darkgrid")
        sns.set_palette("Blues" if theme == 'light' else "Blues_r")
        bg_color = '#F7FAFC' if theme == 'light' else '#1E1E2E'
        text_color = '#1A202C' if theme == 'light' else '#BAC2DE'
        plot_color = '#4299E1' if theme == 'light' else '#585B70'

        image_paths = []

        if numeric_column:
            try:
                plt.figure(figsize=(10, 6), facecolor=bg_color)
                ax = plt.gca()
                ax.set_facecolor(bg_color)
                sns.histplot(data=self.df, x=numeric_column, bins=30, color=plot_color)
                plt.title(f'Распределение {numeric_column}', color=text_color)
                plt.xlabel(numeric_column, color=text_color)
                plt.ylabel('Частота', color=text_color)
                plt.tick_params(colors=text_color)
                hist_path = self.output_dir / f'histogram_{numeric_column}.png'
                plt.savefig(hist_path, bbox_inches='tight', facecolor=bg_color)
                plt.close()
                if hist_path.exists():
                    image_paths.append(hist_path)
            except Exception as e:
                return f"Ошибка создания гистограммы: {str(e)}", []

        if categorical_column:
            try:
                plt.figure(figsize=(12, 6), facecolor=bg_color)
                ax = plt.gca()
                ax.set_facecolor(bg_color)
                self.df[categorical_column].value_counts().plot(kind='bar', color=plot_color)
                plt.title(f'Распределение {categorical_column}', color=text_color)
                plt.xlabel(categorical_column, color=text_color)
                plt.ylabel('Количество', color=text_color)
                plt.xticks(rotation=45, color=text_color)
                plt.yticks(color=text_color)
                plt.tight_layout()
                bar_path = self.output_dir / f'bar_{categorical_column}.png'
                plt.savefig(bar_path, bbox_inches='tight', facecolor=bg_color)
                plt.close()
                if bar_path.exists():
                    image_paths.append(bar_path)
            except Exception as e:
                return f"Ошибка создания столбчатой диаграммы: {str(e)}", []

        if len(numeric_cols) > 1:
            try:
                plt.figure(figsize=(12, 8), facecolor=bg_color)
                ax = plt.gca()
                ax.set_facecolor(bg_color)
                sns.heatmap(self.df[numeric_cols].corr(), annot=True, cmap='Blues' if theme == 'light' else 'Blues_r', 
                            annot_kws={'color': text_color}, cbar_kws={'label': 'Корреляция', 'ticks': [-1, 0, 1]})
                plt.title('Корреляционная матрица', color=text_color)
                plt.tick_params(colors=text_color)
                corr_path = self.output_dir / 'correlation_matrix.png'
                plt.savefig(corr_path, bbox_inches='tight', facecolor=bg_color)
                plt.close()
                if corr_path.exists():
                    image_paths.append(corr_path)
            except Exception as e:
                return f"Ошибка создания корреляционной матрицы: {str(e)}", []

        return f"Визуализации сохранены в папке: {self.output_dir}", image_paths

class DataAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор данных")
        self.root.geometry("1000x700")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.analyzer = DataAnalyzer("D:/INCHIY PROEKT/sample_data.csv")
        self.current_theme = "light"
        self.numeric_column = None
        self.categorical_column = None

        self.canvas = ctk.CTkCanvas(self.root, highlightthickness=0)
        self.canvas.pack(fill=ctk.BOTH, expand=True)
        self.create_gradient()

        self.main_frame = ctk.CTkFrame(self.canvas, fg_color="transparent")
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.95, relheight=0.95)

        self.label = ctk.CTkLabel(self.main_frame, text="💾 Анализатор данных 💾", font=("Helvetica", 24, "bold"), text_color="#2D3748")
        self.label.pack(pady=20)

        self.column_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.column_frame.pack(pady=15)

        self.numeric_label = ctk.CTkLabel(self.column_frame, text="Числовой столбец:", font=("Helvetica", 12))
        self.numeric_label.grid(row=0, column=0, padx=20, pady=15)
        self.numeric_menu = ctk.CTkOptionMenu(self.column_frame, values=["Выберите столбец"], command=self.set_numeric_column)
        self.numeric_menu.grid(row=0, column=1, padx=20, pady=15)

        self.categorical_label = ctk.CTkLabel(self.column_frame, text="Категориальный столбец:", font=("Helvetica", 12))
        self.categorical_label.grid(row=1, column=0, padx=20, pady=15)
        self.categorical_menu = ctk.CTkOptionMenu(self.column_frame, values=["Выберите столбец"], command=self.set_categorical_column)
        self.categorical_menu.grid(row=1, column=1, padx=20, pady=15)

        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(pady=15)

        button_style = {
            "width": 160,
            "height": 45,
            "font": ("Helvetica", 12),
            "fg_color": "#4299E1",
            "hover_color": "#3B82F6",
            "corner_radius": 16,
            "border_width": 1,
            "border_color": "#CBD5E0",
        }
        self.load_button = ctk.CTkButton(self.button_frame, text="Загрузить", command=self.load_data, **button_style)
        self.load_button.grid(row=0, column=0, padx=20, pady=15)

        self.preview_button = ctk.CTkButton(self.button_frame, text="Просмотр данных", command=self.preview_data, **button_style)
        self.preview_button.grid(row=0, column=1, padx=20, pady=15)

        self.analysis_button = ctk.CTkButton(self.button_frame, text="Анализ", command=self.basic_analysis, **button_style)
        self.analysis_button.grid(row=0, column=2, padx=20, pady=15)

        self.clean_button = ctk.CTkButton(self.button_frame, text="Очистить", command=self.clean_data, **button_style)
        self.clean_button.grid(row=0, column=3, padx=20, pady=15)

        self.visualize_button = ctk.CTkButton(self.button_frame, text="Визуализировать", command=self.visualize_data, **button_style)
        self.visualize_button.grid(row=0, column=4, padx=20, pady=15)

        self.text_area = ctk.CTkTextbox(
            self.main_frame,
            height=200,
            font=("Helvetica", 12),
            wrap="word",
            fg_color="#F7FAFC",
            text_color="#1A202C",
            border_color="#CBD5E0",
            border_width=1,
            corner_radius=16,
        )
        self.text_area.pack(pady=20, padx=25, fill=ctk.BOTH, expand=True)

        self.image_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.image_frame.pack(pady=20, padx=25, fill=ctk.BOTH)
        self.image_labels = []

        self.theme_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.theme_frame.pack(pady=15)

        theme_button_style = {
            "width": 200,
            "height": 45,
            "font": ("Helvetica", 12, "bold"),
            "fg_color": "#585B70",
            "hover_color": "#6B7280",
            "corner_radius": 16,
            "border_width": 1,
            "border_color": "#CBD5E0",
        }
        self.theme_button = ctk.CTkButton(self.theme_frame, text="Тёмная тема", command=self.toggle_theme, **theme_button_style)
        self.theme_button.pack()

    def create_gradient(self):
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        self.canvas.delete("all")
        if self.current_theme == "light":
            start_color = (247, 250, 252)
            end_color = (230, 240, 250)
        else:
            start_color = (30, 30, 46)
            end_color = (49, 50, 68)
        for i in range(height):
            r = int(start_color[0] + (end_color[0] - start_color[0]) * (i / height))
            g = int(start_color[1] + (end_color[1] - start_color[1]) * (i / height))
            b = int(start_color[2] + (end_color[2] - start_color[2]) * (i / height))
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas.create_line(0, i, width, i, fill=color)

    def toggle_theme(self):
        if self.current_theme == "light":
            self.current_theme = "dark"
            ctk.set_appearance_mode("dark")
            self.theme_button.configure(
                text="Светлая тема",
                fg_color="#585B70",
                hover_color="#6B7280",
                border_color="#313244"
            )
            self.label.configure(text_color="#BAC2DE")
            self.numeric_label.configure(text_color="#BAC2DE")
            self.categorical_label.configure(text_color="#BAC2DE")
            self.text_area.configure(
                fg_color="#24273A",
                text_color="#BAC2DE",
                border_color="#313244"
            )
            for btn in [self.load_button, self.preview_button, self.analysis_button, self.clean_button, self.visualize_button]:
                btn.configure(
                    fg_color="#585B70",
                    hover_color="#6B7280",
                    border_color="#313244"
                )
            self.numeric_menu.configure(
                fg_color="#24273A",
                button_color="#585B70",
                button_hover_color="#6B7280",
                text_color="#BAC2DE"
            )
            self.categorical_menu.configure(
                fg_color="#24273A",
                button_color="#585B70",
                button_hover_color="#6B7280",
                text_color="#BAC2DE"
            )
        else:
            self.current_theme = "light"
            ctk.set_appearance_mode("light")
            self.theme_button.configure(
                text="Тёмная тема",
                fg_color="#585B70",
                hover_color="#6B7280",
                border_color="#CBD5E0"
            )
            self.label.configure(text_color="#2D3748")
            self.numeric_label.configure(text_color="#2D3748")
            self.categorical_label.configure(text_color="#2D3748")
            self.text_area.configure(
                fg_color="#F7FAFC",
                text_color="#1A202C",
                border_color="#CBD5E0"
            )
            for btn in [self.load_button, self.preview_button, self.analysis_button, self.clean_button, self.visualize_button]:
                btn.configure(
                    fg_color="#4299E1",
                    hover_color="#3B82F6",
                    border_color="#CBD5E0"
                )
            self.numeric_menu.configure(
                fg_color="#F7FAFC",
                button_color="#4299E1",
                button_hover_color="#3B82F6",
                text_color="#1A202C"
            )
            self.categorical_menu.configure(
                fg_color="#F7FAFC",
                button_color="#4299E1",
                button_hover_color="#3B82F6",
                text_color="#1A202C"
            )
        self.create_gradient()

    def log_message(self, message):
        self.text_area.delete(1.0, ctk.END)
        self.text_area.insert(ctk.END, message + "\n\n")
        self.text_area.see(ctk.END)

    def clear_images(self):
        for label in self.image_labels:
            label.destroy()
        self.image_labels = []

    def set_numeric_column(self, column):
        self.numeric_column = column if column != "Выберите столбец" else None

    def set_categorical_column(self, column):
        self.categorical_column = column if column != "Выберите столбец" else None

    def load_data(self):
        self.log_message("Загрузка данных...")
        self.root.update()
        result = self.analyzer.load_data()
        self.log_message(result)
        self.clear_images()
        if self.analyzer.df is not None:
            numeric_cols = list(self.analyzer.df.select_dtypes(include=[np.number]).columns)
            categorical_cols = list(self.analyzer.df.select_dtypes(include=['object', 'category']).columns)
            self.numeric_menu.configure(values=["Выберите столбец"] + numeric_cols)
            self.categorical_menu.configure(values=["Выберите столбец"] + categorical_cols)

    def preview_data(self):
        if self.analyzer.df is None:
            self.log_message("Данные не загружены!")
            return
        self.log_message("Первые 5 строк данных:\n" + str(self.analyzer.df.head()))
        self.clear_images()

    def basic_analysis(self):
        self.log_message("Выполняется анализ...")
        self.root.update()
        result = self.analyzer.basic_analysis()
        self.log_message(result)
        self.clear_images()

    def clean_data(self):
        self.log_message("Очистка данных...")
        self.root.update()
        result = self.analyzer.clean_data()
        self.log_message(result)
        self.clear_images()

    def visualize_data(self):
        self.log_message("Создание визуализаций...")
        self.root.update()
        self.clear_images()
        result, image_paths = self.analyzer.visualize_data(
            numeric_column=self.numeric_column,
            categorical_column=self.categorical_column,
            theme=self.current_theme
        )
        self.log_message(result)

        if not image_paths:
            self.log_message("Нет изображений для отображения! Проверьте выбор столбцов.")
            return

        for path in image_paths:
            try:
                if not path.exists():
                    self.log_message(f"Файл изображения не найден: {path}")
                    continue
                img = Image.open(path)
                img = img.resize((300, 200), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label = ctk.CTkLabel(
                    self.image_frame,
                    image=photo,
                    text="",
                    fg_color="#1E1E2E" if self.current_theme == "dark" else "#F7FAFC",
                    corner_radius=8,
                    border_width=1,
                    border_color="#313244" if self.current_theme == "dark" else "#CBD5E0"
                )
                label.image = photo
                label.pack(side=ctk.LEFT, padx=20, pady=15)
                self.image_labels.append(label)
                self.root.update_idletasks()
            except Exception as e:
                self.log_message(f"Ошибка отображения изображения {path}: {str(e)}")
        
        self.log_message(f"Отображено {len(self.image_labels)} изображений.")

def main():
    root = ctk.CTk()
    app = DataAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()