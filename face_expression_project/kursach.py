import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xlrd2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Функция выбора Excel-файла
def select_file():
    file_path = filedialog.askopenfilename(
        title="Выберите файл Excel",
        filetypes=[("Excel files", "*.xls *.xlsx")]
    )
    if not file_path:
        messagebox.showwarning("Файл не выбран", "Вы не выбрали файл. Программа завершена.")
        root.destroy()
        return None
    return file_path

# Инициализация окна до загрузки данных
root = tk.Tk()
root.withdraw()
excel_path = select_file()
if not excel_path:
    exit()
root.deiconify()

# Загрузка Excel
try:
    workbook = xlrd2.open_workbook(excel_path)
except Exception as e:
    messagebox.showerror("Ошибка", f"Не удалось открыть файл: {str(e)}")
    root.destroy()
    exit()

sheet = workbook.sheet_by_index(0)
row = sheet.nrows

names = sheet.col_values(4)
versions = sheet.col_values(5)
danger_levels = sheet.col_values(12)
exploit_status = sheet.col_values(15)

sql_versions_set = set()
for i in range(3, row):
    if i < len(names) and 'Microsoft SQL Server' in str(names[i]) and i < len(versions):
        sql_versions_set.add(str(versions[i]))
sql_versions = sorted(sql_versions_set)

root.title("Анализ угроз - Microsoft SQL Server")
root.geometry("600x600")

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=('Arial', 12), justify='left', wraplength=500)
result_label.pack(pady=10)
tk.Label(root, text="(Необязательно) выберите версию Microsoft SQL Server:", font=('Arial', 10)).pack()
version_var = tk.StringVar()
version_menu = ttk.Combobox(root, textvariable=version_var, values=sql_versions, state="readonly", width=40)
version_menu.pack(pady=5)

chart_frame = tk.Frame(root)
chart_frame.pack(pady=10)
current_canvas = None
def analyze_data(filter_exploit=None, show_chart=False):
    global current_canvas

    selected_version = version_var.get().strip()
    danger_super = danger_high = danger_middle = danger_low = 0

    for i in range(3, row):
        if i >= len(names) or i >= len(versions) or i >= len(danger_levels) or i >= len(exploit_status):
            continue

        name = str(names[i])
        version = str(versions[i]).strip()
        level = str(danger_levels[i]).strip()
        exploit = str(exploit_status[i]).strip()

        if 'Microsoft SQL Server' not in name:
            continue

        if selected_version and version != selected_version:
            continue

        if filter_exploit and exploit != filter_exploit:
            continue

        if level.startswith('К'):
            danger_super += 1
        elif level.startswith('В'):
            danger_high += 1
        elif level.startswith('С'):
            danger_middle += 1
        else:
            danger_low += 1
    result_var.set(
        f'Критический: {danger_super}\n'
        f'Высокий: {danger_high}\n'
        f'Средний: {danger_middle}\n'
        f'Низкий: {danger_low}'
    )

    if show_chart:
        labels = ['Критический', 'Высокий', 'Средний', 'Низкий']
        sizes = [danger_super, danger_high, danger_middle, danger_low]
        colors = ['red', 'darkorange', 'yellow', 'green']
        filtered = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if not filtered:
            messagebox.showinfo("Диаграмма", "Нет данных для построения диаграммы.")
            return
        labels, sizes, colors = zip(*filtered)
        if current_canvas:
            current_canvas.get_tk_widget().destroy()
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        fig.tight_layout()

        current_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        current_canvas.draw()
        current_canvas.get_tk_widget().pack()

tk.Button(root, text="Показать все угрозы", command=lambda: analyze_data()).pack(pady=5)

exploit_filters = ["Существует", "Данные уточняются", "Существует в открытом доступе"]
for status in exploit_filters:
    tk.Button(root, text=f"Наличие эксплойта: {status}", command=lambda s=status: analyze_data(filter_exploit=s)).pack(pady=2)

tk.Button(root, text="Показать диаграмму", command=lambda: analyze_data(show_chart=True)).pack(pady=10)

root.mainloop()
