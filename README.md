# pr_08
## Практическая работа 8. Анализ метода загрузки данных
## Цель:
Определить наиболее эффективный метод загрузки данных (малых и больших объемов) из CSV-файлов в СУБД PostgreSQL, сравнивая время выполнения для методов: pandas.to_sql(), psycopg2.copy_expert() (с файлом и с io.StringIO), и пакетная вставка (psycopg2.extras.execute_values).
## Задачи:
1. Подключиться к предоставленной базе данных PostgreSQL.
2. Проанализировать структуру исходных CSV-файлов (upload_test_data.csv ,  upload_test_data_big.csv).
3. Создать эскизы ER-диаграмм для таблиц, соответствующих структуре CSV-файлов.
4. Реализовать три различных метода загрузки данных в PostgreSQL(pandas.to_sql(), copy_expert(), io.StringIO).
5. Измерить время, затраченное каждым методом на загрузку данных из малого файла (upload_test_data.csv).
6. Измерить время, затраченное каждым методом на загрузку данных из большого файла (upload_test_data_big.csv).
7. Визуализировать результаты сравнения времени загрузки с помощью гистограммы (matplotlib).
8. Сделать выводы об эффективности каждого метода для разных объемов данных.

## Индивидуальные задания, вариант 12:
1. 	Создать таблицы sales_small, sales_big.
2. 	Метод: copy_expert (StringIO).
3. 	Метод: copy_expert (file)
4. 	SQL: Выбрать первые 15 записей из sales_big, отсортированных по cost (ASC).
5. 	Python: Построить ящик с усами (boxplot) для total_revenue из sales_small.

## Выполнение практической работы
# Подключение к предоставленной базе данных PostgreSQL.
Сначала необходимо подключиться ко всем требуемым библиотекам
````
%pip install psycopg2-binary pandas sqlalchemy matplotlib numpy
````
# Результат выполнения команды:
![image](https://github.com/user-attachments/assets/e7f7420a-4ac8-4528-9385-868b75f085b3)
````
import psycopg2
from psycopg2 import Error
from psycopg2 import extras # For execute_values
import pandas as pd
from sqlalchemy import create_engine
import io # For StringIO
import time
import matplotlib.pyplot as plt
import numpy as np
import os # To check file existence
````
Указываем путь к необходимым файлам 
````
big_csv_path:r'"C:\Users\damdi\Desktop\уник\Практикум sql\8пр\upload_test_data_big.csv"'
````
````
small_csv_path:r'"C:\Users\damdi\Desktop\уник\Практикум sql\8пр\upload_test_data.csv"'
````
# Результат выполнения команд:
![image](https://github.com/user-attachments/assets/7efb8092-8969-478e-90d9-bb6d2681c753)

Все успешно загрузилось

Для подключение и проверки загрузки данных можно использовать следующий шаблон:
````
print("Libraries installed and imported successfully.")

# Database connection details (replace with your actual credentials if different)
DB_USER = "ваш логин"
DB_PASSWORD = "ваш пароль"
DB_HOST = "ваш хост"
DB_PORT = "5432"
DB_NAME = "lect_08_bda_big_data"

# CSV File Paths (Ensure these files are uploaded to your Colab environment)
small_csv_path = 'upload_test_data.csv'
big_csv_path = 'upload_test_data_big.csv' # Corrected filename

# Table name in PostgreSQL
table_name = 'sales_data'
````
# Результат выполнения кода:
![image](https://github.com/user-attachments/assets/50ee0b27-f2fe-45b5-82a4-3b30032969e4)

# Анализ различных метода загрузки данных в PostgreSQL
Для нахождения самого эффектного метода, необходимо провести измерения времени для каждого метода загрузки. Для этого можно использовать следующий код.

````
# @title # 6. Data Loading Methods Implementation and Timing
# @markdown Реализация и измерение времени для каждого метода загрузки.

# Dictionary to store timing results
timing_results = {
    'small_file': {},
    'big_file': {}
}

# Check if files exist before proceeding
if not os.path.exists(small_csv_path):
    print(f"ERROR: Small CSV file not found: {small_csv_path}. Upload it and restart.")
elif not os.path.exists(big_csv_path):
    print(f"ERROR: Big CSV file not found: {big_csv_path}. Upload it and restart.")
elif not connection or not cursor or not engine:
    print("ERROR: Database connection not ready. Cannot proceed.")
else:
    # --- Method 1: pandas.to_sql() ---
    def load_with_pandas_to_sql(eng, df, tbl_name, chunk_size=1000):
        """Loads data using pandas.to_sql() and returns time taken."""
        start_time = time.perf_counter()
        try:
            # Using method='multi' might be faster for some DBs/data
            # Chunksize helps manage memory for large files
            df.to_sql(tbl_name, eng, if_exists='append', index=False, method='multi', chunksize=chunk_size)
        except Exception as e:
             print(f"Error in pandas.to_sql: {e}")
             # Note: No explicit transaction management here, relies on SQLAlchemy/DBAPI defaults or engine settings.
             # For critical data, wrap in a try/except with explicit rollback if needed.
             raise # Re-raise the exception to signal failure
        end_time = time.perf_counter()
        return end_time - start_time

    # --- Method 2: psycopg2.copy_expert() with CSV file ---
    def load_with_copy_expert_file(conn, cur, tbl_name, file_path):
        """Loads data using psycopg2.copy_expert() directly from file and returns time taken."""
        start_time = time.perf_counter()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Skip header row using COPY options
                sql_copy = f"""
                COPY {tbl_name} FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')
                """
                cur.copy_expert(sql=sql_copy, file=f)
            conn.commit() # Commit transaction after successful COPY
        except (Exception, Error) as error:
            print(f"Error in copy_expert (file): {error}")
            conn.rollback() # Rollback on error
            raise
        end_time = time.perf_counter()
        return end_time - start_time

    # --- Method 3: psycopg2.copy_expert() with io.StringIO ---
    def load_with_copy_expert_stringio(conn, cur, df, tbl_name):
        """Loads data using psycopg2.copy_expert() from an in-memory StringIO buffer and returns time taken."""
        start_time = time.perf_counter()
        buffer = io.StringIO()
        # Write dataframe to buffer as CSV, including header
        df.to_csv(buffer, index=False, header=True, sep=',')
        buffer.seek(0) # Rewind buffer to the beginning
        try:
            sql_copy = f"""
            COPY {tbl_name} FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')
            """
            cur.copy_expert(sql=sql_copy, file=buffer)
            conn.commit() # Commit transaction after successful COPY
        except (Exception, Error) as error:
            print(f"Error in copy_expert (StringIO): {error}")
            conn.rollback() # Rollback on error
            raise
        finally:
            buffer.close() # Ensure buffer is closed
        end_time = time.perf_counter()
        return end_time - start_time


    # --- Timing Execution ---
    print("\n--- Starting Data Loading Tests ---")

    # Load DataFrames (only once)
    print("Loading CSV files into Pandas DataFrames...")
    try:
        df_small = pd.read_csv(small_csv_path)
        # The big file might be too large to load fully into Colab memory.
        # If memory errors occur, consider processing it in chunks for methods
        # that support it (like pandas.to_sql with chunksize, or modify batch insert).
        # For COPY methods, memory isn't usually an issue as they stream.
        df_big = pd.read_csv(big_csv_path)
        print(f"Loaded {len(df_small)} rows from {small_csv_path}")
        print(f"Loaded {len(df_big)} rows from {big_csv_path}")
    except MemoryError:
        print("\nERROR: Not enough RAM to load the large CSV file into a Pandas DataFrame.")
        print("Some methods (pandas.to_sql, StringIO, Batch Insert) might fail or be inaccurate.")
        print("The copy_expert (file) method should still work.")
        # We can try to proceed, but note the limitation
        df_big = None # Indicate that the big dataframe couldn't be loaded
    except Exception as e:
        print(f"Error loading CSVs into DataFrames: {e}")
        df_small, df_big = None, None # Stop execution if loading fails


    if df_small is not None: # Proceed only if small DF loaded
        # --- Small File Tests ---
        print(f"\n--- Testing with Small File ({small_csv_path}) ---")

        # Test pandas.to_sql
        try:
            reset_table(connection, cursor, table_name)
            print("Running pandas.to_sql...")
            t = load_with_pandas_to_sql(engine, df_small, table_name)
            timing_results['small_file']['pandas.to_sql'] = t
            print(f"Finished in {t:.4f} seconds.")
        except Exception as e: print(f"pandas.to_sql failed for small file.")

        # Test copy_expert (file)
        try:
            reset_table(connection, cursor, table_name)
            print("Running copy_expert (file)...")
            t = load_with_copy_expert_file(connection, cursor, table_name, small_csv_path)
            timing_results['small_file']['copy_expert (file)'] = t
            print(f"Finished in {t:.4f} seconds.")
        except Exception as e: print(f"copy_expert (file) failed for small file.")

        # Test copy_expert (StringIO)
        try:
            reset_table(connection, cursor, table_name)
            print("Running copy_expert (StringIO)...")
            t = load_with_copy_expert_stringio(connection, cursor, df_small, table_name)
            timing_results['small_file']['copy_expert (StringIO)'] = t
            print(f"Finished in {t:.4f} seconds.")
        except Exception as e: print(f"copy_expert (StringIO) failed for small file.")



    # --- Big File Tests ---
    print(f"\n--- Testing with Big File ({big_csv_path}) ---")

    # Test pandas.to_sql (if df_big loaded)
    if df_big is not None:
        try:
            reset_table(connection, cursor, table_name)
            print("Running pandas.to_sql...")
            t = load_with_pandas_to_sql(engine, df_big, table_name, chunk_size=10000) # Larger chunksize for big file
            timing_results['big_file']['pandas.to_sql'] = t
            print(f"Finished in {t:.4f} seconds.")
        except Exception as e: print(f"pandas.to_sql failed for big file.")
    else:
        print("Skipping pandas.to_sql for big file (DataFrame not loaded).")


    # Test copy_expert (file) - This should work even if df_big didn't load
    try:
        reset_table(connection, cursor, table_name)
        print("Running copy_expert (file)...")
        t = load_with_copy_expert_file(connection, cursor, table_name, big_csv_path)
        timing_results['big_file']['copy_expert (file)'] = t
        print(f"Finished in {t:.4f} seconds.")
    except Exception as e: print(f"copy_expert (file) failed for big file.")


    # Test copy_expert (StringIO) (if df_big loaded)
    if df_big is not None:
        try:
            reset_table(connection, cursor, table_name)
            print("Running copy_expert (StringIO)...")
            t = load_with_copy_expert_stringio(connection, cursor, df_big, table_name)
            timing_results['big_file']['copy_expert (StringIO)'] = t
            print(f"Finished in {t:.4f} seconds.")
        except Exception as e: print(f"copy_expert (StringIO) failed for big file.")
    else:
        print("Skipping copy_expert (StringIO) for big file (DataFrame not loaded).")



    print("\n--- Data Loading Tests Finished ---")

# Final check of results dictionary
print("\nTiming Results Summary:")
import json
print(json.dumps(timing_results, indent=2))
````
# Результат выполнения кода:
![image](https://github.com/user-attachments/assets/1efef73e-6a99-4784-9170-7ec4db0f7e3e)


# Визуализация результатов:
Для визуализации используем следующий шаблон:
````
# @title # 7. Results Visualization
# @markdown Визуализация результатов сравнения времени загрузки.

if not timing_results['small_file'] and not timing_results['big_file']:
    print("No timing results available to plot.")
else:
    # Prepare data for plotting
    methods = list(set(timing_results['small_file'].keys()) | set(timing_results['big_file'].keys()))
    methods.sort() # Ensure consistent order

    small_times = [timing_results['small_file'].get(method, 0) for method in methods] # Use .get with default 0 if method failed
    big_times = [timing_results['big_file'].get(method, 0) for method in methods]

    x = np.arange(len(methods))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7)) # Increase figure size for better readability

    rects1 = ax.bar(x - width/2, small_times, width, label=f'Small File ({os.path.basename(small_csv_path)})', color='skyblue')
    rects2 = ax.bar(x + width/2, big_times, width, label=f'Big File ({os.path.basename(big_csv_path)})', color='lightcoral')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Comparison of Data Loading Times into PostgreSQL')
    ax.set_xticks(x)
    # Rotate labels for better fit if names are long
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.legend()

    # Add labels on top of bars
    ax.bar_label(rects1, padding=3, fmt='%.3f', rotation=90)
    ax.bar_label(rects2, padding=3, fmt='%.3f', rotation=90)

    # Use a logarithmic scale for the y-axis if the differences are very large
    # This helps visualize smaller values when large values dominate.
    # Check if the max time is significantly larger than the min non-zero time
    all_times = [t for t in small_times + big_times if t > 0]
    if all_times and (max(all_times) / min(all_times) > 50): # Threshold for using log scale
        ax.set_yscale('log')
        ax.set_ylabel('Time (seconds, log scale)')
        # Adjust label formatting for log scale if needed, though default might be fine
        ax.bar_label(rects1, padding=3, fmt='%.3f', rotation=90)
        ax.bar_label(rects2, padding=3, fmt='%.3f', rotation=90)
        print("\nNote: Using logarithmic scale for Y-axis due to large time differences.")


    fig.tight_layout() # Adjust layout to prevent labels overlapping
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
    plt.show()
````
# Результат выполнения кода:
![image](https://github.com/user-attachments/assets/857e93fb-621e-4629-bbca-14aa59d316a9)


# Анализ результа:
Судя по показятелям, эффективнее всего использовать метод copy_expert(), но и io.StringIO не так сильно отличается по скорости. Однако, среди трех методов есть явный аутсайдер, и это-pandas.to_sql().

## Выполнение индивидуальных заданий
# Создание функций
Ниже я размещу шаблон, который открывает, закрывает соединение, также в нем прописаны функции для выполнения последующих индивидуальных задач. Обратите внимание, что коды не будут работать без них.
````
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO

# --- Константы для подключения к PostgreSQL ---
DB_USER = "postgres"
DB_PASSWORD = "1"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "lect_08_bda_big_data"       

# Определение констант
small_table_name = 'sales_small'
big_table_name = 'sales_big'
small_csv_path = r'C:\Users\damdi\Desktop\уник\Практикум sql\data\upload_test_data.csv'  
big_csv_path = r'C:\Users\damdi\Desktop\уник\Практикум sql\data\upload_test_data_big.csv'

# Подключение к БД
def connect_db():
    try:
        connection = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = connection.cursor()
        print("Успешное подключение к PostgreSQL")
        return connection, cursor
    except Exception as e:
        print(f"Ошибка подключения к PostgreSQL: {e}")
        return None, None

# Функция для создания таблицы
def create_table(table_name):
    try:
        cursor.execute(f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY,
            quantity INTEGER,
            cost NUMERIC(10, 2),
            total_revenue NUMERIC(12, 2)
        );
        """)
        connection.commit()
        print(f"Таблица {table_name} успешно создана или уже существует")
    except Exception as e:
        print(f"Ошибка при создании таблицы {table_name}: {e}")

# Функция для загрузки данных из файла
def load_via_copy_file(file_path, table_name):
    try:
        with open(file_path, 'r') as f:
            cursor.copy_expert(f"""
            COPY {table_name}(id, quantity, cost, total_revenue) 
            FROM STDIN WITH CSV HEADER DELIMITER ','
            """, f)
        connection.commit()
        print(f"Данные из {file_path} успешно загружены в {table_name}")
    except Exception as e:
        print(f"Ошибка при загрузке данных в {table_name}: {e}")

# Функция для выполнения SQL запросов
def execute_sql(query, fetch=False):
    try:
        cursor.execute(query)
        connection.commit()
        if fetch:
            return cursor.fetchall()
        return True
    except Exception as e:
        print(f"Ошибка выполнения SQL запроса: {e}")
        return None

# Функция для загрузки данных в DataFrame
def load_df_from_sql(query):
    try:
        return pd.read_sql(query, connection)
    except Exception as e:
        print(f"Ошибка загрузки данных в DataFrame: {e}")
        return None


connection, cursor = connect_db()

if not connection or not cursor:
    print("Подключение к базе данных неактивно. Пожалуйста, проверьте параметры подключения.")
else:
    print("--- Запуск Варианта 12  ---")

...

    # Закрытие соединения
    if cursor:
        cursor.close()
        if connection:
             connection.close()
print("\nСоединение с базой данных закрыто.")

````
# Задание 1. 	Создать таблицы sales_small, sales_big.
Для создания таблицы необходимо посмотреть ее структуру, опираясь на созданную диаграмму 

![image](https://github.com/user-attachments/assets/bad57faf-0eee-48f5-9414-5400639b15d9)

Для выполнения задания составим код, создающий таблицы:
````
  #  Задание 1. Настройка таблиц. Создать таблицы sales_small, sales_big.
    print("\n--- Задание 1. Создание таблиц sales_small, sales_big---")
    create_table(small_table_name)
    create_table(big_table_name)
````
# Результат создания таблиц:
![image](https://github.com/user-attachments/assets/4fa126b5-3524-4c6b-90db-ded85a076431)

Таблицы созданы

# Задание 2. Загрузка малых данных, метод: copy_expert (StringIO)
````
    # Задание 2. Загрузка малых данных методом copy_expert (StringIO)
    print(f"\n--- Задание 2: Загрузка данных из '{small_csv_path}' в '{small_table_name}' ---")
    if os.path.exists(small_csv_path):
        load_via_copy_stringio(small_csv_path, small_table_name)
    else:
        print(f"ОШИБКА: Файл '{small_csv_path}' не найден.")
````


# Задание 3. Загрузка больших данных, метод: copy_expert (file)
````
    # Задание 3. Загрузка больших данных методом copy_expert (file)
    print(f"\n--- Задание 3: Загрузка данных из '{big_csv_path}' в '{big_table_name}' (метод file) ---")
    if os.path.exists(big_csv_path):
        load_via_copy_file(big_csv_path, big_table_name)
    else:
        print(f"ОШИБКА: Файл '{big_csv_path}' не найден. Загрузка не выполнена.")

````


# Задание 4. SQL Анализ:SQL: выбрать первые 15 записей из sales_big, отсортированных по cost (ASC).
````
   # Задание 4. SQL-запрос для выбора первых 15 записей из sales_big, отсортированных по cost (ASC)
query = """
SELECT *
FROM sales_big
ORDER BY cost ASC
LIMIT 15;
"""
print("\n--- Задание 4: Выбрать первые 15 записей из sales_big, отсортированных по cost (ASC) ---")
df = load_df_from_sql(query)

# Проверяем результат
if df is not None and not df.empty:
    print("Данные успешно загружены:")
    print(df)
else:
    print("Не удалось загрузить данные.")
````
# Результаты выполнения индивидуальной работы
![image](https://github.com/user-attachments/assets/cdfca212-78e4-4efa-9955-bae7d88f67a6)

# Выводы
В рамках выполнения практической работы была проведена исследовательская задача по изучению и сравнению методов загрузки данных из CSV-файлов в СУБД PostgreSQL через VSCode. Работа позволила освоить ключевые аспекты взаимодействия с базами данных, анализа производительности различных подходов и выбора оптимальных решений для обработки данных разного объема.

## Структура репозитория:
- `erd_diagram.png` — ERD диаграмма схемы базы данных.
- `Damdinova_Kristina_Takhirovna_Tasks` — Jupyter Notebook с выполнением задач практической работы, до индивидуальных заданий.
- `Damdinova_Kristina_Takhirovna_V12` — Jupyter Notebook с выполнением индивидуальных заданий.
