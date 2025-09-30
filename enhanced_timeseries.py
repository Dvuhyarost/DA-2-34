#!/usr/bin/env python3
# -*- coding: utf-8  -*-
# -*- mode:   python -*-

# This file is part of the educational Data Analysis tasks.
#
# Copyright (C) 2025  ferrovovan
# Modifications (C) 2025  Dvuhyarost
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List


def load_dataframe_from_file(path: str, show_progress: bool = False) -> pd.DataFrame:
   """
    Загружает таблицу из CSV файла, содержащего временные метки.

    Параметры
    ---------
    path : str
        Путь до CSV файла с временными метками
    show_progress : bool, optional
        Показывать ли прогресс загрузки, по умолчанию False

    Возвращает
    -------
    pd.DataFrame
        Таблица с одним столбцом "timestamp"

    Исключения
    ----------
    FileNotFoundError
        Если файл не существует
    ValueError
        Если в файле неверная структура или данные не читаются
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    try:
        if show_progress:
            print(f"Загрузка файла {path}...")
            
        df = pd.read_csv(path)
        
        if show_progress:
            print(f"Загружено {len(df)} строк")
            
    except Exception as e:
        raise ValueError(f"Ошибка чтения CSV: {e}")

    if df.shape[1] != 1:
        raise ValueError("Ожидался ровно один столбец во входном CSV.")

    if df.empty:
        raise ValueError("Файл не содержит данных.")

    df.columns = ["timestamp"]
    return df


def create_periodic_dataframe(
        start_timedate_point: str, periods: int, freq: str
    ) -> pd.DataFrame:
     """
    Создает таблицу с синтетическими временными метками.

    Параметры
    ---------
    start_timedate_point : str
        Начальная точка временного ряда в формате "YYYY-MM-DD HH:MM:SS"
    periods : int
        Количество создаваемых временных меток
    freq : str
        Частота генерации меток (например, '4h', '1D', '30min')

    Возвращает
    -------
    pd.DataFrame
        Таблица с синтетическими временными метками

    Исключения
    ----------
    ValueError
        Если количество периодов не положительное или не удалось создать метки
    """
    if periods <= 0:
        raise ValueError("Количество периодов должно быть положительным числом.")

    valid_freqs = ['H', 'h', 'D', 'd', 'min', 'MIN', 'S', 's', 'M', 'Y', 'W']
    if not any(freq.endswith(f) for f in valid_freqs):
        print(f"Предупреждение: частота '{freq}' может быть невалидной. Допустимые: {', '.join(valid_freqs)}")

    try:
        timestamps = pd.date_range(
            start=start_timedate_point,
            periods=periods,
            freq=freq
        )
        return pd.DataFrame({"timestamp": timestamps})
    
    except Exception as e:
        raise ValueError(f"Ошибка при создании временных меток: {e}")


def convert_to_datetime(df: pd.DataFrame, handle_na: str = 'raise') -> pd.DataFrame:
    """
    Преобразует строковые метки во временной формат datetime.

    Параметры
    ---------
    df : pd.DataFrame
        Таблица с столбцом "timestamp" для преобразования
    handle_na : str, optional
        Стратегия обработки пустых значений:
        - 'raise': вызвать исключение
        - 'drop': удалить строки с пустыми значениями  
        - 'coerce': преобразовать в NaT, по умолчанию 'raise'

    Возвращает
    -------
    pd.DataFrame
        Таблица с преобразованными временными метками

    Исключения
    ----------
    KeyError
        Если отсутствует столбец "timestamp"
    ValueError
        Если значения нельзя преобразовать в datetime или неверная стратегия handle_na
    """
    if "timestamp" not in df.columns:
        raise KeyError("Отсутствует обязательный столбец 'timestamp'.")

    na_count = df["timestamp"].isna().sum()
    if na_count > 0:
        if handle_na == 'raise':
            raise ValueError(f"Столбец 'timestamp' содержит {na_count} пустых значений.")
        elif handle_na == 'drop':
            print(f"Удалено {na_count} строк с пустыми значениями timestamp")
            df = df.dropna(subset=["timestamp"]).copy()
        elif handle_na == 'coerce':
            print(f"Обнаружено {na_count} пустых значений, они будут преобразованы в NaT")
        else:
            raise ValueError("Недопустимое значение handle_na. Допустимо: 'raise', 'drop', 'coerce'")

    try:
        result = df.copy()
        result["timestamp"] = pd.to_datetime(
            result["timestamp"], 
            errors='coerce' if handle_na == 'coerce' else 'raise'
        )
        
        if handle_na == 'coerce' and result["timestamp"].isna().any():
            print("Предупреждение: некоторые значения не удалось преобразовать в datetime")
            
        return result

    except Exception as e:
        raise ValueError(f"Ошибка преобразования в datetime: {e}")


def _validate_datetime_column(df: pd.DataFrame) -> None:
    """
    Проверяет валидность столбца с временными метками.

    Параметры
    ---------
    df : pd.DataFrame
        Таблица для валидации

    Исключения
    ----------
    KeyError
        Если отсутствует столбец "timestamp"
    TypeError
        Если столбец не имеет тип datetime
    ValueError
        Если столбец содержит пустые значения
    """
    if "timestamp" not in df.columns:
        raise KeyError("Отсутствует обязательный столбец 'timestamp'.")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("Столбец 'timestamp' должен быть формата datetime.")

    if df["timestamp"].isna().any():
        raise ValueError("Столбец 'timestamp' содержит пустые значения.")


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекает временные признаки из меток (DA-1-35 + DA-2-34).

    Параметры
    ---------
    df : pd.DataFrame
        Таблица с datetime столбцом "timestamp"

    Возвращает
    -------
    pd.DataFrame
        Таблица с извлеченными признаками:
        - day, month, year (DA-1-35)
        - hour, is_morning, is_afternoon, is_evening, is_night (DA-2-34)

    Исключения
    ----------
    KeyError
        Если отсутствует столбец "timestamp"
    TypeError
        Если столбец не имеет тип datetime
    """
    _validate_datetime_column(df)
    
    features_df = pd.DataFrame(index=df.index)
    timestamp_series = df["timestamp"]
    
    # DA-1-35: Базовые признаки даты
    features_df["day"] = timestamp_series.dt.day
    features_df["month"] = timestamp_series.dt.month
    features_df["year"] = timestamp_series.dt.year
    
    # DA-2-34: Признаки времени суток
    features_df["hour"] = timestamp_series.dt.hour
    hour = timestamp_series.dt.hour
    features_df["is_morning"] = ((hour >= 6) & (hour < 12)).astype(int)
    features_df["is_afternoon"] = ((hour >= 12) & (hour < 18)).astype(int)
    features_df["is_evening"] = ((hour >= 18) & (hour < 24)).astype(int)
    features_df["is_night"] = ((hour >= 0) & (hour < 6)).astype(int)
    
    return features_df


def create_time_of_day_bar_plot(
    features_df: pd.DataFrame, 
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    colors: List[str] = None,
    title: str = 'Распределение записей по времени суток (DA-2-34)'
) -> None:
     """
    Создает bar plot для визуализации статистики времени суток.

    Параметры
    ---------
    features_df : pd.DataFrame
        DataFrame с бинарными признаками времени суток
    figsize : tuple, optional
        Размер графика в дюймах, по умолчанию (10, 6)
    colors : List[str], optional
        Список цветов для столбцов, по умолчанию стандартная палитра
    title : str, optional
        Заголовок графика, по умолчанию стандартный

    Исключения
    ----------
    ValueError
        Если отсутствуют необходимые столбцы с признаками времени суток
    """
    if colors is None:
        colors = ['skyblue', 'lightgreen', 'orange', 'purple']
    
    time_columns = ["is_morning", "is_afternoon", "is_evening", "is_night"]
    
    if not all(col in features_df.columns for col in time_columns):
        raise ValueError("Отсутствуют необходимые столбцы с признаками времени суток.")
    
    counts = features_df[time_columns].sum()
    total = len(features_df)
    
    if total == 0:
        print("Нет данных для построения графика")
        return

    time_periods = ["Утро (6-12)", "День (12-18)", "Вечер (18-24)", "Ночь (0-6)"]
    
    fig = plt.figure(figsize=figsize)
    try:
        bars = plt.bar(time_periods, counts.values, color=colors)
        plt.title(title)
        plt.ylabel('Количество записей')
        plt.grid(axis='y', alpha=0.3)
        
        # Добавление значений на столбцы
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            # Создание директории если нужно
            directory = os.path.dirname(save_path) or '.'
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в файл: {save_path}")
        else:
            plt.show()
            
    finally:
        plt.close(fig)


def process_timestamp_data(df: pd.DataFrame, handle_na: str = 'raise') -> pd.DataFrame:
    """
    Выполняет полный пайплайн обработки временных меток.

    Параметры
    ---------
    df : pd.DataFrame
        Исходная таблица с временными метками
    handle_na : str, optional
        Стратегия обработки пустых значений, по умолчанию 'raise'

    Возвращает
    -------
    pd.DataFrame
        Таблица с исходными данными и извлеченными признаками

    Исключения
    ----------
    ValueError
        Если произошла ошибка в процессе обработки
    """
    datetime_df = convert_to_datetime(df, handle_na=handle_na)
    
    features_df = extract_datetime_features(datetime_df)
    
    result_df = pd.concat([datetime_df, features_df], axis=1)
    
    return result_df


def display_results(
    result_df: pd.DataFrame, 
    data_source: str, 
    save_plot: bool = False,
    plot_filename: str = None
) -> None:
    """
    Отображает результаты обработки временных меток.

    Параметры
    ---------
    result_df : pd.DataFrame
        DataFrame с результатами обработки
    data_source : str
        Описание источника данных для информационного вывода
    """
    print(f"=== РЕЗУЛЬТАТЫ ОБРАБОТКИ ({data_source}) ===")
    
    print("1. Исходные данные и признаки (первые 10 строк):")
    print(result_df.head(10))
    print()
    
    time_features = result_df[['is_morning', 'is_afternoon', 'is_evening', 'is_night']]
    counts = time_features.sum()
    total = len(time_features)
    
    print("2. Статистика времени суток:")
    time_labels = {
        "is_morning": "Утро (6-12)",
        "is_afternoon": "День (12-18)", 
        "is_evening": "Вечер (18-24)",
        "is_night": "Ночь (0-6)"
    }
    
    for col, label in time_labels.items():
        count = counts[col]
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"   {label}: {count} записей ({percentage:.1f}%)")
    
    print(f"   Всего записей: {total}")
    print()
    
    # Bar plot
    print("3. Bar plot времени суток:")
    
    if not plot_filename and save_plot:
        # Генерируем уникальное имя файла на основе источника данных
        base_name = data_source.lower().replace(' ', '_').replace(':', '').replace('/', '_')
        plot_filename = f"time_of_day_{base_name}.png"
    
    create_time_of_day_bar_plot(time_features, plot_filename if save_plot else None)


def main():
    """
    Основная функция программы - обрабатывает аргументы командной строки и выполняет пайплайн.

    Исключения
    ----------
    SystemExit
        При корректном завершении работы
    Exception
        При возникновении ошибок в процессе выполнения
    """
    parser = argparse.ArgumentParser(
        description="DA-1-35 + DA-2-34: Работа с временными метками и создание бинарных признаков времени суток."
    )

    group = parser.add_mutually_exclusive_group(required=True)  
    group.add_argument(
        "--example_synthetic",
        action="store_true",
        help="Сгенерировать синтетические временные метки."
    )
    group.add_argument(
        "--file",
        type=str,
        help="Загрузить данные из CSV файла."
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="Сохранить bar plot в файл"
    )
    parser.add_argument(
        "--plot_filename",
        type=str,
        default=None,
        help="Имя файла для сохранения графика (по умолчанию генерируется автоматически)"
    )
    parser.add_argument(
        "--handle_na",
        type=str,
        choices=['raise', 'drop', 'coerce'],
        default='raise',
        help="Стратегия обработки пустых значений: raise - исключение, drop - удаление, coerce - преобразование в NaT"
    )

    args = parser.parse_args()

    try:
        if args.example_synthetic:
            # Синтетические данные
            input_df = create_periodic_dataframe(
                start_timedate_point="2025-09-16 02:35:00",
                periods=15, 
                freq="4h"
            )
            data_source = "СИНТЕТИЧЕСКИЕ ДАННЫЕ"
            show_progress = False
        else:
            input_df = load_dataframe_from_file(args.file, show_progress=True)
            data_source = f"ДАННЫЕ ИЗ ФАЙЛА: {args.file}"
            show_progress = True
        
        result_df = process_timestamp_data(input_df, handle_na=args.handle_na)
        
        display_results(
            result_df, 
            data_source, 
            args.save_plot,
            args.plot_filename
        )
        
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
