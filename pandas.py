import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import Optional, Dict, Any, Tuple, List


warnings.filterwarnings('ignore') # игнорируем предупреждения


class MedicalDeviceAnalyzer:
    """
    Класс для анализа данных медицинского диагностического оборудования.
    """

    def __init__(self, file_path: str) -> None:
        """
        Инициализация анализатора.

        :param file_path: путь к файлу с данными.
        """

        self.file_path = file_path
        self.df = None
        self.current_date = datetime.now()

    def load_data(self) -> pd.DataFrame:
        """
        Загрузка данных из файла.

        :return: pd.DataFrame: загруженный DataFrame.
        """

        self.df = pd.read_excel(self.file_path, sheet_name=0)
        self.df = self.df.drop_duplicates()  # удаляем дубликаты

        # нормализация статусов
        status_mapping = {
            'Operational': 'operational', 'operational': 'operational', 'operational ': 'operational',
            'op': 'operational', 'working': 'operational', 'OK': 'operational',
            'planned_installation': 'planned_installation', 'to_install': 'planned_installation',
            'scheduled_install': 'planned_installation', 'planned': 'planned_installation',
            'maintenance_scheduled': 'maintenance_scheduled', 'maintenance': 'maintenance_scheduled',
            'maint_sched': 'maintenance_scheduled', 'service_scheduled': 'maintenance_scheduled',
            'faulty': 'faulty', 'broken': 'faulty', 'error': 'faulty', 'needs_repair': 'faulty'
        }

        self.df['status'] = self.df['status'].astype(str)
        self.df['status'] = self.df['status'].map(status_mapping)

        # обработка дат
        date_columns = ['install_date', 'warranty_until', 'last_calibration_date', 'last_service_date']
        date_formats = ['%Y-%m-%d', '%d.%m.%Y', '%b %d, %Y']

        for col in date_columns:
            # конвертируем в строки для обработки
            str_dates = self.df[col].astype(str)
            # создаем колонку для результатов
            result = pd.Series(index=self.df.index, dtype='datetime64[ns]')

            for fmt in date_formats:
                # берем только те, что еще не распарсены
                mask = result.isna()
                if mask.any():
                    parsed = pd.to_datetime(str_dates.loc[mask], format=fmt, errors='coerce')
                    result.loc[mask] = parsed

            self.df[col] = result

        # заполняем пропуски
        self.df['issues_text'] = self.df['issues_text'].fillna('')
        self.df['failure_count_12mo'] = pd.to_numeric(self.df['failure_count_12mo'], errors='coerce').fillna(0)
        self.df['issues_reported_12mo'] = pd.to_numeric(self.df['issues_reported_12mo'], errors='coerce').fillna(0)
        self.df['uptime_pct'] = pd.to_numeric(self.df['uptime_pct'], errors='coerce')

        return self.df

    def filter_by_warranty(self) -> pd.DataFrame:
        """
        Фильтрация данных по гарантии.

        :return: pd.DataFrame: отфильтрованный DataFrame с устройствами, у которых гарантия <= 30 дней.
        """

        # рассчитываем оставшиеся дни гарантии
        self.df['warranty_days_left'] = (self.df['warranty_until'] - self.current_date).dt.days

        # категории гарантии
        self.df['warranty_category'] = pd.cut(
            self.df['warranty_days_left'],
            bins=[-np.inf, 0, 30, 90, 180, 365, np.inf],
            labels=['истекла', '< 30 дней', '30-90 дней', '90-180 дней', '180-365 дней', 'больше года']
        )

        # фильтруем устройства с истекшей или истекающей гарантией
        filtered = self.df[self.df['warranty_days_left'] <= 30].copy()

        print("\nРаспределение по статусам гарантии:")
        print(self.df['warranty_category'].value_counts().sort_index())

        return filtered

    def find_clinics_with_most_problems(self, top_n: int = 10) -> pd.DataFrame:
        """
        Нахождение клиник с наибольшим количеством проблем.

        :param top_n: количество клиник для вывода.

        :return: pd.DataFrame: DataFrame с топ клиниками по количеству проблем.
        """

        # определяем устройства с проблемами
        self.df['has_problems'] = (self.df['failure_count_12mo'] > 0) | (self.df['issues_reported_12mo'] > 0)

        # группировка по клиникам
        clinic_stats = self.df.groupby(['clinic_id', 'clinic_name', 'city']).agg({
            'device_id': 'count',
            'has_problems': 'sum',
            'failure_count_12mo': 'sum',
            'issues_reported_12mo': 'sum',
            'uptime_pct': 'mean'
        }).reset_index()

        # меняем названия колонок
        clinic_stats.columns = ['clinic_id', 'clinic_name', 'city', 'total_devices', 'devices_with_problems',
                                'total_failures', 'total_issues_reported', 'avg_uptime']

        clinic_stats['avg_uptime'] = clinic_stats['avg_uptime'].round(2)

        # новый столбец - процент проблемных единиц оборудования
        clinic_stats['problem_percent'] = (clinic_stats['devices_with_problems'] / clinic_stats['total_devices'] * 100)
        clinic_stats['problem_percent'] = clinic_stats['problem_percent'].round(2)

        # топ клиник
        top_clinics = clinic_stats.nlargest(top_n, 'devices_with_problems')
        cols = ['clinic_name', 'city', 'total_devices', 'devices_with_problems', 'total_failures', 'problem_percent']

        print('\nТоп клиник по проблемам')
        print('\n', top_clinics[cols].to_string(index=False))

        return top_clinics

    def calibration_report(self) -> pd.DataFrame:
        """
        Построение отчёта по срокам калибровки.

        :return: pd.DataFrame: DataFrame с отчетом по калибровке.
        """

        # рассчитываем дни с последней калибровки
        self.df['days_since_calibration'] = (self.current_date - self.df['last_calibration_date']).dt.days

        # создаем категории
        bins = [-np.inf, 0, 30, 90, 180, 365, 730, np.inf]
        labels = ['< 30 дней', '30-90 дней', '90-180 дней', '180-365 дней', '1-2 года', '> 2 лет', 'нет данных']

        self.df['calibration_category'] = pd.cut(
            self.df['days_since_calibration'],
            bins=bins,
            labels=labels
        )

        # отчет по калибровке
        report = self.df.groupby('calibration_category').agg({
            'device_id': 'count',
            'failure_count_12mo': 'sum',
            'issues_reported_12mo': 'sum',
            'uptime_pct': 'mean'
        }).reset_index()

        # меняем названия колонок
        report.columns = ['срок с последней калибровки', 'количество устройств', 'количество отказов',
                          'количество проблем', 'среднее время работы']
        report['среднее время работы'] = report['среднее время работы'].round(2)

        print('\nОтчёт')
        print('\n', report.to_string(index=False))

        return report

    def create_pivot_table(self) -> pd.DataFrame:
        """
        Сводная таблица по клиникам и оборудованию.

        :return: pd.DataFrame: сводная таблица.
        """

        # сводная таблица
        pivot = pd.pivot_table(
            self.df,
            values=['device_id', 'failure_count_12mo', 'issues_reported_12mo', 'uptime_pct'],
            index=['clinic_name', 'city', 'model'],
            columns=['status'],
            aggfunc={
                'device_id': 'count',
                'failure_count_12mo': 'sum',
                'issues_reported_12mo': 'sum',
                'uptime_pct': 'mean'
            },
            fill_value=0
        )

        # переименовываем колонки
        pivot.columns = [f'{col[1]}_{col[0]}' for col in pivot.columns]
        pivot = pivot.reset_index()

        # получение всех статусов
        status_cols = [col for col in pivot.columns if '_device_id' in col]

        # расчет итогов
        pivot['total_devices'] = pivot[status_cols].sum(axis=1)

        # расчет отказов и проблем
        failure_cols = [col.replace('_device_id', '_failure_count_12mo') for col in status_cols]
        issues_cols = [col.replace('_device_id', '_issues_reported_12mo') for col in status_cols]

        pivot['total_failures'] = pivot[[col for col in failure_cols if col in pivot.columns]].sum(axis=1)
        pivot['total_issues'] = pivot[[col for col in issues_cols if col in pivot.columns]].sum(axis=1)

        # расчет среднего времени работы (взвешенный)
        uptime_cols = [col.replace('_device_id', '_uptime_pct') for col in status_cols]
        uptime_cols = [col for col in uptime_cols if col in pivot.columns]

        if uptime_cols and len(status_cols) > 0:
            # создаем матрицы для взвешенного среднего
            uptime_values = pivot[uptime_cols].values
            device_counts = pivot[status_cols].values

            # считаем взвешенное среднее (сумма произведений / общее количество)
            weighted_sum = np.sum(uptime_values * device_counts, axis=1)
            total_devices = np.sum(device_counts, axis=1)

            # избегаем деления на ноль
            pivot['avg_uptime'] = np.where(
                total_devices > 0,
                (weighted_sum / total_devices).round(2),
                0
            )
        else:
            pivot['avg_uptime'] = 0

        print("\nПервые 10 строк сводной таблицы:")
        print(pivot.head(10).to_string(index=False))

        return pivot

    def save_reports(self,
                     filtered: pd.DataFrame,
                     top_clinics: pd.DataFrame,
                     cal_report: pd.DataFrame,
                     pivot: pd.DataFrame,
                     filename: str = 'medical_devices_report.xlsx') -> None:
        """
        Сохранение всех отчетов в Excel файл.

        :param filtered: отфильтрованные данные по гарантии.
        :param top_clinics: топ клиник с проблемами.
        :param cal_report: отчет по калибровке.
        :param pivot: сводная таблица.
        :param filename: имя файла для сохранения.
        """

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.df.to_excel(writer, sheet_name='Исходные данные', index=False)
            filtered.to_excel(writer, sheet_name='Фильтр по гарантии', index=False)
            top_clinics.to_excel(writer, sheet_name='Топ проблемных клиник', index=False)
            cal_report.to_excel(writer, sheet_name='Отчет по калибровке', index=False)
            pivot.to_excel(writer, sheet_name='Сводная таблица', index=False)

        print(f"\nВсе отчеты сохранены в файл: {filename}")


if __name__ == "__main__":
    analyzer = MedicalDeviceAnalyzer('medical_diagnostic_devices_10000.xlsx')
    analyzer.load_data()

    filtered = analyzer.filter_by_warranty()
    top_clinics = analyzer.find_clinics_with_most_problems()
    cal_report = analyzer.calibration_report()
    pivot = analyzer.create_pivot_table()

    analyzer.save_reports(filtered, top_clinics, cal_report, pivot)