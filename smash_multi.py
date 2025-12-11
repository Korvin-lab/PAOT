import concurrent.futures
import gc
import os
import time
import warnings
from datetime import datetime
from functools import lru_cache

import joblib
import networkx as nx
import numpy as np
import pandas as pd
# from prof import profiler
from pyvis.network import Network
from tqdm import tqdm
import math

from config import settings
from tech_mode.smash.rough_temperature_model import RoughTemperatureModel

# Отключение всех предупреждений
warnings.filterwarnings("ignore")
    
def convert_files_to_parquet(edges_file, tr_file, tech_mode_file, loc_info_file, 
                           kvch_file, fhs_plasts_file):
    # Оптимизированная конвертация edges.xlsx
    try:
        edges_df = pd.read_excel(edges_file, sheet_name="Лист2")
        # Преобразуем строковые колонки
        str_columns = edges_df.select_dtypes(include=['object']).columns
        for col in str_columns:
            edges_df[col] = edges_df[col].astype(str).replace('nan', '')
        edges_df.to_parquet(edges_file.replace(".xlsx", ".parquet"))
    except Exception as e:
        print(f"Ошибка при конвертации edges_file: {e}")
        raise

    # Оптимизированная конвертация kvch.xlsx
    try:
        kvch_df = pd.read_excel(kvch_file, header=0)
        # Преобразуем числовые колонки
        kvch_df['КВЧ'] = pd.to_numeric(kvch_df['КВЧ'], errors='coerce')
        # Преобразуем строковые колонки
        str_columns = kvch_df.select_dtypes(include=['object']).columns
        for col in str_columns:
            kvch_df[col] = kvch_df[col].astype(str).replace('nan', '')
        kvch_df.to_parquet(kvch_file.replace(".xlsx", ".parquet"))
    except Exception as e:
        print(f"Ошибка при конвертации kvch_file: {e}")
        raise

    # Оптимизированная конвертация fhs_plasts.xlsx
    try:
        fhs_df = pd.read_excel(fhs_plasts_file)
        # Преобразуем числовые колонки
        numeric_columns = ['CO2', 'Минерализация', 'pH']
        for col in numeric_columns:
            if col in fhs_df.columns:
                fhs_df[col] = pd.to_numeric(fhs_df[col], errors='coerce')
        # Преобразуем строковые колонки
        str_columns = fhs_df.select_dtypes(include=['object']).columns
        for col in str_columns:
            fhs_df[col] = fhs_df[col].astype(str).replace('nan', '')
        fhs_df.to_parquet(fhs_plasts_file.replace(".xlsx", ".parquet"))
    except Exception as e:
        print(f"Ошибка при конвертации fhs_plasts_file: {e}")
        raise

    # Оптимизированная конвертация tr_file
    try:
        tr_df = pd.read_csv(tr_file)
        # Преобразуем числовые колонки
        numeric_columns = ['liq_rate', 'oil_rate', 'water_cut', 'gas_factor', 'liquid_viscosity']
        for col in numeric_columns:
            if col in tr_df.columns:
                tr_df[col] = pd.to_numeric(tr_df[col], errors='coerce')
        # Преобразуем строковые колонки
        str_columns = tr_df.select_dtypes(include=['object']).columns
        for col in str_columns:
            tr_df[col] = tr_df[col].astype(str).replace('nan', '')
        tr_df.to_parquet(tr_file.replace(".csv", ".parquet"))
    except Exception as e:
        print(f"Ошибка при конвертации tr_file: {e}")
        raise

    # Оптимизированная конвертация tech_mode_file
    try:
        tech_df = pd.read_csv(tech_mode_file)
        # Преобразуем числовые колонки
        numeric_columns = ['q', 'density_liquid_work', 'density_oil', 'gas_content_rate',
                         'p_start', 'v_liquid_true', 'viscosity_liquid_work']
        for col in numeric_columns:
            if col in tech_df.columns:
                tech_df[col] = pd.to_numeric(tech_df[col], errors='coerce')
        # Преобразуем строковые колонки
        str_columns = tech_df.select_dtypes(include=['object']).columns
        for col in str_columns:
            tech_df[col] = tech_df[col].astype(str).replace('nan', '')
        tech_df.to_parquet(tech_mode_file.replace(".csv", ".parquet"))
    except Exception as e:
        print(f"Ошибка при конвертации tech_mode_file: {e}")
        raise

    # Оптимизированная конвертация loc_info_file
    try:
        loc_df = pd.read_csv(loc_info_file)
        # Преобразуем строковые колонки
        str_columns = loc_df.select_dtypes(include=['object']).columns
        for col in str_columns:
            loc_df[col] = loc_df[col].astype(str).replace('nan', '')
        loc_df.to_parquet(loc_info_file.replace(".csv", ".parquet"))
    except Exception as e:
        print(f"Ошибка при конвертации loc_info_file: {e}")
        raise

    print("Все файлы успешно сконвертированы в формат parquet")

@lru_cache(maxsize=1)
def load_data(edges_file, tr_file, tech_mode_file, loc_info_file, 
              kvch_file, fhs_plasts_file):
    """
    Загружает данные из parquet файлов или конвертирует их из исходных файлов
    """
    if not os.path.exists(edges_file.replace(".xlsx", ".parquet")):
        convert_files_to_parquet(edges_file, tr_file, tech_mode_file,
                               loc_info_file, kvch_file, fhs_plasts_file)
    
    edges_df = pd.read_parquet(edges_file.replace(".xlsx", ".parquet"))
    tr_df = pd.read_parquet(tr_file.replace(".csv", ".parquet"))
    tech_mode_df = pd.read_parquet(tech_mode_file.replace(".csv", ".parquet"))
    loc_info_df = pd.read_parquet(loc_info_file.replace(".csv", ".parquet"))
    kvch_df = pd.read_parquet(kvch_file.replace(".xlsx", ".parquet"))
    fhs_plasts_df = pd.read_parquet(fhs_plasts_file.replace(".xlsx", ".parquet"))
    
    return edges_df, tr_df, tech_mode_df, loc_info_df, kvch_df, fhs_plasts_df

def preprocess_data(all_edges_df, all_tr_df, all_tech_mode_df, all_loc_info_df,
                   all_kvch_df, all_fhs_plasts_df, all_ingib_df,
                   first_name, second_name, target_date):
    """
    Оптимизированная функция предобработки данных
    """
    # Фильтрация edges_df - используем boolean indexing вместо copy()
    mask = (all_edges_df["Местонахождение_Месторождение"] == first_name) & \
           (all_edges_df["Местонахождение_Принадлежность_к_объекту_(площадка)"] == second_name)
    edges_df = all_edges_df[mask]

    # Фильтрация loc_info_df без создания копии
    loc_info_df = all_loc_info_df[all_loc_info_df["field"] == first_name]
    
    # Преобразование и фильтрация tr_df
    if not pd.api.types.is_datetime64_any_dtype(all_tr_df["calc_date"]):
        all_tr_df["calc_date"] = pd.to_datetime(all_tr_df["calc_date"])
    tr_df = all_tr_df[all_tr_df["calc_date"].dt.to_period("D") == target_date]

    # Оптимизированное преобразование fhs_plasts_df
    first_col_name = all_fhs_plasts_df.columns[0]
    selected_columns = all_fhs_plasts_df.columns[[1, 2, 3, 8, 9]]
    
    # Объединяем фильтрацию и выбор колонок в одну операцию
    fhs_plasts_df = (all_fhs_plasts_df[all_fhs_plasts_df[first_col_name] == first_name]
                     [selected_columns]
                     .assign(Пласт=lambda x: x.iloc[:, 0].str.split(".").str[0])
                     .rename(columns={selected_columns[0]: "Пласт",
                                    selected_columns[1]: "Код пласта",
                                    selected_columns[2]: "CO2",
                                    selected_columns[3]: "Минерализация",
                                    selected_columns[4]: "pH"}))
    
    # Преобразуем числовые колонки одним вызовом
    numeric_cols = ["CO2", "Минерализация", "pH"]
    fhs_plasts_df[numeric_cols] = fhs_plasts_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    fhs_plasts_df = fhs_plasts_df.dropna()

    # Оптимизированное объединение tr_df
    tr_df["well_id"] = tr_df["well_id"].astype(str).str.split(".").str[0]
    loc_info_df["well_id"] = loc_info_df["well_id"].astype(str).str.split(".").str[0]
    
    # Цепочка merge операций
    tr_df = (tr_df.merge(loc_info_df[["well_id", "kp"]], on="well_id", how="left")
             .merge(fhs_plasts_df[["Код пласта", "Пласт"]], 
                   left_on="layer_id", right_on="Код пласта", how="left"))
    
    tr_df = tr_df.dropna(subset=["kp"])
    tr_df["plast_processed"] = tr_df["Пласт"].astype(str).str.split(".").str[0]

    # Оптимизированная агрегация для kusts_df с использованием agg
    agg_dict = {
        "liq_rate": "sum",
        "oil_rate": "sum",
        "water_cut": "mean",
        "gas_factor": "sum",
        "liquid_viscosity": "mean",
        "plast_processed": lambda x: list(pd.unique(x.dropna()))
    }
    
    kusts_df = (tr_df.groupby("kp")
                .agg(agg_dict)
                .reset_index()
                .rename(columns={"plast_processed": "plast"}))
    
    kusts_df["kp"] = kusts_df["kp"].str.lower()
    kusts_df = kusts_df.fillna(0)

    # Оптимизированная агрегация для wells_df
    wells_agg_dict = {
        "liq_rate": "sum",
        "oil_rate": "sum",
        "water_cut": "mean",
        "gas_factor": "sum",
        "Пласт": "first",
        "liquid_viscosity": "mean",
        "temperature": "mean",
    }
    
    wells_df = (tr_df.groupby("well_id")
                .agg(wells_agg_dict)
                .reset_index())
    
    wells_df["plast"] = wells_df["Пласт"].astype(str).str.split(".").str[0]
    wells_df = wells_df.merge(loc_info_df[["well_id", "well_num"]], 
                             on="well_id", how="left")
    wells_df["well_num"] = wells_df["well_num"].astype(str).str.lower()

    # Оптимизированная обработка kvch_df
    if not pd.api.types.is_datetime64_any_dtype(all_kvch_df["Дата отбора пробы"]):
        all_kvch_df["Дата отбора пробы"] = pd.to_datetime(all_kvch_df["Дата отбора пробы"])
    
    kvch_mask = ((all_kvch_df["Месторождение"] == first_name) & 
                 (all_kvch_df["Дата отбора пробы"].dt.to_period("D") == target_date))
    kvch_df = all_kvch_df[kvch_mask]

    # Оптимизированное получение средних значений КВЧ
    kvch_mean = kvch_df.groupby("Скважина")["КВЧ"].mean()
    wells_df["КВЧ"] = wells_df["well_num"].map(kvch_mean)
    
    # Заполнение пропусков средним значением КВЧ по пластам
    kvch_mean_plast = wells_df.groupby("plast")["КВЧ"].mean()
    wells_df["КВЧ"] = wells_df["КВЧ"].fillna(wells_df["plast"].map(kvch_mean_plast))
    wells_df = wells_df.fillna(0)

    # Оптимизированное преобразование edges_df
    edges_df["начало"] = edges_df["начало"].str.lower().str.replace(" ", "")
    edges_df["конец"] = edges_df["конец"].str.lower().str.replace(" ", "")
    loc_info_df["kp"] = loc_info_df["kp"].str.lower().str.replace(" ", "")

    # Создание новых строк для edges_df одной операцией
    merged_df = edges_df.merge(loc_info_df[["kp", "well_num", "well_id"]], 
                              left_on="начало", right_on="kp", how="inner")
    
    new_rows_df = pd.DataFrame({
        "начало": merged_df["well_num"].astype(str) + "_",
        "конец": merged_df["начало"],
        "Данные_по_протяжённости_в_структуре_сети_OISPIPE_Паспортизация_ID_простого_участка_трубопровода": 
            merged_df["well_id"].astype(str)
    })
    
    edges_df = pd.concat([edges_df, new_rows_df], ignore_index=True)
    
    # Оптимизированное преобразование tech_mode_df
    if not pd.api.types.is_datetime64_any_dtype(all_tech_mode_df["calc_date"]):
        all_tech_mode_df["calc_date"] = pd.to_datetime(all_tech_mode_df["calc_date"])
    
    tech_mode_df = all_tech_mode_df[
        all_tech_mode_df["calc_date"].dt.to_period("D") == target_date
    ]
    
    # Оптимизированное объединение с edges_df
    # Явно приводим типы данных к одному формату
    tech_mode_df["simple_section_id"] = tech_mode_df["simple_section_id"].astype(str)
    edges_df["Данные_по_протяжённости_в_структуре_сети_OISPIPE_Паспортизация_ID_простого_участка_трубопровода"] = (
        edges_df["Данные_по_протяжённости_в_структуре_сети_OISPIPE_Паспортизация_ID_простого_участка_трубопровода"]
        .astype(str)
    )
    
    avg_q = tech_mode_df.groupby("simple_section_id")["q"].mean()
    tech_mode_df = (tech_mode_df.merge(avg_q.rename("avg_q"), 
                                      on="simple_section_id", how="left")
                    .merge(edges_df,
                          left_on="simple_section_id",
                          right_on="Данные_по_протяжённости_в_структуре_сети_OISPIPE_Паспортизация_ID_простого_участка_трубопровода",
                          how="left"))
    tech_mode_df['simple_section_id'] = tech_mode_df['simple_section_id'].astype(float).astype(str)

    # Оптимизированное преобразование ingib_df
    if not pd.api.types.is_datetime64_any_dtype(all_ingib_df["Дата"]):
        all_ingib_df["Дата"] = pd.to_datetime(all_ingib_df["Дата"])
    
    ingib_mask = ((all_ingib_df["Месторождение"] == first_name) & 
                  (all_ingib_df["Дата"].dt.to_period("D") == target_date))
    ingib_df = all_ingib_df[ingib_mask]
    
    ingib_df["Куст"] = ingib_df["Куст"].astype(str).str.lower().str.replace(" ", "")
    ingib_df["Принятие к оплате"] = ingib_df["Принятие к оплате"].clip(lower=0)

    # Очистка памяти
    del all_edges_df, all_tr_df, all_tech_mode_df, all_loc_info_df, all_kvch_df, all_fhs_plasts_df
    gc.collect()

    return edges_df, kusts_df, wells_df, tech_mode_df, fhs_plasts_df, ingib_df


def build_graph(edges_df):
    """
    Строит направленный граф из DataFrame с рёбрами.
    Оптимизированная версия с прямым созданием графа из списка рёбер.
    
    Args:
        edges_df (pd.DataFrame): DataFrame с колонками 'начало', 'конец' и данными по рёбрам
        
    Returns:
        nx.DiGraph: Построенный направленный граф
    """
    # Фильтруем рёбра где начало не равно концу
    mask = edges_df["начало"].str.strip() != edges_df["конец"].str.strip()
    
    # Создаем список рёбер и их атрибутов напрямую
    edges_and_attrs = [
        (row.начало.strip(), 
         row.конец.strip(), 
         {"id": row.Данные_по_протяжённости_в_структуре_сети_OISPIPE_Паспортизация_ID_простого_участка_трубопровода})
        for row in edges_df[mask].itertuples(index=False)
    ]
    
    # Создаем граф и добавляем все рёбра одной операцией
    G = nx.DiGraph()
    G.add_edges_from(edges_and_attrs)
    
    return G


def calculate_flow(G, tech_mode_df, kusts_df, wells_df, fhs_plasts_df, ingib_df, pipe_info):
    kust_dict = kusts_df.set_index("kp").to_dict("index")
    wells_dict = wells_df.set_index("well_num").to_dict("index")
    edge_q_dict = tech_mode_df.set_index('simple_section_id')['avg_q'].to_dict()
    
    node_attrs = {}
    edge_attrs = {}
    
    for node in nx.topological_sort(G):
        node_str = str(node).lower()
        if node_str in kust_dict:
            kust_data = kust_dict[node_str]
            wells_data = {}
            plasts = []
            
            # Собираем данные о предшественниках (скважинах)
            for pred in G.predecessors(node):
                if "_" in pred:
                    well_name = pred.strip().replace("_", "").lower()
                    data = wells_dict.get(well_name, {})
                    if not data:
                        continue
                        
                    # Извлекаем данные один раз и сохраняем в переменных
                    liq_rate = data.get("liq_rate", 0)
                    oil_rate = data.get("oil_rate", 0)
                    water_cut = data.get("water_cut", 0)
                    kvch = data.get("КВЧ", 0) if not pd.isna(data.get("КВЧ", 0)) else 0
                    gas_factor = data.get("gas_factor", 0)
                    liq_viscosity = data.get("liquid_viscosity", 0)
                    plast = data.get("plast", None)
                    temperature = data.get("temperature", 0)
                    
                    wells_data[pred] = (
                        liq_rate,
                        oil_rate,
                        water_cut,
                        kvch,
                        plast,
                        gas_factor,
                        liq_viscosity,
                        temperature
                    )
                    plasts.append(plast)

            wells_data_items = list(wells_data.items())
            
            # Определяем пласты
            plasts = (
                kust_data["plast"]
                if not any(isinstance(item, str) for item in plasts)
                else plasts
            )
            plasts = pd.Series(plasts).dropna().unique().tolist()

            # Вычисляем ratio
            ratio = {}
            kust_liq_rate = kust_data["liq_rate"]
            kust_oil_rate = kust_data["oil_rate"]
            kust_gas_factor = kust_data["gas_factor"]
            
            for key, value in wells_data_items:
                if key not in ratio:
                    ratio[key] = {}
                    ratio[key]["liq_rate"] = (value[0] / kust_liq_rate * 100) if kust_liq_rate != 0 else 0
                    ratio[key]["oil_rate"] = (value[1] / kust_oil_rate * 100) if kust_oil_rate != 0 else 0
                    ratio[key]["gas_factor"] = (value[1] / kust_gas_factor * 100) if kust_gas_factor != 0 else 0
                    ratio[key]["plast"] = value[4]

            # Вычисляем агрегированные значения
            Qv = Qn = Qzh = sum_kvch = Qgas = liq_viscosity = temperature = 0
            for item in wells_data_items:
                liq_rate, oil_rate, water_cut, kvch, plast, gas_factor, liq_visc, temp = item[1]
                Qzh += liq_rate
                Qn += oil_rate
                Qv += liq_rate * water_cut / 100
                sum_kvch += liq_rate * water_cut / 100 * kvch
                Qgas += oil_rate * gas_factor
                liq_viscosity += liq_rate * liq_visc
                
            for item in wells_data_items:
                liq_rate, oil_rate, water_cut, kvch, plast, gas_factor, liq_visc, temp = item[1]
                temperature += temp * liq_rate / Qzh if Qzh != 0 else 0

            water_cut = (Qv * 100 / Qzh) if Qzh != 0 else kust_data.get("water_cut", 0)
            kvch = (sum_kvch / Qv) if Qv != 0 else 0
            liq_viscosity = (liq_viscosity / (Qzh * 1000)) if Qzh != 0 else 0

            q_oil_mass_rate = Qn # kg/s
            q_wat_mass_rate = Qv # kg/s
            q_gas_mass_rate = Qgas # kg/s
            HEAT = (2200, 2100, 4200)  # HTC, CO, CG, CW
            mass_rate = q_oil_mass_rate + q_gas_mass_rate + q_wat_mass_rate
            if mass_rate != 0:
                c_p = HEAT[0] * (q_oil_mass_rate / mass_rate) + HEAT[2] * (q_wat_mass_rate / mass_rate) + HEAT[1] * (
                            q_gas_mass_rate / mass_rate)
                length = 2800 # m - берем из нового датафрейма
                tid = 0.168 # m внутренний диаметр трубы
                surrounding_temperature = 5 # типо константа
                HTC = 16000 # тоже константа
                tir = 5e-5 # шероховатость трубы (константа)
                # T_INPUT = 58  # degrees C # Температура со стороны источников
                q_heat_in_by_degree = mass_rate*c_p # Вт

                t_input = temperature

                solution = RoughTemperatureModel(length, tid, tir, q_heat_in_by_degree, surrounding_temperature, HTC, t_input, direction=1).run()

                temperature = solution(0)
            else:
                temperature = 0

            # Группируем ratio по пластам
            grouped_ratio = {}
            for key, value in ratio.items():
                plast = value["plast"]
                if plast not in grouped_ratio and plast is not None:
                    grouped_ratio[plast] = {"liq_rate": 0, "oil_rate": 0, "gas_factor": 0}
                if plast is not None:
                    grouped_ratio[plast]["liq_rate"] += value["liq_rate"]
                    grouped_ratio[plast]["oil_rate"] += value["oil_rate"]
                    grouped_ratio[plast]["gas_factor"] += value["gas_factor"]

            # Вычисляем CO2, Min, pH
            sum_CO2 = sum_Min = sum_pH = 0
            for plast in plasts:
                try:
                    sum_CO2 += ((grouped_ratio[plast]["gas_factor"] / 100) * grouped_ratio[plast]["oil_rate"] / (0.86 * 1000)) * \
                               fhs_plasts_df[fhs_plasts_df["Пласт"] == plast]["CO2"].iloc[0]
                except:
                    sum_CO2 = 0
                try:
                    sum_Min += (grouped_ratio[plast]["liq_rate"] / 100) * \
                               fhs_plasts_df[fhs_plasts_df["Пласт"] == plast]["Минерализация"].iloc[0]
                except:
                    sum_Min = 0
                try:
                    sum_pH += (grouped_ratio[plast]["liq_rate"] / 100) * \
                              fhs_plasts_df[fhs_plasts_df["Пласт"] == plast]["pH"].iloc[0]
                except:
                    sum_pH = 0

            # Вычисляем ing_sum
            try:
                ing_sum = (
                    ingib_df[ingib_df["Куст"] == node_str]["Принятие к оплате"].iloc[0]
                    * 0.94
                    * 1000
                    / Qzh
                    if Qzh != 0
                    else 0
                )
            except:
                ing_sum = 0

            # Сохраняем атрибуты узла
            node_attrs[node] = {
                "liq_rate": kust_data["liq_rate"],
                "oil_rate": kust_data["oil_rate"],
                "water_cut": water_cut,
                "plasts": plasts,
                "Qzh": Qzh,
                "Qv": Qv,
                "Qn": Qn,
                "Qgas": Qgas,
                "liq_viscosity": liq_viscosity,
                "kvch": kvch,
                "ratio": ratio,
                "CO2": sum_CO2,
                "Min": sum_Min,
                "pH": sum_pH,
                "ing_sum": ing_sum,
                "temperature": temperature,
            }

    # Применяем все атрибуты узлов одним вызовом
    nx.set_node_attributes(G, node_attrs)
    # Создаем словарь для быстрого поиска avg_q по edge_id
    edge_q_dict = tech_mode_df.set_index("simple_section_id")["avg_q"].to_dict()

    for node in nx.topological_sort(G):
        if "liq_rate" not in G.nodes[node] or "oil_rate" not in G.nodes[node]:
            continue

        out_edges = list(G.out_edges(node, data=True))
        if not out_edges:
            continue

        total_q = 0
        valid_edges = []
        edges_without_tech_mode = []

        for edge in out_edges:
            edge_id = edge[2]["id"]
            edge_q = edge_q_dict.get(edge_id, 0)
            if edge_q > 0:
                total_q += edge_q
                valid_edges.append((edge, edge_q))
            else:
                edges_without_tech_mode.append(edge)

        avg_q = total_q / len(valid_edges) if valid_edges else 1

        edges_without_tech_mode = [
            edge
            for edge in edges_without_tech_mode
            if str(edge[2]["id"]).startswith("125")
        ]  # Дроп id скважин

        for edge in edges_without_tech_mode:
            valid_edges.append((edge, avg_q))
            total_q += avg_q

        if not valid_edges:
            continue

        for edge, weight in valid_edges:
            target_node = edge[1]
            if (
                str(target_node).lower() in kust_dict
            ):  # Расчет значений для куста (свои значения + предыдущие кусты)
                if (
                    G.nodes[target_node].get("Qv", 0) + G.nodes[node].get("Qv", 0)
                ) != 0:
                    G.nodes[target_node]["water_cut"] = (
                        G.nodes[target_node].get("water_cut", 0)
                        * G.nodes[target_node].get("Qv", 0)
                        + G.nodes[node]["water_cut"] * G.nodes[node].get("Qv", 0)
                    ) / (G.nodes[target_node].get("Qv", 0) + G.nodes[node].get("Qv", 0))

                    G.nodes[target_node]["kvch"] = (
                        G.nodes[target_node].get("kvch", 0)
                        * G.nodes[target_node].get("Qv", 0)
                        + G.nodes[node]["kvch"] * G.nodes[node].get("Qv", 0)
                    ) / (G.nodes[target_node].get("Qv", 0) + G.nodes[node].get("Qv", 0))

                    G.nodes[target_node]["CO2"] = (
                        G.nodes[target_node].get("CO2", 0)
                        * G.nodes[target_node].get("Qgas", 0)
                        + G.nodes[node]["CO2"] * G.nodes[node].get("Qgas", 0)
                    ) / (G.nodes[target_node].get("Qgas", 0) + G.nodes[node].get("Qgas", 0))

                    G.nodes[target_node]["Min"] = (
                        G.nodes[target_node].get("Min", 0)
                        * G.nodes[target_node].get("Qv", 0)
                        + G.nodes[node]["Min"] * G.nodes[node].get("Qv", 0)
                    ) / (G.nodes[target_node].get("Qv", 0) + G.nodes[node].get("Qv", 0))

                    G.nodes[target_node]["pH"] = (
                        G.nodes[target_node].get("pH", 0)
                        * G.nodes[target_node].get("Qv", 0)
                        + G.nodes[node]["pH"] * G.nodes[node].get("Qv", 0)
                    ) / (G.nodes[target_node].get("Qv", 0) + G.nodes[node].get("Qv", 0))

                    if (
                        G.nodes[target_node].get("Qzh", 0) + G.nodes[node].get("Qzh", 0)
                    ) != 0:
                        G.nodes[target_node]["liq_viscosity"] = (
                            G.nodes[target_node].get("liq_viscosity", 0)
                            * G.nodes[target_node].get("Qzh", 0)
                            + G.nodes[node]["liq_viscosity"]
                            * G.nodes[node].get("Qzh", 0)
                        ) / (
                            G.nodes[target_node].get("Qzh", 0)
                            + G.nodes[node].get("Qzh", 0)
                        )

                        G.nodes[target_node]["temperature"] = (
                            G.nodes[target_node].get("temperature", 0)
                            * G.nodes[target_node].get("Qzh", 0)
                            + G.nodes[node]["temperature"] * G.nodes[node].get("Qzh", 0)
                        ) / (G.nodes[target_node].get("Qzh", 0) + G.nodes[node].get("Qzh", 0))
                        
                else:
                    G.nodes[target_node]["water_cut"] = 0
                    G.nodes[target_node]["kvch"] = 0
                    G.nodes[target_node]["CO2"] = 0
                    G.nodes[target_node]["Min"] = 0
                    G.nodes[target_node]["pH"] = 0
                    G.nodes[target_node]["liq_viscosity"] = 0
                    G.nodes[target_node]["temperature"] = 0

                G.nodes[target_node]["ing_sum"] = (
                    (
                        (
                            G.nodes[target_node].get("ing_sum", 0)
                            * G.nodes[target_node].get("Qzh", 0)
                            + G.nodes[node]["ing_sum"] * G.nodes[node].get("Qzh", 0)
                        )
                        / (
                            G.nodes[target_node].get("Qzh", 0)
                            + G.nodes[node].get("Qzh", 0)
                        )
                    )
                    if (
                        G.nodes[target_node].get("Qzh", 0) + G.nodes[node].get("Qzh", 0)
                    )
                    != 0
                    else 0
                )
                G.nodes[target_node]["Qgas"] = (
                    (
                        (
                            G.nodes[target_node].get("Qgas", 0)
                            * G.nodes[target_node].get("Qn", 0)
                            + G.nodes[node]["Qgas"] * G.nodes[node].get("Qn", 0)
                        )
                        / (
                            G.nodes[target_node].get("Qn", 0)
                            + G.nodes[node].get("Qn", 0)
                        )
                    )
                    if (G.nodes[target_node].get("Qn", 0) + G.nodes[node].get("Qn", 0))
                    != 0
                    else 0
                )
            else:
                sum_Qn = 0
                sum_Qzh = 0
                sum_Qv = 0
                sum_Qgas = 0
                sum_liq_viscosity = 0
                sum_water_cut = 0
                sum_kvch = 0
                sum_CO2 = 0
                sum_Min = 0
                sum_pH = 0
                ing_sum = 0
                sum_temp = 0
                for pred in G.predecessors(target_node):
                    sum_Qv += G.nodes[pred].get("Qv", 0)
                    sum_Qzh += G.nodes[pred].get("Qzh", 0)
                    sum_Qn += G.nodes[pred].get("Qn", 0)
                    sum_Qgas += G.nodes[pred].get("Qgas", 0)
                    sum_water_cut += G.nodes[pred].get("Qv", 0) * G.nodes[pred].get(
                        "water_cut", 0
                    )
                    sum_kvch += G.nodes[pred].get("Qv", 0) * G.nodes[pred].get(
                        "kvch", 0
                    )
                    sum_CO2 += G.nodes[pred].get("Qgas", 0) * G.nodes[pred].get("CO2", 0)
                    sum_Min += G.nodes[pred].get("Qv", 0) * G.nodes[pred].get("Min", 0)
                    sum_pH += G.nodes[pred].get("Qv", 0) * G.nodes[pred].get("pH", 0)
                    ing_sum += G.nodes[pred].get("Qzh", 0) * G.nodes[pred].get(
                        "ing_sum", 0
                    )
                    sum_liq_viscosity += G.nodes[pred].get("Qzh", 0) * G.nodes[
                        pred
                    ].get("liq_viscosity", 0)

                    sum_temp += G.nodes[pred].get("Qzh", 0) * G.nodes[pred].get(
                        "temperature", 0
                    )

                G.nodes[target_node]["water_cut"] = (
                    (sum_water_cut / sum_Qv) if sum_Qv != 0 else 0
                )
                G.nodes[target_node]["kvch"] = (sum_kvch / sum_Qv) if sum_Qv != 0 else 0
                G.nodes[target_node]["CO2"] = (sum_CO2 / sum_Qgas) if sum_Qgas != 0 else 0
                G.nodes[target_node]["Min"] = (sum_Min / sum_Qv) if sum_Qv != 0 else 0
                G.nodes[target_node]["pH"] = (sum_pH / sum_Qv) if sum_Qv != 0 else 0
                G.nodes[target_node]["ing_sum"] = (
                    (ing_sum / sum_Qzh) if sum_Qzh != 0 else 0
                )
                G.nodes[target_node]["liq_viscosity"] = (
                    (sum_liq_viscosity / sum_Qzh) if sum_Qzh != 0 else 0
                )

                q_oil_mass_rate = sum_Qn # kg/s
                q_wat_mass_rate = sum_Qv # kg/s
                q_gas_mass_rate = sum_Qgas # kg/s
                HEAT = (2200, 2100, 4200)  # HTC, CO, CG, CW
                mass_rate = q_oil_mass_rate + q_gas_mass_rate + q_wat_mass_rate
                if mass_rate != 0:
                    c_p = HEAT[0] * (q_oil_mass_rate / mass_rate) + HEAT[2] * (q_wat_mass_rate / mass_rate) + HEAT[1] * (
                                q_gas_mass_rate / mass_rate)
                    edge_id = edge[2]["id"]
                    row = tech_mode_df.loc[
                        tech_mode_df["simple_section_id"] == edge_id,
                        ["length", "d_mm"]
                    ]

                    if not row.empty:
                        length = pd.to_numeric(row["length"], errors="coerce").iloc[0]
                        tid = pd.to_numeric(row["d_mm"], errors="coerce").iloc[0]
                    else:
                        row = pipe_info.loc[
                            pipe_info["simple_section_id"] == edge_id,
                            ["length", "d_mm"]
                        ]
                        if not row.empty:
                            length = pd.to_numeric(row["length"], errors="coerce").iloc[0]
                            tid = pd.to_numeric(row["d_mm"], errors="coerce").iloc[0]

                    if isinstance(length, float) and math.isnan(length):
                        length = 1020 # средние значения из всего датасета

                    if isinstance(tid, float) and math.isnan(tid):
                        tid = 0.22 # средние значения из всего датасета


                    surrounding_temperature = 5 # типо константа
                    HTC = 15000 # тоже константа
                    tir = 5e-5 # шероховатость трубы (константа)
                    # T_INPUT = 58  # degrees C # Температура со стороны источников
                    q_heat_in_by_degree = mass_rate*c_p # Вт

                    t_input = (sum_temp / sum_Qzh) if sum_Qzh != 0 else 0

                    solution = RoughTemperatureModel(length, tid, tir, q_heat_in_by_degree, surrounding_temperature, HTC, t_input, direction=1).run()

                    G.nodes[target_node]["temperature"] = solution(0)
                    
                    if edge_id in ['1250001862.0', '1250001863.0', '1250004481.0']:
                        print(f"id: {edge_id}")
                        # print(row)
                        print(f"length: {length}")
                        print(f"tid: {tid}")
                        # print(f"tir: {tir}")
                        print(f"q_heat_in_by_degree: {q_heat_in_by_degree}")
                        # print(f"surrounding_temperature: {surrounding_temperature}")
                        # print(f"HTC: {HTC}")
                        # print(f"mass_rate: {mass_rate}")
                        # print(f"c_p: {c_p}")
                        # print(f"q_oil_mass_rate: {q_oil_mass_rate}")
                        # print(f"q_wat_mass_rate: {q_wat_mass_rate}")
                        # print(f"q_gas_mass_rate: {q_gas_mass_rate}")
                        print(f"t_input: {t_input}")
                        print(f"solution: {solution(0)}")
                else:
                    G.nodes[target_node]["temperature"] = 0

            G.edges[edge[:2]]["water_cut"] = G.nodes[node]["water_cut"]
            G.edges[edge[:2]]["kvch"] = G.nodes[node]["kvch"]
            G.edges[edge[:2]]["CO2"] = G.nodes[node]["CO2"]
            G.edges[edge[:2]]["Min"] = G.nodes[node]["Min"]
            G.edges[edge[:2]]["pH"] = G.nodes[node]["pH"]
            G.edges[edge[:2]]["ing_sum"] = G.nodes[node]["ing_sum"]
            G.edges[edge[:2]]["liq_viscosity"] = G.nodes[node]["liq_viscosity"]
            G.edges[edge[:2]]["temperature"] = G.nodes[node]["temperature"]

        for edge, weight in valid_edges:  # Расчет разливов по соседям через q_ratio
            q_ratio = weight / total_q
            target_node = edge[1]

            G.nodes[target_node]["liq_rate"] = (
                G.nodes[target_node].get("liq_rate", 0)
                + G.nodes[node]["liq_rate"] * q_ratio
            )
            G.nodes[target_node]["oil_rate"] = (
                G.nodes[target_node].get("oil_rate", 0)
                + G.nodes[node]["oil_rate"] * q_ratio
            )

            G.nodes[target_node]["Qv"] = (
                G.nodes[target_node].get("Qv", 0) + G.nodes[node]["Qv"] * q_ratio
            )
            G.nodes[target_node]["Qzh"] = (
                G.nodes[target_node].get("Qzh", 0) + G.nodes[node]["Qzh"] * q_ratio
            )
            G.nodes[target_node]["Qn"] = (
                G.nodes[target_node].get("Qn", 0) + G.nodes[node]["Qn"] * q_ratio
            )
            G.nodes[target_node]["Qgas"] = (
                G.nodes[target_node].get("Qgas", 0) + G.nodes[node]["Qgas"] * q_ratio
            )

            G.edges[edge[:2]]["liq_rate"] = G.nodes[node]["liq_rate"] * q_ratio
            G.edges[edge[:2]]["oil_rate"] = G.nodes[node]["oil_rate"] * q_ratio
            G.edges[edge[:2]]["Qv"] = G.nodes[node]["Qv"] * q_ratio
            G.edges[edge[:2]]["Qzh"] = G.nodes[node]["Qzh"] * q_ratio
            G.edges[edge[:2]]["Qn"] = G.nodes[node]["Qn"] * q_ratio
            G.edges[edge[:2]]["Qgas"] = G.nodes[node]["Qgas"] * q_ratio

            edge_id = edge[2]["id"]
            if edge_id in tech_mode_df["simple_section_id"].values:
                density_liquid_work = tech_mode_df[
                    tech_mode_df["simple_section_id"] == edge_id
                ]["density_liquid_work"].iloc[0]
                density_oil = tech_mode_df[
                    tech_mode_df["simple_section_id"] == edge_id
                ]["density_oil"].iloc[0]
                gas_content_true = tech_mode_df[
                    tech_mode_df["simple_section_id"] == edge_id
                ]["gas_content_rate"].iloc[0]
                p_start = tech_mode_df[tech_mode_df["simple_section_id"] == edge_id][
                    "p_start"
                ].iloc[0]
                v_liquid_true = tech_mode_df[
                    tech_mode_df["simple_section_id"] == edge_id
                ]["v_liquid_true"].iloc[0]
                # viscosity_liquid_work = tech_mode_df[
                #     tech_mode_df["simple_section_id"] == edge_id
                # ]["viscosity_liquid_work"].iloc[0]
                viscosity_liquid_work = G.edges[edge[:2]]["liq_viscosity"]
                if pd.isna(density_liquid_work):
                    loaded_model = joblib.load(
                        "/tmp/tmp_predict_files/files/density_liquid_work.pkl"
                    )
                    X = np.array(
                        [G.edges[edge[:2]]["Qn"], G.edges[edge[:2]]["water_cut"]]
                    ).reshape(1, -1)
                    G.edges[edge[:2]]["density_liquid_work"] = loaded_model.predict(X)
                else:
                    G.edges[edge[:2]]["density_liquid_work"] = density_liquid_work

                if pd.isna(density_oil):
                    G.edges[edge[:2]]["density_oil"] = 0.86
                else:
                    G.edges[edge[:2]]["density_oil"] = density_oil

                if pd.isna(gas_content_true):
                    loaded_model = joblib.load(
                        "/tmp/tmp_predict_files/files/gas_content_true.pkl"
                    )
                    X = np.array(
                        [G.edges[edge[:2]]["Qzh"], G.edges[edge[:2]]["Qgas"]]
                    ).reshape(1, -1)
                    G.edges[edge[:2]]["gas_content_true"] = loaded_model.predict(X)
                else:
                    G.edges[edge[:2]]["gas_content_true"] = gas_content_true

                if pd.isna(p_start):
                    loaded_model = joblib.load(
                        "/tmp/tmp_predict_files/files/p_start.pkl"
                    )
                    X = np.array(
                        [
                            G.edges[edge[:2]]["Qzh"],
                            G.edges[edge[:2]]["Qgas"],
                            tech_mode_df[tech_mode_df["simple_section_id"] == edge_id][
                                "Общие_данные_D,_мм"
                            ].iloc[0],
                        ]
                    ).reshape(1, -1)
                    G.edges[edge[:2]]["p_start"] = loaded_model.predict(X)
                else:
                    G.edges[edge[:2]]["p_start"] = p_start

                if pd.isna(v_liquid_true):
                    loaded_model = joblib.load(
                        "/tmp/tmp_predict_files/files/v_liquid_true.pkl"
                    )
                    X = np.array(
                        [G.edges[edge[:2]]["Qzh"], G.edges[edge[:2]]["Qgas"]]
                    ).reshape(1, -1)
                    G.edges[edge[:2]]["v_liquid_true"] = loaded_model.predict(X)
                else:
                    G.edges[edge[:2]]["v_liquid_true"] = v_liquid_true

                if pd.isna(viscosity_liquid_work):
                    print(edge_id)
                    viscosity_liquid_work = 0

                G.edges[edge[:2]]["reynolds"] = (
                    (
                        density_liquid_work
                        * v_liquid_true
                        * tech_mode_df[tech_mode_df["simple_section_id"] == edge_id][
                            "Общие_данные_D,_мм"
                        ].iloc[0]
                        / viscosity_liquid_work
                    )
                    if viscosity_liquid_work != 0
                    else 0
                )
                G.edges[edge[:2]]["fruda"] = (
                    v_liquid_true**2
                    / tech_mode_df[tech_mode_df["simple_section_id"] == edge_id][
                        "Общие_данные_D,_мм"
                    ].iloc[0]
                    * 100
                )


def get_active_subgraph(G, edges_df, kusts_df):
    k_names = kusts_df["kp"].unique()
    unique_nodes = set(edges_df["начало"]).union(set(edges_df["конец"]))
    unique_nodes = set(map(str, unique_nodes))
    unique_nodes = set(map(str.strip, unique_nodes))
    unique_nodes = set(map(str.lower, unique_nodes))

    working_kusts = unique_nodes.intersection(k_names)
    working_kusts = set(*[list(working_kusts)])
    connected_nodes = set()

    for row in edges_df.itertuples():
        if row.конец in working_kusts:
            connected_nodes.add(row.начало)

    for node in working_kusts:
        if node in G:
            connected_nodes.add(node)
            connected_nodes.update(nx.descendants(G, node))

    subgraph = G.subgraph(connected_nodes).copy()
    for node in subgraph.nodes():
        node = str(node).lower()
        if node in working_kusts:
            node_params = kusts_df[kusts_df["kp"] == node][
                ["liq_rate", "oil_rate"]
            ].to_dict("records")[0]
            nx.set_node_attributes(subgraph, {node: node_params})
    return subgraph


def find_and_remove_cycles(G, tech_mode_df):
    removed_edges = []
    while True:
        try:
            cycle = nx.find_cycle(G, orientation="original")
            for edge in cycle:
                u, v = edge[:2]
                edge_data = G.edges[u, v]
                edge_id = edge_data.get("id")
                if edge_id in tech_mode_df["simple_section_id"].values:
                    q = tech_mode_df[tech_mode_df["simple_section_id"] == edge_id]["avg_q"].iloc[0]
                    if q == 0:
                        G.remove_edge(u, v)
                        removed_edges.append((u, v, edge_data))
                        break
            else:
                u, v = cycle[0][:2]
                edge_data = G.edges[u, v]
                G.remove_edge(u, v)
                removed_edges.append((u, v, edge_data))
        except nx.NetworkXNoCycle:
            break
    return removed_edges


def add_removed_edges(G, removed_edges):
    for u, v, edge_data in removed_edges:
        G.add_edge(u, v, **edge_data)


def visualize_flow(G, tech_mode_df, kusts_df, wells_df, first_name, second_name, date):
    """
    Оптимизированная версия без визуализации, только расчет DataFrame
    """
    data_dict = {
        "date": [],
        "id": [],
        "Qv": [],
        "Qzh": [],
        "Qn": [],
        "Qgas": [],
        "water_cut": [],
        "kvch": [],
        "CO2": [],
        "Min": [],
        "pH": [],
        "ing_sum": [],
        "viscosity_liquid_work": [],
        "density_liquid_work": [],
        "density_oil": [],
        "gas_content_true": [],
        "p_start": [],
        "v_liquid_true": [],
        "reynolds": [],
        "fruda": [],
        "temperature": [],
    }
    
    # Pre-calculate edge data once
    edges_data = []
    for edge in G.edges(data=True):
        source, target, data = edge
        if data.get("Qzh", 0) != 0:
            edges_data.append({
                "date": date,
                "id": data.get("id", "Unknown"),
                "Qv": data.get("Qv", 0),
                "Qzh": data.get("Qzh", 0),
                "Qn": data.get("Qn", 0),
                "Qgas": data.get("Qgas", 0),
                "water_cut": data.get("water_cut", 0),
                "kvch": data.get("kvch", 0),
                "CO2": data.get("CO2", 0),
                "Min": data.get("Min", 0),
                "pH": data.get("pH", 0),
                "ing_sum": data.get("ing_sum", 0),
                "viscosity_liquid_work": data.get("liq_viscosity", 0),
                "density_liquid_work": float(data.get("density_liquid_work", 0)),
                "density_oil": float(data.get("density_oil", 0)),
                "gas_content_true": float(data.get("gas_content_true", 0)),
                "p_start": float(data.get("p_start", 0)),
                "v_liquid_true": float(data.get("v_liquid_true", 0)),
                "reynolds": float(data.get("reynolds", 0)),
                "fruda": float(data.get("fruda", 0)),
                "temperature": float(data.get("temperature", 0))
            })
    
    # Create DataFrame directly from list of dictionaries
    return pd.DataFrame(edges_data)


def process_item(key, value, date_range):
    """
    Оптимизированная обработка элементов
    
    Args:
        key: название месторождения
        value: название площадки
        date_range: диапазон дат для обработки
    """
    first_name = key
    second_name = value
    print(f"Processing {first_name}___{second_name}")

    # Используем список для накопления результатов вместо DataFrame
    object_df = []
    
    # Предварительно получаем глобальные переменные для избежания лишних обращений
    global all_edges_df, all_tr_df, all_tech_mode_df, all_loc_info_df
    global all_kvch_df, all_fhs_plasts_df, all_ingib_df

    pipe_info = all_tech_mode_df[['simple_section_id', 'd_mm', 'length']].copy()
    pipe_info['simple_section_id'] = pipe_info['simple_section_id'].astype(float).astype(str)

    try:
        for date in tqdm(date_range, desc=f"{first_name}__{second_name}", unit="date"):
            # Получаем предобработанные данные
            try:
                edges_df, kusts_df, wells_df, tech_mode_df, fhs_plasts_df, ingib_df = preprocess_data(
                    all_edges_df, all_tr_df, all_tech_mode_df, all_loc_info_df,
                    all_kvch_df, all_fhs_plasts_df, all_ingib_df,
                    first_name, second_name, date
                )
                print(f"date: {date}")
            except Exception as e:
                print(f"Error in preprocess_data for {first_name}__{second_name} at {date}: {e}")
                continue

            # Строим и обрабатываем граф
            try:
                G = build_graph(edges_df)
                subgraph = get_active_subgraph(G, edges_df, kusts_df)
                removed_edges = find_and_remove_cycles(subgraph, tech_mode_df)
                calculate_flow(subgraph, tech_mode_df, kusts_df, wells_df, fhs_plasts_df, ingib_df, pipe_info)
                add_removed_edges(subgraph, removed_edges)
                
                res_df = visualize_flow(
                    subgraph, tech_mode_df, kusts_df, wells_df,
                    first_name, second_name, date
                )
                
                object_df.append(res_df)
                
                # Очищаем память после обработки каждой даты
                del G, subgraph, removed_edges, res_df
                gc.collect()
                
            except Exception as e:
                print(f"Error processing graph for {first_name}__{second_name} at {date}: {e}")
                continue

    except Exception as e:
        print(f"Error in main loop for {first_name}__{second_name}: {e}")
        raise

    # Объединяем все результаты
    if not object_df:
        print(f"No results for {first_name}__{second_name}")
        return pd.DataFrame()  # Возвращаем пустой DataFrame если нет результатов
        
    res = pd.concat(object_df, ignore_index=True)
    res.to_csv(f"/home/gsurinov@corp.nedra.digital/predict_ml_be/src/tech_mode/smash/{first_name}__{second_name}.csv")    
    return res


    # def main():
def create_flow(ing):
    """
    Оптимизированная версия create_flow с улучшенным управлением памятью и progress bar
    """
    
    data_dir = "."
    edges_file = os.path.join(data_dir, settings.tech_mode_calc.edges_file)
    tr_file = os.path.join(data_dir, settings.tech_mode_calc.tr_file)
    tech_mode_file = settings.tech_mode_calc.tech_mode_file
    loc_info_file = os.path.join(data_dir, settings.tech_mode_calc.loc_info_file)
    kvch_file = os.path.join(data_dir, settings.tech_mode_calc.kvch_file)
    fhs_plasts_file = os.path.join(data_dir, settings.tech_mode_calc.fhs_plasts_file)

    # Читаем только нужные колонки из edges_file для определения схем
    edges_schema_df = pd.read_excel(
        edges_file, 
        sheet_name="Лист2",
        usecols=[
            "Местонахождение_Месторождение",
            "Местонахождение_Принадлежность_к_объекту_(площадка)"
        ]
    )
    
    schemas = (
        edges_schema_df.groupby("Местонахождение_Месторождение")[
            "Местонахождение_Принадлежность_к_объекту_(площадка)"
        ]
        .unique()
        .to_dict()
    )
    
    # Очищаем память от временного DataFrame
    del edges_schema_df
    gc.collect()

    global all_edges_df, all_tr_df, all_tech_mode_df, all_loc_info_df, all_kvch_df, all_fhs_plasts_df
    (
        all_edges_df,
        all_tr_df,
        all_tech_mode_df,
        all_loc_info_df,
        all_kvch_df,
        all_fhs_plasts_df,
    ) = load_data(
        edges_file,
        tr_file,
        tech_mode_file,
        loc_info_file,
        kvch_file,
        fhs_plasts_file,
    )

    global all_ingib_df
    all_ingib_df = ing

    # all_tech_mode_df = all_tech_mode_df.rename(columns={"date": "calc_date", "id": "simple_section_id"})

    print(all_tech_mode_df)

    # start_date = all_tech_mode_df["calc_date"].min()
    start_date = pd.Period("2014-01-01", "D")
    end_date = all_tech_mode_df["calc_date"].max()
    
    #debug dates
    # start_date = pd.Period("2023-01-03", "D")
    # end_date = pd.Period("2014-01-01", "D")

    date_range = pd.period_range(
        start=start_date,
        end=end_date,
        freq="D"
    )
    

    results = []
    
    # Подготавливаем список всех задач
    tasks = []
    for key in schemas.keys():
        if key not in ["Красноленинское"]:
            for value in schemas[key]:
                tasks.append((key, value))
    
    # Создаем прогресс бар для всех месторождений
    fields = [key for key, _ in tasks]
    unique_fields = list(dict.fromkeys(fields))  # сохраняем порядок
    
    with tqdm(total=len(unique_fields), desc="Processing fields") as pbar:
        # Сохраняем эффективный ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            # Отслеживаем завершенные месторождения
            completed_fields = set()
            
            # Запускаем задачи и получаем Future objects
            future_to_field = {
                executor.submit(process_item, key, value, date_range): key 
                for key, value in tasks
                # if value in ['ДНС-2']
            }
            
            # Обрабатываем результаты по мере их поступления
            for future in concurrent.futures.as_completed(future_to_field):
                field = future_to_field[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Если все задачи для месторождения завершены, обновляем прогресс
                    if field not in completed_fields:
                        remaining_tasks = sum(1 for f, k in future_to_field.items() 
                                           if k == field and not f.done())
                        if remaining_tasks == 0:
                            completed_fields.add(field)
                            pbar.update(1)
                            
                except Exception as e:
                    print(f"Error processing {field}: {e}")

    # Оптимизированное объединение результатов
    all_res = pd.concat(results, ignore_index=True)
    # Sort by column: 'date' (descending)
    all_res = all_res.sort_values(['date', 'id'], ascending=[False, False])
    # id to int
    all_res['id'] = all_res['id'].astype(float).astype(int)

    return all_res

if __name__ == "__main__":
    start_time = time.perf_counter()    
    # load ing
    ing = pd.read_csv('/home/gsurinov@corp.nedra.digital/predict_ml_be/data/raw/ing_res.csv')
    res = create_flow(ing)
    res.to_csv('flow_res_temp.csv', index=False)
    total_time = time.perf_counter() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
