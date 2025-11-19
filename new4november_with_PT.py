import concurrent.futures
import os
import time
import warnings
from datetime import datetime
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network

# Отключение всех предупреждений
warnings.filterwarnings("ignore")

def convert_files_to_parquet(combined_data_file, tech_mode_file, kust_tube_file):
    """Конвертация файлов в parquet формат"""
    
    # Конвертация combined_data
    combined_df = pd.read_csv(combined_data_file,index_col=False)
    mixed_type_columns = combined_df.select_dtypes(include=["object"]).columns
    for col in mixed_type_columns:
        combined_df[col] = combined_df[col].astype(str)
        combined_df[col] = combined_df[col].replace("nan", "")
    combined_df.to_parquet(combined_data_file.replace(".csv", ".parquet"))

    # Конвертация tech_mode
    tech_df = pd.read_excel(tech_mode_file)
    mixed_type_columns = tech_df.select_dtypes(include=["object"]).columns
    for col in mixed_type_columns:
        tech_df[col] = tech_df[col].astype(str)
        tech_df[col] = tech_df[col].replace("nan", "")
    tech_df.to_parquet(tech_mode_file.replace(".xlsx", ".parquet"))

    # Конвертация связки куст-труба
    kust_tube_df = pd.read_excel(kust_tube_file)
    mixed_type_columns = kust_tube_df.select_dtypes(include=["object"]).columns
    for col in mixed_type_columns:
        kust_tube_df[col] = kust_tube_df[col].astype(str)
        kust_tube_df[col] = kust_tube_df[col].replace("nan", "")
    kust_tube_df.to_parquet(kust_tube_file.replace(".xlsx", ".parquet"))

    print("Все файлы успешно сконвертированы в формат parquet")

@lru_cache(maxsize=1)
def load_data(combined_data_file, tech_mode_file, kust_tube_file):
    """Загрузка данных"""
    if not os.path.exists(combined_data_file.replace(".csv", ".parquet")):
        convert_files_to_parquet(combined_data_file, tech_mode_file, kust_tube_file)
    
    combined_df = pd.read_parquet(combined_data_file.replace(".csv", ".parquet"))
    tech_mode_df = pd.read_parquet(tech_mode_file.replace(".xlsx", ".parquet"))
    kust_tube_df = pd.read_parquet(kust_tube_file.replace(".xlsx", ".parquet"))

    return combined_df, tech_mode_df, kust_tube_df

def load_tube_angles(angles_file):
    """Загрузка данных об углах наклона труб"""
    try:
        angles_df = pd.read_csv(angles_file)
        return angles_df
    except Exception as e:
        print(f"Ошибка загрузки файла углов: {e}")
        return pd.DataFrame()

def calculate_tube_angles(angles_df, tube_ids):
    """Расчет среднего угла наклона для первых 20 метров каждой трубы"""
    tube_angles = {}
    
    for tube_id in tube_ids:
        tube_data = angles_df[angles_df['id'] == tube_id]
        if not tube_data.empty:
            # Фильтруем данные для первых 20 метров
            first_20m = tube_data[tube_data['distance'] <= 20]
            if not first_20m.empty:
                avg_angle = first_20m['Z'].mean()
                tube_angles[tube_id] = avg_angle
            else:
                tube_angles[tube_id] = 0  # Горизонтально по умолчанию
        else:
            tube_angles[tube_id] = 0  # Горизонтально по умолчанию
    
    return tube_angles

def preprocess_data(combined_df, tech_mode_df, kust_tube_df, target_date, angles_df=None):
    """Предобработка данных для конкретной даты"""
    
    # Фильтрация combined_data по дате
    combined_df['Дата замера'] = pd.to_datetime(combined_df['Дата замера'])
    filtered_combined = combined_df[combined_df['Дата замера'].dt.to_period("D") == target_date].copy()
    print(f"DEBUG preprocess_data: Дата {target_date}")
    print(f"DEBUG: Отфильтровано строк: {len(filtered_combined)}")
    print(f"DEBUG: Данные filtered_combined:")
    print(filtered_combined[['Номер скважины', 'Куст', 'Дебит жидкости, м3/сут, ТР']].head())
    
    # ИСПРАВЛЕНИЕ: Проверяем уникальность кустов перед агрегацией
    unique_kusts_before = filtered_combined['Куст'].nunique()
    print(f"DEBUG: Уникальных кустов в отфильтрованных данных: {unique_kusts_before}")
    
    # Агрегация по кустам
    kusts_df = filtered_combined.groupby('Куст').agg({
        'Дебит жидкости, м3/сут, ТР': 'sum',
        'Дебит нефти, т/сут, ТР': 'sum',
        'Обводненность, %, ТР': 'mean',
        'Вяз-ть жидкости, ТР': 'mean',
        'КВЧ, ТР': 'mean',
        'Газ. фактор, м3/т, ТР': 'mean',
        'Давление в линии, атм, ТР': 'mean'
    }).reset_index()
    
    # ИСПРАВЛЕНИЕ: Удаляем дубликаты кустов (если они появились из-за связки куст-труба)
    kusts_df = kusts_df.drop_duplicates(subset=['Куст'])
    print(f"DEBUG: После удаления дубликатов: {len(kusts_df)} кустов (было: {unique_kusts_before})")
    
    # Данные по скважинам
    wells_df = filtered_combined[[
        'Номер скважины', 'Куст', 'Дебит жидкости, м3/сут, ТР', 
        'Дебит нефти, т/сут, ТР', 'Обводненность, %, ТР', 'Вяз-ть жидкости, ТР',
        'КВЧ, ТР', 'Газ. фактор, м3/т, ТР', 'Давление в линии, атм, ТР'
    ]].copy()
    wells_df['Номер скважины'] = wells_df['Номер скважины'].astype(str).str.lower()
    
    # Построение графа из tech_mode
    edges_df = tech_mode_df[['Узел начала', 'Узел конца', 'ID простого участка']].copy()
    edges_df['Узел начала'] = edges_df['Узел начала'].astype(str).str.lower().str.replace(" ", "")
    edges_df['Узел конца'] = edges_df['Узел конца'].astype(str).str.lower().str.replace(" ", "")
    
    # Добавляем связи скважин с кустами
    wells_df['Куст'] = wells_df['Куст'].astype(str).str.lower().str.replace(" ", "")
    kusts_df['Куст'] = kusts_df['Куст'].astype(str).str.lower().str.replace(" ", "")
    
    # Создаем связи скважина -> куст
    well_to_kust_edges = pd.DataFrame({
        'Узел начала': wells_df['Номер скважины'] + '_well',
        'Узел конца': wells_df['Куст'],
        'ID простого участка': 'well_' + wells_df['Номер скважины']
    })
    edges_df = pd.concat([edges_df, well_to_kust_edges], ignore_index=True)
    
    # ДОБАВЛЕНО: Создаем связи куст -> труба с улучшенной отладкой
    kust_to_tube_edges = []
    processed_kust_tube_pairs = set()  # Для отслеживания уникальных пар куст-труба
    
    for _, row in kust_tube_df.iterrows():
        kust = str(row['куст']).lower().replace(" ", "")
        tube_id = str(row['id простого участка'])
        
        # Проверяем, не обрабатывали ли мы уже эту пару куст-труба
        pair_key = (kust, tube_id)
        if pair_key in processed_kust_tube_pairs:
            continue
        processed_kust_tube_pairs.add(pair_key)
        
        # Находим узел трубы в tech_mode данных
        tube_data = tech_mode_df[tech_mode_df['ID простого участка'] == tube_id]
        if not tube_data.empty:
            tube_node = str(tube_data['Узел начала'].iloc[0]).lower().replace(" ", "")
            edge_id = f"kust_{kust}_to_{tube_id}"
            
            kust_to_tube_edges.append({
                'Узел начала': kust,
                'Узел конца': tube_node,
                'ID простого участка': edge_id
            })
            print(f"DEBUG: Создана связь куст-труба: {kust} -> {tube_node} (id: {edge_id})")
        else:
            print(f"DEBUG: ВНИМАНИЕ: Не найдена труба с ID={tube_id} в tech_mode данных для куста {kust}")
    
    if kust_to_tube_edges:
        edges_df = pd.concat([edges_df, pd.DataFrame(kust_to_tube_edges)], ignore_index=True)
        print(f"DEBUG: Добавлено {len(kust_to_tube_edges)} связей куст-труба для {len(processed_kust_tube_pairs)} уникальных пар")
    
    # Подготовка данных о геометрии труб
    tube_geometry = {}
    tube_pvt = {}
    tube_initial_conditions = {}
    
    for _, row in tech_mode_df.iterrows():
        tube_id = str(row['ID простого участка'])
        
        # Геометрия
        tube_geometry[tube_id] = {
            'length': row.get('L,м', 100),
            'diameter': row.get('D,мм', 100) / 1000,  # переводим в метры
            'roughness': 0.10,  # константа
            'ambient_temp': 5,  # константа
        }
        
        # PVT свойства
        tube_pvt[tube_id] = {
            'oil_density': row.get('Нефти, кг/м3', 800),
            'liquid_density': row.get('Жидкости, кг/м3', 900),
            'gas_density': row.get('Газа, кг/м3', 1.2),
            'surface_tension': 0.03,  # константа
        }
        
        # Начальные условия
        tube_initial_conditions[tube_id] = {
            'pressure': row.get('В начале участка фактическое, атм', 10) * 0.101325,  # переводим в МПа
            'temperature': row.get('Средняя температура, C', 60),
        }
    
    # Расчет углов наклона
    tube_angles = {}
    if angles_df is not None:
        tube_ids = list(tube_geometry.keys())
        tube_angles = calculate_tube_angles(angles_df, tube_ids)
    
    # Подготовка tech_mode данных для распределения потоков
    tech_mode_df['ID простого участка'] = tech_mode_df['ID простого участка'].astype(str)
    tech_mode_filtered = tech_mode_df.copy()
    
    # Финальная проверка
    print(f"DEBUG: Итоговые размеры:")
    print(f"  edges_df: {len(edges_df)} строк")
    print(f"  kusts_df: {len(kusts_df)} кустов")
    print(f"  wells_df: {len(wells_df)} скважин")
    print(f"  tube_geometry: {len(tube_geometry)} труб")
    
    return edges_df, kusts_df, wells_df, tech_mode_filtered, kust_tube_df, tube_geometry, tube_pvt, tube_initial_conditions, tube_angles


def build_graph(edges_df):
    """Построение графа"""
    G = nx.DiGraph()
    print(f"DEBUG: Всего ребер для построения графа: {len(edges_df)}")
    
    for _, row in edges_df.iterrows():
        start_node = str(row['Узел начала']).strip()
        end_node = str(row['Узел конца']).strip()
        edge_id = row['ID простого участка']
        
        if start_node != end_node:
            G.add_edge(start_node, end_node, id=edge_id)
    
    print(f"DEBUG: Построен граф с {G.number_of_nodes()} узлами и {G.number_of_edges()} ребрами")
    return G

def find_and_remove_cycles(G, tech_mode_df):
    """Поиск и удаление циклов в графе"""
    removed_edges = []
    while True:
        try:
            cycle = nx.find_cycle(G, orientation="original")
            for edge in cycle:
                u, v = edge[:2]
                edge_data = G.edges[u, v]
                edge_id = edge_data.get("id")
                if edge_id in tech_mode_df['ID простого участка'].values:
                    # Проверяем расход в трубе
                    flow_data = tech_mode_df[tech_mode_df['ID простого участка'] == edge_id]
                    if not flow_data.empty and flow_data['Жидкости, м3/сут'].iloc[0] == 0:
                        G.remove_edge(u, v)
                        removed_edges.append((u, v, edge_data))
                        break
            else:
                # Если все трубы с ненулевым расходом, удаляем первое ребро
                u, v = cycle[0][:2]
                edge_data = G.edges[u, v]
                G.remove_edge(u, v)
                removed_edges.append((u, v, edge_data))
        except nx.NetworkXNoCycle:
            break
    return removed_edges

def add_removed_edges(G, removed_edges):
    """Восстановление удаленных ребер после расчетов"""
    for u, v, edge_data in removed_edges:
        G.add_edge(u, v, **edge_data)

def calculate_pvt_properties_for_tube(liq_rate, oil_rate, water_cut, gas_factor, viscosity, pvt_data):
    """
    Расчет PVT свойств для трубы
    """
    # Расчет дебитов компонентов
    water_rate_m3_day = liq_rate * water_cut / 100  # м3/сут воды
    oil_rate_m3_day = oil_rate / pvt_data['oil_density']  # м3/сут нефти
    
    # Общий дебит жидкости
    total_liquid_rate_m3_day = water_rate_m3_day + oil_rate_m3_day
    
    # Дебит газа
    gas_rate_m3_day = oil_rate * gas_factor  # м3/сут газа
    
    # Плотность смеси жидкости
    liquid_density_mix = (pvt_data['oil_density'] * oil_rate_m3_day + 
                         pvt_data['liquid_density'] * water_rate_m3_day) / total_liquid_rate_m3_day
    
    # Параметры для Beggs-Brill
    pvt_params = {
        'ql_rc_m3day': total_liquid_rate_m3_day,
        'qg_rc_m3day': gas_rate_m3_day,
        'rho_lrc_kgm3': liquid_density_mix,
        'rho_grc_kgm3': pvt_data['gas_density'],
        'sigma_l_nm': pvt_data['surface_tension'],
        'mul_rc_cp': viscosity,
        'mug_rc_cp': 0.02,  # Вязкость газа (константа)
    }
    
    return pvt_params

def calculate_pressure_drop(G, tube_geometry, tube_pvt, tube_angles, tube_initial_conditions):
    """
    Расчет перепада давления в трубах по Beggs-Brill
    """
    from _beggsbrill import BeggsBrill
    
    for edge in G.edges(data=True):
        source, target, data = edge
        edge_id = data.get('id', 'Unknown')
        
        # Проверяем, что это труба (не связь скважина-куст)
        if (not source.endswith('_well') and 
            str(source).lower() not in ['k301', '301'] and
            edge_id in tube_geometry):
            
            if 'liq_rate' in data and data['liq_rate'] > 0:
                try:
                    # Получаем параметры потока
                    liq_rate = data['liq_rate']
                    oil_rate = data.get('oil_rate', 0)
                    water_cut = data.get('water_cut', 0)
                    gas_factor = data.get('kvch', 0)  # Газовый фактор
                    viscosity = data.get('viscosity', 1.0)
                    
                    # Получаем геометрию и начальные условия
                    geometry = tube_geometry[edge_id]
                    pvt_data = tube_pvt[edge_id]
                    initial_conditions = tube_initial_conditions[edge_id]
                    angle = tube_angles.get(edge_id, 0)
                    
                    length = geometry['length']
                    diameter = geometry['diameter']
                    roughness = geometry['roughness']
                    
                    pressure_in = initial_conditions['pressure']  # МПа
                    temperature_in = initial_conditions['temperature']
                    
                    # Расчет PVT свойств
                    pvt_params = calculate_pvt_properties_for_tube(
                        liq_rate, oil_rate, water_cut, gas_factor, viscosity, pvt_data
                    )
                    
                    # Расчет перепада давления по Beggs-Brill
                    bb = BeggsBrill(diameter)
                    bb.calc_params(
                        theta_deg=angle,
                        ql_rc_m3day=pvt_params['ql_rc_m3day'],
                        qg_rc_m3day=pvt_params['qg_rc_m3day'],
                        rho_lrc_kgm3=pvt_params['rho_lrc_kgm3'],
                        rho_grc_kgm3=pvt_params['rho_grc_kgm3'],
                        sigma_l_nm=pvt_params['sigma_l_nm'],
                        c_calibr_grav=1.0
                    )
                    
                    # Расчет градиентов
                    dp_dl_grav = bb.calc_grav(angle, 1.0)
                    dp_dl_fric = bb.calc_fric(
                        eps_m=roughness,
                        ql_rc_m3day=pvt_params['ql_rc_m3day'],
                        mul_rc_cp=pvt_params['mul_rc_cp'],
                        mug_rc_cp=pvt_params['mug_rc_cp'],
                        c_calibr_fric=1.0
                    )
                    
                    # Общий перепад давления (Па/м)
                    total_dp_dl = dp_dl_grav + dp_dl_fric
                    
                    # Перепад давления на всей длине трубы
                    pressure_drop_pa = total_dp_dl * length  # Па
                    pressure_drop_mpa = pressure_drop_pa / 1e6  # МПа
                    
                    # Давление на выходе
                    pressure_outlet_mpa = pressure_in - pressure_drop_mpa
                    
                    # Сохраняем результаты
                    data['pressure_inlet_mpa'] = pressure_in
                    data['pressure_outlet_mpa'] = pressure_outlet_mpa
                    data['pressure_drop_mpa'] = pressure_drop_mpa
                    data['dp_dl_grav_pa_m'] = dp_dl_grav
                    data['dp_dl_fric_pa_m'] = dp_dl_fric
                    data['temperature_inlet'] = temperature_in
                    
                    print(f"Труба {edge_id}: Pвх = {pressure_in:.2f} МПа, Pвых = {pressure_outlet_mpa:.2f} МПа, ΔP = {pressure_drop_mpa:.4f} МПа")
                    
                except Exception as e:
                    print(f"Ошибка расчета давления для трубы {edge_id}: {e}")
                    # Устанавливаем значения по умолчанию при ошибке
                    data['pressure_inlet_mpa'] = 0
                    data['pressure_outlet_mpa'] = 0
                    data['pressure_drop_mpa'] = 0
                    data['dp_dl_grav_pa_m'] = 0
                    data['dp_dl_fric_pa_m'] = 0
                    data['temperature_inlet'] = 0

def calculate_temperature_drop(G, tube_geometry, tube_initial_conditions):
    """
    Расчет изменения температуры в трубах
    """
    from rough_temperature_model import RoughTemperatureModel
    
    for edge in G.edges(data=True):
        source, target, data = edge
        edge_id = data.get('id', 'Unknown')
        
        if (not source.endswith('_well') and 
            'liq_rate' in data and data['liq_rate'] > 0 and
            edge_id in tube_geometry):
            
            try:
                geometry = tube_geometry[edge_id]
                initial_conditions = tube_initial_conditions[edge_id]
                
                length = geometry['length']
                diameter = geometry['diameter']
                roughness = geometry['roughness']
                ambient_temp = geometry['ambient_temp']
                
                liq_rate = data['liq_rate']
                temperature_in = data.get('temperature_inlet', initial_conditions['temperature'])
                
                # Упрощенный расчет теплового потока
                heat_capacity = 4200  # Дж/кг/К, удельная теплоемкость
                density = 1000  # кг/м3
                
                mass_flow = liq_rate * density / 86400  # кг/с
                q_heat_in_by_degree = mass_flow * heat_capacity  # Вт/К
                
                # Расчет температуры
                temp_model = RoughTemperatureModel(
                    length=length,
                    tid=diameter,
                    tir=roughness,
                    q_heat_in_by_degree=q_heat_in_by_degree,
                    surrounding_temperature=ambient_temp,
                    htc=15,  # Коэффициент теплопередачи
                    t_bound=temperature_in,
                    direction=1
                )
                
                temp_profile = temp_model.run()
                outlet_temp = temp_profile(length)
                
                data['temperature_outlet'] = outlet_temp
                data['temperature_drop'] = temperature_in - outlet_temp
                
                print(f"Труба {edge_id}: Tвх = {temperature_in:.1f}°C, Tвых = {outlet_temp:.1f}°C, ΔT = {data['temperature_drop']:.2f}°C")
                
            except Exception as e:
                print(f"Ошибка расчета температуры для трубы {edge_id}: {e}")
                data['temperature_outlet'] = data.get('temperature_inlet', 0)
                data['temperature_drop'] = 0

def calculate_flow(G, kusts_df, wells_df, kust_tube_df, tech_mode_df, tube_geometry, tube_pvt, tube_initial_conditions, tube_angles):
    """Перерасчет параметров по графу с связкой куст-труба"""
    
    print("=== DEBUG: НАЧАЛО calculate_flow ===")
    
    # Создаем словари
    kust_dict = {}
    for _, row in kusts_df.iterrows():
        kust_name = str(row['Куст']).lower().replace(" ", "")
        if kust_name not in kust_dict:
            kust_dict[kust_name] = {
                'Дебит жидкости, м3/сут, ТР': row['Дебит жидкости, м3/сут, ТР'],
                'Дебит нефти, т/сут, ТР': row['Дебит нефти, т/сут, ТР'],
                'Обводненность, %, ТР': row['Обводненность, %, ТР'],
                'Вяз-ть жидкости, ТР': row['Вяз-ть жидкости, ТР'],
                'КВЧ, ТР': row['КВЧ, ТР'],
                'Газ. фактор, м3/т, ТР': row['Газ. фактор, м3/т, ТР'],
                'Давление в линии, атм, ТР': row['Давление в линии, атм, ТР']
            }

    wells_dict = {}
    for _, row in wells_df.iterrows():
        well_id = str(row['Номер скважины']).lower()
        wells_dict[well_id] = {
            'Дебит жидкости, м3/сут, ТР': row['Дебит жидкости, м3/сут, ТР'],
            'Дебит нефти, т/сут, ТР': row['Дебит нефти, т/сут, ТР'],
            'Обводненность, %, ТР': row['Обводненность, %, ТР'],
            'Вяз-ть жидкости, ТР': row['Вяз-ть жидкости, ТР'],
            'КВЧ, ТР': row['КВЧ, ТР'],
            'Газ. фактор, м3/т, ТР': row['Газ. фактор, м3/т, ТР'],
            'Давление в линии, атм, ТР': row['Давление в линии, атм, ТР']
        }
    
    # Создаем словарь для связи куст -> список труб
    kust_to_tubes = {}
    for _, row in kust_tube_df.iterrows():
        kust = str(row['куст']).lower().replace(" ", "")
        tube_id = str(row['id простого участка'])
        if kust not in kust_to_tubes:
            kust_to_tubes[kust] = []
        kust_to_tubes[kust].append(tube_id)
    
    print(f"DEBUG: Кусты в данных: {list(kust_dict.keys())}")
    print(f"DEBUG: Связка куст-труба: {kust_to_tubes}")
    
    # ШАГ 1: Инициализация параметров для кустов
    print("=== ШАГ 1: Инициализация кустов ===")
    for node in G.nodes():
        node_str = str(node).lower().replace(" ", "")
        
        if node_str in kust_dict:
            print(f"DEBUG: Найден куст в графе: {node} (нормализовано: {node_str})")
            kust_data = kust_dict[node_str]
            
            # Находим все скважины, подключенные к этому кусту
            well_contributions = []
            total_liq = 0
            
            for pred in G.predecessors(node):
                if pred.endswith('_well'):
                    well_id = pred.replace('_well', '')
                    if well_id in wells_dict:
                        well_data = wells_dict[well_id]
                        well_liq = well_data['Дебит жидкости, м3/сут, ТР']
                        well_contributions.append((well_data, well_liq))
                        total_liq += well_liq
                        
                        # Передаем данные от скважины к кусту по ребру
                        G.edges[pred, node].update({
                            'liq_rate': well_data['Дебит жидкости, м3/сут, ТР'],
                            'oil_rate': well_data['Дебит нефти, т/сут, ТР'],
                            'water_cut': well_data['Обводненность, %, ТР'],
                            'viscosity': well_data['Вяз-ть жидкости, ТР'],
                            'kvch': well_data['КВЧ, ТР'],
                            'pressure': well_data['Давление в линии, атм, ТР']
                        })
                        print(f"  СКВАЖИНА -> КУСТ: {pred} -> {node}, liq_rate = {well_liq:.2f}")
            
            # Расчет параметров для куста
            if total_liq > 0:
                water_cut = sum(w[0]['Обводненность, %, ТР'] * w[1] for w in well_contributions) / total_liq
                viscosity = sum(w[0]['Вяз-ть жидкости, ТР'] * w[1] for w in well_contributions) / total_liq
                kvch = sum(w[0]['КВЧ, ТР'] * w[1] for w in well_contributions) / total_liq
                pressure = sum(w[0]['Давление в линии, атм, ТР'] * w[1] for w in well_contributions) / total_liq
            else:
                water_cut = kust_data['Обводненность, %, ТР']
                viscosity = kust_data['Вяз-ть жидкости, ТР']
                kvch = kust_data['КВЧ, ТР']
                pressure = kust_data['Давление в линии, атм, ТР']
            
            # Сохраняем параметры в узле куста
            node_params = {
                'liq_rate': kust_data['Дебит жидкости, м3/сут, ТР'],
                'oil_rate': kust_data['Дебит нефти, т/сут, ТР'],
                'water_cut': water_cut,
                'viscosity': viscosity,
                'kvch': kvch,
                'pressure': pressure,
                'well_count': len(well_contributions)
            }
            nx.set_node_attributes(G, {node: node_params})
            print(f"  КУСТ ИНИЦИАЛИЗИРОВАН: {node}, liq_rate = {node_params['liq_rate']:.2f}, скважин = {len(well_contributions)}")

    # ШАГ 2: Передача данных от кустов к трубам
    print("=== ШАГ 2: Передача от кустов к трубам ===")
    for node in G.nodes():
        node_str = str(node).lower().replace(" ", "")
        
        if node_str in kust_dict and 'liq_rate' in G.nodes[node]:
            kust_params = G.nodes[node]
            print(f"DEBUG: Обрабатываем куст: {node} с данными")
            
            # Находим все исходящие ребра от этого куста
            successors = list(G.successors(node))
            print(f"  Исходящие связи куста {node}: {successors}")
            
            for successor in successors:
                edge_data = G.edges[node, successor]
                edge_id = edge_data.get('id', 'Unknown')
                
                print(f"  Проверяем связь: {node} -> {successor} (id: {edge_id})")
                
                # Проверяем, что это связь куст-труба
                is_kust_to_tube = False
                if node_str in kust_to_tubes:
                    if edge_id in kust_to_tubes[node_str]:
                        is_kust_to_tube = True
                        print(f"    Найдена связь куст-труба по kust_to_tubes")
                
                # ИЛИ проверяем по формату ID
                if not is_kust_to_tube and edge_id.startswith(f"kust_{node_str}_to_"):
                    is_kust_to_tube = True
                    print(f"    Найдена связь куст-труба по формату ID")
                
                if is_kust_to_tube:
                    # Передаем данные куста в ребро
                    G.edges[node, successor].update({
                        'liq_rate': kust_params['liq_rate'],
                        'oil_rate': kust_params['oil_rate'],
                        'water_cut': kust_params['water_cut'],
                        'viscosity': kust_params['viscosity'],
                        'kvch': kust_params['kvch'],
                        'pressure': kust_params['pressure']
                    })
                    
                    print(f"    ПЕРЕДАЧА ДАННЫХ: {node} -> {successor}, liq_rate = {kust_params['liq_rate']:.2f}")
                    
                    # Также передаем данные в узел-получатель (трубу)
                    if 'liq_rate' not in G.nodes[successor]:
                        G.nodes[successor].update({
                            'liq_rate': kust_params['liq_rate'],
                            'oil_rate': kust_params['oil_rate'],
                            'water_cut': kust_params['water_cut'],
                            'viscosity': kust_params['viscosity'],
                            'kvch': kust_params['kvch'],
                            'pressure': kust_params['pressure']
                        })
                        print(f"    ИНИЦИАЛИЗАЦИЯ ТРУБЫ: {successor}, liq_rate = {kust_params['liq_rate']:.2f}")
                else:
                    print(f"    Связь НЕ опознана как куст-труба")

    # ШАГ 3: Распространение данных по цепочкам
    print("=== ШАГ 3: Распространение по цепочкам ===")
    
    def propagate_from_node(start_node, visited=None):
        if visited is None:
            visited = set()
        
        if start_node in visited:
            return
        visited.add(start_node)
        
        if 'liq_rate' in G.nodes[start_node]:
            node_params = G.nodes[start_node]
            
            for successor in G.successors(start_node):
                # Пропускаем связи скважина->куст
                if start_node.endswith('_well') and str(successor).lower().replace(" ", "") in kust_dict:
                    continue
                    
                # Передаем данные по ребру
                G.edges[start_node, successor].update({
                    'liq_rate': node_params['liq_rate'],
                    'oil_rate': node_params['oil_rate'],
                    'water_cut': node_params['water_cut'],
                    'viscosity': node_params['viscosity'],
                    'kvch': node_params['kvch'],
                    'pressure': node_params['pressure']
                })
                
                print(f"  ЦЕПОЧКА: {start_node} -> {successor}, liq_rate = {node_params['liq_rate']:.2f}")
                
                # Также передаем данные в узел-получатель
                if 'liq_rate' not in G.nodes[successor] and str(successor).lower().replace(" ", "") not in kust_dict:
                    G.nodes[successor].update({
                        'liq_rate': node_params['liq_rate'],
                        'oil_rate': node_params['oil_rate'],
                        'water_cut': node_params['water_cut'],
                        'viscosity': node_params['viscosity'],
                        'kvch': node_params['kvch'],
                        'pressure': node_params['pressure']
                    })
                
                # Рекурсивно обрабатываем следующий узел
                propagate_from_node(successor, visited)
    
    visited_nodes = set()
    for node in G.nodes():
        if 'liq_rate' in G.nodes[node] and node not in visited_nodes:
            propagate_from_node(node, visited_nodes)

    # ШАГ 4: Расчет давления и температуры
    print("=== ШАГ 4: Расчет давления и температуры ===")
    calculate_pressure_drop(G, tube_geometry, tube_pvt, tube_angles, tube_initial_conditions)
    calculate_temperature_drop(G, tube_geometry, tube_initial_conditions)

    # Финальная проверка
    edges_with_data = 0
    for edge in G.edges(data=True):
        if 'liq_rate' in edge[2]:
            edges_with_data += 1
    
    print(f"DEBUG: Всего ребер с данными: {edges_with_data} из {G.number_of_edges()}")
    print("=== DEBUG: КОНЕЦ calculate_flow ===")

def visualize_flow(G, kusts_df, wells_df, date):
    """Визуализация графа с разными типами узлов"""
    nt = Network("1080px", "1720px", directed=True)
    
    nt.set_options("""
    var options = {
      "nodes": {
        "font": {"size": 12},
        "shape": "dot"
      },
      "edges": {
        "smooth": {"enabled": true, "type": "continuous"},
        "font": {"size": 10},
        "arrows": {"to": {"enabled": true, "scaleFactor": 1.2}}
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        }
      }
    }
    """)
    
    # Добавляем узлы с разными цветами и формами
    for node in G.nodes():
        node_str = str(node).lower()
        
        if node_str in kusts_df['Куст'].astype(str).str.lower().values:
            # Куст - оранжевый квадрат
            node_color = "#FFA500"
            node_shape = "square"
            params = G.nodes[node]
            
            # УЛУЧШЕННАЯ ИНФОРМАЦИЯ ДЛЯ КУСТА
            title = (f"КУСТ: {node}\n"
                    f"Дебит жидкости: {params.get('liq_rate', 0):.1f} м3/сут\n"
                    f"Дебит нефти: {params.get('oil_rate', 0):.1f} т/сут\n"
                    f"Обводненность: {params.get('water_cut', 0):.1f}%\n"
                    f"Вязкость: {params.get('viscosity', 0):.2f}\n"
                    f"КВЧ: {params.get('kvch', 0):.2f}\n"
                    f"Давление: {params.get('pressure', 0):.1f} атм\n"
                    f"Скважин: {params.get('well_count', 0)}")
        
        elif node.endswith('_well'):
            # Скважина - красный треугольник  
            node_color = "#FF0000"
            node_shape = "triangle"
            well_id = node.replace('_well', '')
            params = G.nodes[node] if 'liq_rate' in G.nodes[node] else {}
            
            title = (f"СКВАЖИНА: {well_id}\n"
                    f"Дебит жидкости: {params.get('liq_rate', 0):.1f} м3/сут\n"
                    f"Обводненность: {params.get('water_cut', 0):.1f}%")
        
        elif any(char.isdigit() for char in node) and not node.startswith('well_'):
            # Труба (содержит цифры) - зеленый ромб
            node_color = "#00FF00"
            node_shape = "diamond"
            params = G.nodes[node]
            title = (f"ТРУБА: {node}\n"
                    f"Дебит: {params.get('liq_rate', 0):.1f} м3/сут\n"
                    f"Обводненность: {params.get('water_cut', 0):.1f}%")
        
        else:
            # Другие узлы - голубой круг
            node_color = "#97C2FC"
            node_shape = "dot"
            params = G.nodes[node]
            title = (f"УЗЕЛ: {node}\n"
                    f"Дебит: {params.get('liq_rate', 0):.1f} м3/сут\n"
                    f"Обводненность: {params.get('water_cut', 0):.1f}%")
        
        nt.add_node(node, label=str(node), title=title, color=node_color, shape=node_shape,
                   size=25 if node_shape == "square" else 20)  # Увеличиваем размер кустов
    
    # Добавляем ребра и собираем данные для CSV
    data_dict = {
        'date': [], 'edge_id': [], 'source': [], 'target': [], 
        'liq_rate': [], 'oil_rate': [], 'water_cut': [],
        'viscosity': [], 'kvch': [], 'pressure': [], 
        'Qzh': [], 'Qn': [], 'Qv': [],
        'pressure_inlet_mpa': [], 'pressure_outlet_mpa': [], 'pressure_drop_mpa': [],
        'temperature_inlet': [], 'temperature_outlet': [], 'temperature_drop': [],
        'dp_dl_grav_pa_m': [], 'dp_dl_fric_pa_m': []
    }
    
    for edge in G.edges(data=True):
        source, target, data = edge
        edge_id = data.get('id', 'Unknown')
        
        # Получаем актуальные данные из ребра
        liq_rate = data.get('liq_rate', 0)
        oil_rate = data.get('oil_rate', 0)
        water_cut = data.get('water_cut', 0)
        viscosity = data.get('viscosity', 0)
        kvch = data.get('kvch', 0)
        pressure = data.get('pressure', 0)
        
        # Получаем данные по давлению и температуре
        pressure_inlet_mpa = data.get('pressure_inlet_mpa', 0)
        pressure_outlet_mpa = data.get('pressure_outlet_mpa', 0)
        pressure_drop_mpa = data.get('pressure_drop_mpa', 0)
        temperature_inlet = data.get('temperature_inlet', 0)
        temperature_outlet = data.get('temperature_outlet', 0)
        temperature_drop = data.get('temperature_drop', 0)
        dp_dl_grav_pa_m = data.get('dp_dl_grav_pa_m', 0)
        dp_dl_fric_pa_m = data.get('dp_dl_fric_pa_m', 0)
        
        print(f"ДОБАВЛЕНИЕ В CSV: {source} -> {target} ({edge_id}) - liq_rate: {liq_rate}")
        
        # Заполняем словарь для CSV
        data_dict['date'].append(date)
        data_dict['edge_id'].append(edge_id)
        data_dict['source'].append(source)
        data_dict['target'].append(target)
        data_dict['liq_rate'].append(liq_rate)
        data_dict['oil_rate'].append(oil_rate)
        data_dict['water_cut'].append(water_cut)
        data_dict['viscosity'].append(viscosity)
        data_dict['kvch'].append(kvch)
        data_dict['pressure'].append(pressure)
        data_dict['Qzh'].append(liq_rate)
        data_dict['Qn'].append(oil_rate)
        data_dict['Qv'].append(liq_rate * water_cut / 100)
        
        # Новые колонки для давления и температуры
        data_dict['pressure_inlet_mpa'].append(pressure_inlet_mpa)
        data_dict['pressure_outlet_mpa'].append(pressure_outlet_mpa)
        data_dict['pressure_drop_mpa'].append(pressure_drop_mpa)
        data_dict['temperature_inlet'].append(temperature_inlet)
        data_dict['temperature_outlet'].append(temperature_outlet)
        data_dict['temperature_drop'].append(temperature_drop)
        data_dict['dp_dl_grav_pa_m'].append(dp_dl_grav_pa_m)
        data_dict['dp_dl_fric_pa_m'].append(dp_dl_fric_pa_m)
        
        title = (f"ID: {edge_id}\n"
                f"От: {source} -> К: {target}\n"
                f"Дебит жидкости: {liq_rate:.2f} м3/сут\n"
                f"Дебит нефти: {oil_rate:.2f} т/сут\n"
                f"Обводненность: {water_cut:.2f}%\n"
                f"Вязкость: {viscosity:.2f}\n"
                f"КВЧ: {kvch:.2f}\n"
                f"Давление: {pressure:.2f} атм")
        
        # Добавляем информацию о давлении и температуре для труб
        if not source.endswith('_well'):
            title += (f"\nPвх: {pressure_inlet_mpa:.3f} МПа\n"
                     f"Pвых: {pressure_outlet_mpa:.3f} МПа\n"
                     f"ΔP: {pressure_drop_mpa:.4f} МПа\n"
                     f"Tвх: {temperature_inlet:.1f}°C\n"
                     f"Tвых: {temperature_outlet:.1f}°C\n"
                     f"ΔT: {temperature_drop:.2f}°C")
        
        edge_color = "#ff0000" if liq_rate > 0 else "#cccccc"
        nt.add_edge(source, target, title=title, color=edge_color, width=2 if liq_rate > 0 else 1)
    
    return nt, pd.DataFrame(data_dict)

def process_date(target_date, combined_df, tech_mode_df, kust_tube_df, angles_df, output_dir):
    """Обработка данных для конкретной даты"""
    print(f"Обработка даты: {target_date}")
    
    try:
        edges_df, kusts_df, wells_df, tech_mode_filtered, kust_tube_df, tube_geometry, tube_pvt, tube_initial_conditions, tube_angles = preprocess_data(
            combined_df, tech_mode_df, kust_tube_df, target_date, angles_df
        )
        
        if edges_df.empty or kusts_df.empty:
            print(f"Нет данных для даты {target_date}")
            return pd.DataFrame()
        
        G = build_graph(edges_df)
        
        # Обработка циклов
        removed_edges = find_and_remove_cycles(G, tech_mode_filtered)
        
        # Расчет потоков
        calculate_flow(G, kusts_df, wells_df, kust_tube_df, tech_mode_filtered, tube_geometry, tube_pvt, tube_initial_conditions, tube_angles)
        
        # Восстановление удаленных ребер
        add_removed_edges(G, removed_edges)
        
        # Визуализация
        nt, result_df = visualize_flow(G, kusts_df, wells_df, target_date)
        
        # Сохраняем визуализацию
        nt.save_graph(f"{output_dir}/graph_{target_date}.html")
        
        return result_df
        
    except Exception as e:
        print(f"Ошибка при обработке даты {target_date}: {e}")
        return pd.DataFrame()

def main():
    # Пути к файлам
    data_dir = "."
    combined_data_file = os.path.join(data_dir, "combined_data_301.csv")
    tech_mode_file = os.path.join(data_dir, "тех_режим_301.xlsx")
    kust_tube_file = os.path.join(data_dir, "Связка куст_труба.xlsx")
    angles_file = os.path.join(data_dir, "alt_with_deg.csv")  # Файл с углами наклона
    output_dir = "output"
    
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Загрузка данных
    combined_df, tech_mode_df, kust_tube_df = load_data(
        combined_data_file, tech_mode_file, kust_tube_file
    )
    
    # Загрузка данных об углах наклона
    angles_df = load_tube_angles(angles_file)
    
    print('Combined_data:\n',combined_df.head())
    print('Tech_mode_df:\n',tech_mode_df.head())
    print('kust_tube_df:\n',kust_tube_df.head())
    print('Angles data loaded:', not angles_df.empty)

    # Диапазон дат
    start_date = pd.Period("2015-06-01", "D")
    end_date = pd.Period("2025-09-01", "D")
    date_range = pd.period_range(start=start_date, end=end_date, freq="D")
    
    # Обработка по датам
    all_results = pd.DataFrame()
    
    for date in date_range:
        result_df = process_date(date, combined_df, tech_mode_df, kust_tube_df, angles_df, output_dir)
        if not result_df.empty:
            all_results = pd.concat([all_results, result_df], ignore_index=True)
    
    # Сохраняем результаты
    if not all_results.empty:
        all_results.to_csv(f"{output_dir}/refined_tech_mode.csv", index=False)
        print(f"Результаты сохранены в {output_dir}/refined_tech_mode.csv")
        print(f"Всего записей: {len(all_results)}")
        print(f"Колонки в результате: {list(all_results.columns)}")
    
    print("Обработка завершена")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    total_time = time.perf_counter() - start_time
    print(f"Общее время выполнения: {total_time:.2f} секунд")