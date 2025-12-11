
import numpy as np
from scipy.integrate import solve_ivp

class RoughTemperatureModel():
    """
    Модель показывает хорошую сходимость с PROSPER

    length - длина участка, м
    tid - внутренний диаметр участка, м
    tir - внутренняя шероховатость участка, м
    q_heat_in - тепловой поток на кельвин, Вт/K (массовый расход каждой фазы*теплоемкость каждой фазы)
    surrounding_temperature - температура окружающей среды, градусы Цельсия
    htc - коэффициент теплопроводности, (Вт/м2/K)
    conditions - температура потока на входе, градусы Цельсия
    direction - направление расчета.
    """
    def __init__(self, length:float, tid:float, tir, q_heat_in_by_degree:float, surrounding_temperature:float,
                 htc: float, t_bound: float, direction: int = 1):
        self.q_heat_in_by_degree = q_heat_in_by_degree
        self.length = length # m
        self.tid = tid # m
        self.tir = tir # m
        self.surrounding_temperature = surrounding_temperature # deg C

        self.calculation_step = 100 # для определения количества шагов трубопровода
        self.minimal_steps_counts = 5 # минимальное количество расчетных шагов трубопровода

        self.htc = htc  # U (W/m2/K)
        self.t_bound = t_bound  # Temperature (C)

        self.direction = direction  # direction = 1 (от забоя), direction = -1 (от устья)

        self.x = None  # Диапазон решения
        self.is_init = False

    def temperature_grad(self, x_i, t_avg, coef: float, temp:float) -> float:
        """
        coef - 1/(kg/s*J/Kg/K)
        """
        # print(f"t_avg={t_avg}, temp={temp}, coef={coef}")
        return coef * (t_avg[0] - temp)  # C/m

    def _prepare(self):

        l_o = 0
        l_n = self.length

        if self.direction == -1:
            _temp = l_n
            l_n = l_o
            l_o = _temp

        step = self.calculation_step
        num = int(max(l_o, l_n)/step)
        num = max(num, self.minimal_steps_counts)
        self.x = np.linspace(l_o, l_n, num)
        self.is_init = True

    def run(self):

        if not self.is_init:
            self._prepare()

        t_o = self.t_bound
        
        # print(f'self/direction: {self.direction}, self.q_heat_in_by_degree: {self.q_heat_in_by_degree}, self.htc: {self.htc}, self.tid: {self.tid}, self.tir: {self.tir}')

        coef = -self.direction * np.pi / self.q_heat_in_by_degree * self.htc * (self.tid-2*self.tir)

        # print(f'coef: {coef}')

        l_o, l_n = self.x[0], self.x[-1]

        # print(f'l_o {l_o}, l_n: {l_n}, t_o: {t_o}')
        # print(f'self.x: {self.x}')
        # Выполнение расчета
        solution = solve_ivp(self.temperature_grad, t_span=(l_o, l_n), y0=[(t_o)], t_eval=self.x,
                             args=(coef, self.surrounding_temperature))

        # print(f'solution_ivp: {solution}')

        x_arr = solution.t[::-1]
        y_arr = solution.y[0][::-1]

        return lambda x: np.interp(x, x_arr, y_arr)

if __name__ == '__main__':

    # Расчет теплового потока.
    q_oil_mass_rate = 0.46296296296296297 # kg/s
    q_wat_mass_rate = 0.5787037037037037 # kg/s
    q_gas_mass_rate = 0.021701388888888888 # kg/s
    HEAT = (2200, 2100, 4200)  # HTC, CO, CG, CW
    mass_rate = q_oil_mass_rate + q_gas_mass_rate + q_wat_mass_rate
    c_p = HEAT[0] * (q_oil_mass_rate / mass_rate) + HEAT[2] * (q_wat_mass_rate / mass_rate) + HEAT[1] * (
                q_gas_mass_rate / mass_rate)

    length = 3555 # m
    tid = 0.062 # m
    surrounding_temperature = 15
    HTC = 15
    tir = 5e-5
    T_INPUT = 58  # degrees C # Температура со стороны источников
    q_heat_in_by_degree = mass_rate*c_p # Вт

    solution = RoughTemperatureModel(length, tid, tir, q_heat_in_by_degree, surrounding_temperature, HTC, T_INPUT, direction=-1).run()

    print(solution(0))
    answer = 17.205506 # 17.21609