"""
    Version 1.1.

    This program is used to calculate the flux motion of ReBCO stack/coil.
    The calculation theory can be found at: https://dx.doi.org/10.1088/1361-6668/abc567
    Copyright (C) 2020  Beijing Eastforce Superconducting Technology Co., Ltd.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program.

    If you have any question about this program, please contact me.

    Dr. Lingfeng Lai
    Email:lailingfeng@eastfs.com
    2020-11-20

    Python Version: 3.6
    Modules used: numpy, matplotlib, scipy
"""
import os
from scipy.linalg import cholesky
from scipy.integrate import solve_ivp, ode
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import time
import pickle
import numba as nb
from PureHeatEquation import HeatEq
from sklearn.preprocessing import StandardScaler
from scipy.integrate import quad,quad_vec
#330/200)*
class YBCOtape:
    """
    class for YBCO tape  
    """
    def __init__(self, a=4e-3, b=1e-6, nvalue=38, jc0=(330/200)*2.8e10, B0=0.04265, k=0.29515, alpha=0.7):
        """
        define the YBCO tape
        ratio for j0 conversion (330/200)
        ic0 is 313
        :param a: tape width
        :param b: tape thickness
        :param nvalue: n value
        :param jc0: ciricle current density
        :param B0: parameter for Kim model
        :param k:  parameter for Kim model
        :param alpha:  parameter for Kim model
        """
        self.a = a
        self.b = b
        self.nvalue = nvalue
        self.jc = np.array(jc0)
        self.ic = jc0 *(a*b)
        self.B0 = B0
        self.k = k
        self.alpha = alpha

    def jcB(self, Bpara, Bperp):
        """
        calculte the critical curent density in magnetic field parallel field Bpara and perpendicular field Bperp.
        :param Bpara: parallel field
        :param Bperp: perpendicular field
        :return: critical current density
        """
        return self.jc / (1 + np.sqrt((self.k * Bpara)**2 + Bperp**2) / self.B0)**self.alpha


class Coil:
    """
    class for the YBCO coil/stack
    """
    def __init__(self, rin=0.1, rout=0.2, zmin=0.1, count_turns=10):
        """
        define the geometry of the coil/stack
        :param rin:  inner diameter
        :param rout:  outter diameter
        :param zmin:  the bottom position of the coil
        :param count_turns:  total turn of the coil
        """
        self.rin = rin
        self.rout = rout
        self.zmin = zmin
        self.count_turns = count_turns


class AClossFT2D:
    """
    class to solve the flux penetration process
    """
    def __init__(self, tape_type=YBCOtape(4e-3, 20, 120), csys=0, sym=(0, 0)):
        """
        define the ReBCO tape used, the coordinate system, and symmetry condition.
        :param tape_type: (YBCOtape) YBCO tape defined
        :param csys: 0: Cartesian coordinate system (stack)  1: Cylindrical coordinates (coil)
        :param sym: (x-z, y-z) 1: 'symmetry', -1: 'anti- symmetry', 0: 'no-symmetry'
        """
        self.tape_type = tape_type
        self.csys = csys
        self.sym = sym
        self.coils = []
        self.save_time = []
        self.save_time_step = []
        self.j_limit = 1.4
        self.timestep_type = 0
        self.timestep_min = 1e-5
        self.timestep_max = 1e-4
        self.time_point_no = 0
        self.dta = []
        self.time_array = []
        self.power = []
        self.ny = 0
        self.dy = 0
        self.dx = 0
        self.count_turns = 0
        self.posi_x = np.empty(0)
        self.posi_y = np.empty(0)
        self.count_ele = 0
        self.Ba = 0
        self.Ba_fre = 0
        self.Ba_phi = 0
        self.Ia = 0
        self.Ia_fre = 0
        self.Ia_phi = 0
        self.jt = []
        self.jc = []
        self.L = None
        self.Qij_inv = None
        self.M = None
        self.LM_inv = None
        self.Mbx = None
        self.Mby = None
        self.step = 0
        print('The current date ' + time.asctime(time.localtime(time.time())))

    def add_coil(self, rin=0.1, rout=0.2, zmin=0.0, count_turns=10):
        """
        adding a coil in the system
        :param rin: innter diameter
        :param rout:  outter diameter
        :param zmin:   the bottom position of the coil
        :param count_turns:  the turn number of the coil
        :return: None
        """
        self.coils.append(Coil(rin, rout, zmin, count_turns))

    def mesh(self, ny):
        """
        mesh, diving each tape into ny element evenly
        :param ny: number of elements in each tape
        :return: None
        """
        
        tm1 = time.perf_counter()
        self.ny = ny
        self.dy = self.tape_type.a / ny
        self.dx = self.tape_type.b
        self.count_turns = sum([c.count_turns for c in self.coils])
        #print(f'count turns {self.count_turns}')
        self.posi_x = np.empty(0)
        self.posi_y = np.empty(0)
        for i, coil in enumerate(self.coils):
            #[0,1,2,3,...ny] +0.5 
            yk_new = (np.arange(0, ny) + 0.5) * self.tape_type.a / ny + coil.zmin
            #print(f'yk_new {yk_new}')
            if coil.count_turns == 1:
                ValueError('MUST HAVE COUNTS TURNS GREATER THAN 1')
            else:
                dis_turn_x = (coil.rout - coil.rin) / (coil.count_turns - 1)
                #print(f'for coil {i} dis turn {dis_turn_x}')
            for turn_num in range(coil.count_turns):
                posi_x_turn = coil.rin + turn_num * dis_turn_x
                #print(f' for coil {i} posi_x_turn {posi_x_turn}')
                self.posi_x = np.hstack((self.posi_x, np.ones(ny) * posi_x_turn))
                self.posi_y = np.hstack((self.posi_y, yk_new.copy())) 
        
        
        #print(f' x pos {self.posi_x}')
        #print(f' y pos {self.posi_y}')

        self.count_ele = len(self.posi_x)
        tm2 = time.perf_counter()
        tm3= tm2-tm1
        print(f"Time taken for Mesh creation {np.round(tm3,3)} ")



    def set_time(self, save_time, timestep_min, timestep_max, timestep_type, j_limit=1.1):
        """
        define the solver
        :param save_time: times to save results
        :param timestep_min: minimum steptime
        :param timestep_max:  maximum step time
        :param timestep_type: 1: auto steptime between timestep_min and timestep_max 0: dt=timestep_min
        :param j_limit: Time step criterion used when timestep_type=1
        :return: None
        """
        self.save_time = save_time
        self.timestep_min = timestep_min
        self.timestep_max = timestep_max
        self.timestep_type = timestep_type
        self.j_limit = j_limit

    def set_field(self, Ba, fre, phi=0):
        """
        define the appled field
        :param Ba:  field amplitude
        :param fre: field frequency
        :param phi: phase angle
        :return: None
        """
        self.Ba = Ba
        self.Ba_fre = fre
        self.Ba_phi = phi

    def set_current(self, Ia, fre):
        """
        define the applied current
        :param Ia: current amplitude
        :param fre: current frequency
        :return: None
        """
        self.Ia = Ia
        self.Ia_fre = fre
        self.Ia_phi = 0

    def _cal_b(self, time1):
        """
        calculate the applied field in different time
        :param time1: array of times
        :return: array of magnetic fields
        """
        return np.sin(2 * np.pi * self.Ba_fre * time1 + self.Ba_phi) * self.Ba

    def _cal_db(self, time1):
        """
        calculate the applied field gradient dB/dt in different time
        :param time1: array of times
        :return: array of magnetic fields gradient
        """
        return 2 * np.pi * self.Ba_fre * np.cos(2 * np.pi * self.Ba_fre * time1 + self.Ba_phi) * self.Ba

    def _cal_i(self, time1):
        """
        calculate the applied current in different time
        :param time1: array of times
        :return:  array of current
        """
        return np.sin(2 * np.pi * self.Ia_fre * time1 + self.Ia_phi) * self.Ia

    def _cal_di(self, time1):
        """
        calculate the applied current gradient dI/dt in different time
        :param time1: array of times
        :return:  array of current gradient
        """
        return 2 * np.pi * self.Ia_fre * np.cos(2 * np.pi * self.Ia_fre * time1 + self.Ia_phi) * self.Ia
    
    # Using numba decorator is a trivial way to parallelize a class method
    #@nb.jit(nopython=False,forceobj=True)
    def run(self, build_matrix=True, cal_jcb=True):
        """
        solve the problem
        :param build_matrix: True: rebuild the matrixes (usually set as True)
        :param cal_jcb: True: Considering Jc(B) characteristics of the tape
        :return: None
        """
        t1 = time.perf_counter()
        #print(time.asctime(time.localtime(time.time())) + ': Start solving.')
        self.jt = [None] * len(self.save_time)  # Array to current density distribution
        self.jc = [None] * len(self.save_time)  # Array to critical current density distribution
        dx = self.dx  # element size
        dy = self.dy  # element size 

        Ec = 1e-4
        # Reshape the position arrays to N*1
        posi_x = np.reshape(self.posi_x, (self.count_ele, 1))
        posi_y = np.reshape(self.posi_y, (self.count_ele, 1))

        # Calculate the matr ixes
        if isinstance(self.L, type(None)) or build_matrix:
            #print(time.asctime(time.localtime(time.time())) + ': Start buliding matrices. ')
            if self.csys == 0:
                # Cartesian coordinate system
                #print(time.asctime(time.localtime(time.time())) + ': Cartesian coordinate system.')

                self._build_matrix_car()
            elif self.csys == 1:
                # Cylindrical coordinate system
                #print(time.asctime(time.localtime(time.time())) + ': Cylindrical coordinates system.')
                self._build_matrix_cyl()
                
                #self._build_matrix_cyl_corrected()
            else:
                #print(time.asctime(time.localtime(time.time())) + ': Wrong coordinates system.')
                #return
                ValueError('Provide 0 or 1 to indicate Cartesian or Cylindrical coordinates')
        else:
            #print(time.asctime(time.localtime(time.time())) + ': Using last matrices.')
            ValueError(' L is not properly initialized \
                       Or did not provide correct csys 0 or 1 to indicate Cartesian or Cylindrical coordinates')

        time_run = 0    # model time
        self.step = 0   # model step
        self.save_time_step.append(self.step)  # Array to record steps of saved results
        self.time_point_no = 1  # number of saved results
        self.dta = []  # Array to record each step time used in solving
        self.time_array = []  # Array to record the time for each step
        self.power = []  # Array to record the AC loss in each step
        lastjt = np.zeros((self.count_ele, 1))  # current density in last stedp
        self.jc[0] = np.ones((self.count_ele, 1)) * self.tape_type.jc  # Critical current density at step [0]
        self.jt[0] = lastjt.copy()  # current density at step [0]
        reach_save = False  # switch to record
        i=0
        while self.time_point_no < len(self.save_time):
        #for _ in range(len(self.save_time)):
            tl1= time.perf_counter()
        #for i, n in enumerate(range(len(self.save_time))):
            # Main solving loop
            self.time_array.append(time_run)  # record the model time
            if cal_jcb:
                # Calculate the magnetic filed at each element and corresponding jc
                bx = np.matmul(self.Mbx, lastjt) + self._cal_b(time_run) * self.posi_y.reshape((self.count_ele, 1))
                by = np.matmul(self.Mby, lastjt)
                #print(f'by {by}')
                #print(f'bx {bx}')
    
                jc = self.tape_type.jcB(by, bx)
            else: 
                jc = self.tape_type.jc
            Ee = np.abs(Ec * (lastjt / jc)**self.tape_type.nvalue) * np.sign(lastjt)
            if np.isnan(Ee[0]):
                # Record the Jt when the calculation does not converge
                self.breakj = lastjt
                break
            # power_temp = ((Ee * lastjt) * (abs(lastjt) > jc)+(Ec * lastjt**2 / jc)*(abs(lastjt) <= jc)).reshape((self.count_turns, self.ny)).sum(axis=1)
            power_temp = (Ee * lastjt).reshape((self.count_turns, self.ny)).sum(axis=1)  # calculte the AC loss
            self.power.append(power_temp)   # Record the Ac loss
            # Calculate the first part of dj
            jtdb = np.matmul(self.Qij_inv, Ee + posi_y * self._cal_db(time_run))
            # Calculate the Ea
            Ea = np.matmul(self.LM_inv, self._cal_di(time_run) / dx / dy * np.ones((self.count_turns, 1))
                             - np.matmul(self.L, jtdb))
            # Calculate the second part of dj
            jtdc = np.matmul(self.M, Ea)
            dj = jtdb + jtdc
            # Calculate dt 
            if self.timestep_type == 1:
                # auto dt
                s1 = dj > 0
                s2 = dj < 0
                dts = self.timestep_max
                if s1.max():
                    dts = min(self.timestep_max, abs((self.j_limit * jc - lastjt)[s1] / dj[s1]).min())
                if s2.max():
                    dts = min(dts, abs((-self.j_limit * jc - lastjt)[s2] / dj[s2]).min())
                dts = max(dts, self.timestep_min)
            else:
                # fixed dt
                dts = self.timestep_min
            if dts >= self.save_time[self.time_point_no] - time_run:
                # Reach the save point
                reach_save = True
                dt = self.save_time[self.time_point_no] - time_run
            else:
                dt = dts
            
            self.dta.append(dt)  # Record dt
            time_run += dt  # update time
            lastjt += dj * dt  # update current
            self.step += 1  # update step
            if reach_save:
                # Record jt jc
                self.jt[self.time_point_no] = lastjt.copy()
                self.jc[self.time_point_no] = jc.copy()
                #print(time.asctime(time.localtime(time.time())) + ': Result at time = %5.4f is recorded.' % time_run)
                self.time_point_no += 1
                i+=1
                self.save_time_step.append(self.step)
                reach_save = False
                tl2= time.perf_counter()
                tl3 =tl2 - tl1
                print(f'TOTAL TIME FOR LAST STEP {np.round(tl3,3)}')
        
        t2=time.perf_counter()
        t3= t2- t1
        #print(f'shape of self.jt{np.array(self.jt).shape}')
        print(f'TOTAL TIME TAKEN {np.round(t3,3)}')



    def run_rosenbrock(self, build_matrix=True, cal_jcb=True):
        """
        solve the problem
        :param build_matrix: True: rebuild the matrixes (usually set as True)
        :param cal_jcb: True: Considering Jc(B) characteristics of the tape
        :return: None
        """
        t1 = time.perf_counter()
        #print(time.asctime(time.localtime(time.time())) + ': Start solving.')
        self.jt = [None] * len(self.save_time)  # Array to current density distribution
        self.jc = [None] * len(self.save_time)  # Array to critical current density distribution
        dx = self.dx  # element size
        dy = self.dy  # element size 

        Ec = 1e-4
        # Reshape the position arrays to N*1
        posi_x = np.reshape(self.posi_x, (self.count_ele, 1))
        posi_y = np.reshape(self.posi_y, (self.count_ele, 1))

        # Calculate the matrixes
        if isinstance(self.L, type(None)) or build_matrix:
            #print(time.asctime(time.localtime(time.time())) + ': Start buliding matrices. ')
            if self.csys == 0:
                # Cartesian coordinate system
                #print(time.asctime(time.localtime(time.time())) + ': Cartesian coordinate system.')

                self._build_matrix_car()
            elif self.csys == 1:
                # Cylindrical coordinate system
                #print(time.asctime(time.localtime(time.time())) + ': Cylindrical coordinates system.')
                self._build_matrix_cyl()
                
                #self._build_matrix_cyl_corrected()
            else:
                #print(time.asctime(time.localtime(time.time())) + ': Wrong coordinates system.')
                #return
                ValueError('Provide 0 or 1 to indicate Cartesian or Cylindrical coordinates')
        else:
            #print(time.asctime(time.localtime(time.time())) + ': Using last matrices.')
            ValueError(' L is not properly initialized \
                       Or did not provide correct csys 0 or 1 to indicate Cartesian or Cylindrical coordinates')

        time_run = 0    # model time
        self.step = 0   # model step
        self.save_time_step.append(self.step)  # Array to record steps of saved results
        self.time_point_no = 1  # number of saved results
        self.dta = []  # Array to record each step time used in solving
        self.time_array = []  # Array to record the time for each step
        self.power = []  # Array to record the AC loss in each step
        lastjt = np.zeros((self.count_ele, 1))  # current density in last stedp
        self.jc[0] = np.ones((self.count_ele, 1)) * self.tape_type.jc  # Critical current density at step [0]
        self.jt[0] = lastjt.copy()  # current density at step [0]
        reach_save = False  # switch to record
        i=0
        while self.time_point_no < len(self.save_time):
            #print(f'lastjt{lastjt}')
            tl1= time.perf_counter()
        #for i, n in enumerate(range(len(self.save_time))):
            # Main solving loop
            # consider tiny step inbetween our next time
            ib_time_run = time_run+self.timestep_min/3
            #assert(ib_time_run >= self.timestep_min)
            self.time_array.append(time_run)  # record the model time
            if cal_jcb:
                # Calculate the magnetic filed at each element and corresponding jc
                bx = np.matmul(self.Mbx, lastjt) + self._cal_b(time_run) * self.posi_y.reshape((self.count_ele, 1))
                by = np.matmul(self.Mby, lastjt)
                bx2 = np.matmul(self.Mbx, lastjt) + self._cal_b(ib_time_run) * self.posi_y.reshape((self.count_ele, 1))
                by2 = np.matmul(self.Mby, lastjt)
        
                jc = self.tape_type.jcB(by, bx)
                jc2 = self.tape_type.jcB(by2, bx2)

            else:
                jc = self.tape_type.jc

    
            Ee = np.abs(Ec * (lastjt / jc)**self.tape_type.nvalue) * np.sign(lastjt)
            print(f'this is Ee {Ee}')
            Ee2 = np.abs(Ec * (lastjt / jc2)**self.tape_type.nvalue) * np.sign(lastjt)
            if np.isnan(Ee[0]):
                # Record the Jt when the calculation does not converge
                self.breakj = lastjt
                break
            # power_temp = ((Ee * lastjt) * (abs(lastjt) > jc)+(Ec * lastjt**2 / jc)*(abs(lastjt) <= jc)).reshape((self.count_turns, self.ny)).sum(axis=1)
            power_temp = (Ee * lastjt).reshape((self.count_turns, self.ny)).sum(axis=1)  # calculte the AC loss
               # Record the Ac loss
            # Calculate the first part of dj
            jtdb = np.matmul(self.Qij_inv, Ee + posi_y * self._cal_db(time_run))
            jtdb2 = np.matmul(self.Qij_inv, Ee2 + posi_y * self._cal_db(ib_time_run))

            # Calculate the Ea
            Ea = np.matmul(self.LM_inv, self._cal_di(time_run) / dx / dy * np.ones((self.count_turns, 1))
                             - np.matmul(self.L, jtdb))
            Ea2 = np.matmul(self.LM_inv, self._cal_di(ib_time_run) / dx / dy * np.ones((self.count_turns, 1))
                             - np.matmul(self.L, jtdb2))
            # Calculate the second part of dj
            jtdc = np.matmul(self.M, Ea)
            jtdc2 = np.matmul(self.M, Ea2)
            dj1 = jtdb + jtdc
            dj2 = jtdb2 + jtdc2
            # time average sum 
            # Nonlinear dynamic system have rapid changes over varying time scales
            # ie due to phase of power supply and rapid coupling nonlineaity of the system the dt is hard to parameterize
            # in typical Euler Time Integration
            # Rosenbrock semi-implicitt trapezoid method provides stability taking into account the future derivative hence semi implicit
            dj = (dj1)
            print(f'this is dj {dj}')
            # Numerical technique Lai uses to Determine dt 
            if self.timestep_type == 1:
                # auto dt
                s1 = (dj) > 0
                s2 = (dj) < 0
                dts = self.timestep_max
                if s1.max():
                    dts = min(self.timestep_max, abs((self.j_limit * jc - lastjt)[s1] / dj[s1]).min())
                if s2.max():
                    dts = min(dts, abs((-self.j_limit * jc - lastjt)[s2] / dj[s2]).min())
                dts = max(dts, self.timestep_min)
            else:
                # fixed dt
                dts = self.timestep_min
            if dts >= self.save_time[self.time_point_no] - time_run:
                # Reach the save point
                reach_save = True
                dt = self.save_time[self.time_point_no] - time_run
            else:
                dt = dts
            
            
            time_run += dt  # update time
            lastjt += dj * dt  # update current
            self.step += 1  # update step
            if reach_save:
                # Record jt jc
                self.dta.append(dt)
                self.power.append(power_temp)
                  # Record dt
                self.jt[self.time_point_no] = lastjt.copy()
                self.jc[self.time_point_no] = jc.copy()
                #print(time.asctime(time.localtime(time.time())) + ': Result at time = %5.4f is recorded.' % time_run)
                self.time_point_no += 1
                i+=1
                self.save_time_step.append(self.step)
                reach_save = False
                tl2= time.perf_counter()
                tl3 =tl2 - tl1
                #print(f'TOTAL TIME FOR FIRST STEP {np.round(tl3,3)}')
        self.cumdta = np.array(self.dta).reshape(-1,).cumsum()
        print(f'self.dta is {self.dta}')
        print(f'cumdta is {self.cumdta}')
        #print(f'self.jt{self.jt}')
    

    def run_rosenbrock_scipy(self, build_matrix=True, cal_jcb=True):


        def run_rosenbrock_sub():
            #self.save_time = t
            """
            solve the problem
            :param build_matrix: True: rebuild the matrixes (usually set as True)
            :param cal_jcb: True: Considering Jc(B) characteristics of the tape
            :return: None
            """
            t1 = time.perf_counter()
            #print(time.asctime(time.localtime(time.time())) + ': Start solving.')
            self.jt = [None] * len(self.save_time)  # Array to current density distribution
            self.jc = [None] * len(self.save_time)  # Array to critical current density distribution
            dx = self.dx  # element size
            dy = self.dy  # element size 

            Ec = 1e-4
            # Reshape the position arrays to N*1
            posi_x = np.reshape(self.posi_x, (self.count_ele, 1))
            posi_y = np.reshape(self.posi_y, (self.count_ele, 1))

            # Calculate the matrixes
            if isinstance(self.L, type(None)) or build_matrix:
                #print(time.asctime(time.localtime(time.time())) + ': Start buliding matrices. ')
                if self.csys == 0:
                    # Cartesian coordinate system
                    #print(time.asctime(time.localtime(time.time())) + ': Cartesian coordinate system.')

                    self._build_matrix_car()
                elif self.csys == 1:
                    # Cylindrical coordinate system
                    #print(time.asctime(time.localtime(time.time())) + ': Cylindrical coordinates system.')
                    self._build_matrix_cyl()
                    
                    #self._build_matrix_cyl_corrected()
                else:
                    #print(time.asctime(time.localtime(time.time())) + ': Wrong coordinates system.')
                    #return
                    ValueError('Provide 0 or 1 to indicate Cartesian or Cylindrical coordinates')
            else:
                #print(time.asctime(time.localtime(time.time())) + ': Using last matrices.')
                ValueError(' L is not properly initialized \
                        Or did not provide correct csys 0 or 1 to indicate Cartesian or Cylindrical coordinates')

            time_run = 0    # model time
            self.step = 0   # model step
            self.save_time_step.append(self.step)  # Array to record steps of saved results
            self.time_point_no = 1  # number of saved results
            self.dta = []  # Array to record each step time used in solving
            self.time_array = []  # Array to record the time for each step
            self.power = []  # Array to record the AC loss in each step
            djss=[]  # save dj/dt for scipy integration form
            lastjt = np.zeros((self.count_ele, 1))  # current density in last stedp
            self.jc[0] = np.ones((self.count_ele, 1)) * self.tape_type.jc  # Critical current density at step [0]
            self.jt[0] = lastjt.copy()  # current density at step [0]
            reach_save = False  # switch to record
            i=0
            while self.time_point_no < len(self.save_time):
                #print(f'lastjt{lastjt}')
                tl1= time.perf_counter()
            #for i, n in enumerate(range(len(self.save_time))):
                # Main solving loop
                # consider tiny step inbetween our next time
                ib_time_run = time_run+self.timestep_min
                assert(ib_time_run >= self.timestep_min)
                self.time_array.append(time_run)  # record the model time
                if cal_jcb:
                    # Calculate the magnetic filed at each element and corresponding jc
                    bx = np.matmul(self.Mbx, lastjt) + self._cal_b(time_run) * self.posi_y.reshape((self.count_ele, 1))
                    by = np.matmul(self.Mby, lastjt)
                    bx2 = np.matmul(self.Mbx, lastjt) + self._cal_b(ib_time_run) * self.posi_y.reshape((self.count_ele, 1))
                    by2 = np.matmul(self.Mby, lastjt)
            
                    jc = self.tape_type.jcB(by, bx)
                    jc2 = self.tape_type.jcB(by2, bx2)

                else:
                    jc = self.tape_type.jc

        
                Ee = np.abs(Ec * (lastjt / jc)**self.tape_type.nvalue) * np.sign(lastjt)
                Ee2 = np.abs(Ec * (lastjt / jc2)**self.tape_type.nvalue) * np.sign(lastjt)
                if np.isnan(Ee[0]):
                    # Record the Jt when the calculation does not converge
                    self.breakj = lastjt
                    break
                # power_temp = ((Ee * lastjt) * (abs(lastjt) > jc)+(Ec * lastjt**2 / jc)*(abs(lastjt) <= jc)).reshape((self.count_turns, self.ny)).sum(axis=1)
                power_temp = (Ee * lastjt).reshape((self.count_turns, self.ny)).sum(axis=1)  # calculte the AC loss
                self.power.append(power_temp)   # Record the Ac loss
                # Calculate the first part of dj
                jtdb = np.matmul(self.Qij_inv, Ee + posi_y * self._cal_db(time_run))
                jtdb2 = np.matmul(self.Qij_inv, Ee2 + posi_y * self._cal_db(ib_time_run))

                # Calculate the Ea
                Ea = np.matmul(self.LM_inv, self._cal_di(time_run) / dx / dy * np.ones((self.count_turns, 1))
                                - np.matmul(self.L, jtdb))
                Ea2 = np.matmul(self.LM_inv, self._cal_di(ib_time_run) / dx / dy * np.ones((self.count_turns, 1))
                                - np.matmul(self.L, jtdb2))
                # Calculate the second part of dj
                jtdc = np.matmul(self.M, Ea)
                jtdc2 = np.matmul(self.M, Ea2)
                dj1 = jtdb + jtdc
                dj2 = jtdb2 + jtdc2
                # time average sum 
                # Nonlinear dynamic system have rapid changes over varying time scales
                # ie due to phase of power supply and rapid coupling nonlineaity of the system the dt is hard to parameterize
                # in typical Euler Time Integration
                # Rosenbrock semi-implicitt trapezoid method provides stability taking into account the future derivative hence semi implicit
                dj = dj1
                
                # Numerical technique Lai uses to Determine dt 
                if self.timestep_type == 1:
                    # auto dt
                    s1 = (dj) > 0
                    s2 = (dj) < 0
                    dts = self.timestep_max
                    if s1.max():
                        dts = min(self.timestep_max, abs((self.j_limit * jc - lastjt)[s1] / dj[s1]).min())
                    if s2.max():
                        dts = min(dts, abs((-self.j_limit * jc - lastjt)[s2] / dj[s2]).min())
                    dts = max(dts, self.timestep_min)
                else:
                    # fixed dt
                    dts = self.timestep_min
                if dts >= self.save_time[self.time_point_no] - time_run:
                    # Reach the save point
                    reach_save = True
                    dt = self.save_time[self.time_point_no] - time_run
                else:
                    dt = dts
                
                
                time_run += dt  # update time
                lastjt += dj * dt  # update current
                self.step += 1  # update step
                if reach_save:
                    # Record jt jc
                    self.jt[self.time_point_no] = lastjt.copy()
                    self.jc[self.time_point_no] = jc.copy()
                    #print(time.asctime(time.localtime(time.time())) + ': Result at time = %5.4f is recorded.' % time_run)
                    self.time_point_no += 1
                    i+=1
                    self.save_time_step.append(self.step)
                    reach_save = False
                    self.dta.append(dt)  # Record dt
                    djss.append(dj)
                    tl2= time.perf_counter()
                    tl3 =tl2 - tl1
                    #print(f'TOTAL TIME FOR FIRST STEP {np.round(tl3,3)}')

            return np.array(djss),np.array(self.dta).reshape(-1,)
        #print(self.dta)
        #by default numpys sort fucntion uses ascending order
        dj1,dta = run_rosenbrock_sub()
        dta1 = dta.cumsum()
        #dta1 = np.sort(dta1,axis = 0, kind = 'quicksort')
        #dta1 = dta1.cumsum()
        ogshape = dj1.shape
        print(f'ogshape {ogshape}')
        dj1 = dj1.reshape((1500,12))
        print(dj1)
        print(dj1.shape)
        print(dta1)
        def IVPDJDT(t,y):

            if t==0.0:
                return np.zeros((1500,1))
            
            #try: 
            print(dj1.shape)
            djdts = np.zeros((1500,1))
            print(f'time passed into ode {t}')
            print(djdts.shape)
            print(f'index of {t} in dta1 is {np.where(dta1== t)}')
            #tolerance = 1e-12
            #if np.any(np.abs(dta1 - t) < tolerance):
                #raise ValueError('The time is not apart of the simulated DJDT ERROR')
            
            indexs = np.where(dta1== t)[0]
            djdts[:,indexs] = dj1[:,indexs]
            djdts = djdts.reshape((1500,1))
            print(djdts.shape)
            
            
            #print(dj[:][np.where(np.array(dta).reshape(-1,)==t)].shape)
            assert(djdts.shape == (1500,1))
            return(djdts)
            #except:
                    #print("TRY AGAIN NEXT TIME")
        
         # Time span from 0 to 5
        y0 = np.zeros(1500)  #


        #solver = ode(IVPDJDT).set_integrator('vode', method='bdf',atol=1e-7, rtol=1e-7)
        #solver.set_initial_value(y0,t=0 )
        t_span = [dta1[0], dta1[-1]]

# Solve the ODEs using solve_ivp with VODE method
        solution = solve_ivp(IVPDJDT, y0=y0, method='RK45')

# The solution object contains the results, including times and solution values
        times = solution.t
        y_sol = solution.y
        print(f'solution shape is {y_sol.shape}')
        print(f'solution is {y_sol}')
        print(f'evaluated at the times {times}')
        print(f'comapred with dta1 {dta1}')
       
        def semi_implicit_trapezoid_rosenbrock(J0, djdt, dt):
            """
            Semi-implicit trapezoidal integration using the Rosenbrock method.
            
            Parameters:
                J0 (numpy.ndarray): Initial values of the variable (column vector).
                djdt (numpy.ndarray): Matrix containing derivatives at each time step (columns).
                dt (numpy.ndarray): Array containing time steps.
                
            Returns:
                numpy.ndarray: Matrix containing integrated values of the variable for each time step (columns).
            """
            num_time_steps = len(dt)
            num_variables = self.count_ele
            
            J = np.zeros((num_variables, num_time_steps))  # Initialize integrated variable matrix
            J[:, 0] = J0  # Set initial values
            
            for i in range(1, num_time_steps):
                # Calculate J_i using trapezoidal rule with Rosenbrock method for each variable
                #for j in range(num_variables):
                J[:, i] = J[:, i - 1] + 0.5 * (djdt[:, i - 1] * dt[i - 1] + djdt[:, i] * dt[i]) 
            
            return J

        # Example usage
        #J0 = 0  # Initial value of the variable
        #djdt = np.array([2, 3, 1, 4])  # Derivatives at each time step
        #dt = np.array([0.1, 0.2, 0.3, 0.2])  # Time steps

        #self.jt = semi_implicit_trapezoid_rosenbrock(J0, dj1, dta1)
        # invert array back to original shape and datatype
        #self.jt.shape = ogshape
        #self.jt = list(self.jt)
        #print("Integrated Values:", integrated_values)
        #dj = dj.reshape(1500,12)
        #print(f'shape of dj{dj[:,0].shape}')
        #print(f'shape of dta{np.array(dta).shape}')
        #djdts = np.zeros(1500)
        #djdts = np.squeeze(dj[:,np.where(np.array(dta).reshape(-1,)==dta[0])],axis=2)
        #print(f'shape of djdts{djdts.shape}')
        

        #plt.figure(figsize=(8, 6))
        #plt.plot(solver.t, solver.y[0], label='Rosenbrock Method')
        #plt.xlabel('t')
        #plt.ylabel('y(t)')
        #plt.title('Solution of the ODE using Rosenbrock Method')
        #plt.legend()
        #plt.grid(True)
        #plt.show()    
        #print(time.asctime(time.localtime(time.time())) + ': Finish solving.')
    def run_rosenbrock_scipytwo(self, build_matrix=True, cal_jcb=True):
        
        # This will rerun over and over again inside of 
        if isinstance(self.L, type(None)) or build_matrix:
                #print(time.asctime(time.localtime(time.time())) + ': Start buliding matrices. ')
                if self.csys == 0:
                    # Cartesian coordinate system
                    #print(time.asctime(time.localtime(time.time())) + ': Cartesian coordinate system.')

                    self._build_matrix_car()
                elif self.csys == 1:
                    # Cylindrical coordinate system
                    #print(time.asctime(time.localtime(time.time())) + ': Cylindrical coordinates system.')
                    self._build_matrix_cyl()
                    
                    #self._build_matrix_cyl_corrected()
                else:
                    #print(time.asctime(time.localtime(time.time())) + ': Wrong coordinates system.')
                    #return
                    ValueError('Provide 0 or 1 to indicate Cartesian or Cylindrical coordinates')
        else:
            #print(time.asctime(time.localtime(time.time())) + ': Using last matrices.')
            ValueError(' L is not properly initialized \
                    Or did not provide correct csys 0 or 1 to indicate Cartesian or Cylindrical coordinates')

        def DJDT(t,y):
            #self.save_time = t
            """
            solve the problem
            :param build_matrix: True: rebuild the matrixes (usually set as True)
            :param cal_jcb: True: Considering Jc(B) characteristics of the tape
            :return: None
            """
            t1 = time.perf_counter()
            dt= t
            #print(time.asctime(time.localtime(time.time())) + ': Start solving.')
            #self.jt = [None] * len(1)  # Array to current density distribution
            #self.jc =[] # Array to critical current density distribution
            dx = self.dx  # element size
            dy = self.dy  # element size 
            Ec = 1e-4
            # Reshape the position arrays to N*1
            posi_x = np.reshape(self.posi_x, (self.count_ele, 1))
            posi_y = np.reshape(self.posi_y, (self.count_ele, 1))
            # Calculate the matrixes
            time_run = t    # model time
            self.step = 0   # model step
            self.save_time_step.append(self.step)  # Array to record steps of saved results
            self.time_point_no = 1  # number of saved results
            self.dta = []  # Array to record each step time used in solving
            self.time_array = []  # Array to record the time for each step
            self.power = []  # Array to record the AC loss in each step
              # save dj/dt for scipy integration form

            print(f'y input {y}')
            print(f'shape of y {y.shape}')
            y = y.reshape((-1,1))
            lastjt =y.reshape((self.count_ele,1))  # current density in last stedp
            self.jc = np.ones((self.count_ele, 1)) * self.tape_type.jc # Critical current density at step [0]
            #self.jt = lastjt.copy()  # current density at step [0]
             # switch to record
           
            
            #print(f'lastjt{lastjt}')
            tl1= time.perf_counter()
        #for i, n in enumerate(range(len(self.save_time))): 
            # Main solving loop
            # consider tiny step inbetween our next time
            
            self.time_array.append(time_run)  # record the model time
            if cal_jcb:
                # Calculate the magnetic filed at each element and corresponding jc
                bx = np.matmul(self.Mbx, lastjt) + self._cal_b(time_run) * self.posi_y.reshape((self.count_ele, 1))
                by = np.matmul(self.Mby, lastjt)
                
        
                jc = self.tape_type.jcB(by, bx)

            else:
                jc = self.tape_type.jc
            print(f'shape of jc {jc.shape}')
            Ee = np.abs(Ec * (lastjt / jc)**self.tape_type.nvalue) * np.sign(lastjt)
            print(f'shape of Ee {Ee.shape}')
            #self.power.append(power_temp)   # Record the Ac loss
            # Calculate the first part of dj
            jtdb = np.matmul(self.Qij_inv, Ee + posi_y * self._cal_db(time_run))
            print(f'shape of jtdb {jtdb.shape}')
            # Calculate the Ea
            Ea = np.matmul(self.LM_inv, self._cal_di(time_run) / dx / dy * np.ones((self.count_turns, 1))
            - np.matmul(self.L, jtdb))
            print(f'shape of Ea {Ea.shape}')
            # Calculate the second part of dj
            jtdc = np.matmul(self.M, Ea)
            print(f'shape of jtdc {jtdc.shape}')
            dj = jtdb + jtdc
            print(f'shape of dj {dj.shape}')
            t2=time.perf_counter()
            tl3 = t2-t1 
            print(f'TOTAL TIME FOR FIRST STEP {np.round(tl3,3)}')
            return dj
        #print(self.dta)
        #by default numpys sort fucntion uses ascending order
        #dj1,dta = run_rosenbrock_sub()
        #dta1 = dta.cumsum()
        #dta1 = np.sort(dta1,axis = 0, kind = 'quicksort')
        #dta1 = dta1.cumsum()
        #ogshape = dj1.shape
        #dj1 = dj1.reshape((1500,12))
        dta1prev = [0.0003411,0.00036508, 0.00126922, 0.00160255, 0.00193589, 0.00226922,
        0.00236844, 0.00241685, 0.00277779, 0.00311112, 0.00344446, 0.00377779]
        dta1 = np.linspace(3.3e-6,4e-5,13)
        t_span=[dta1[0],dta1[-1]]
        ogshape =(25, self.count_ele, 1)
        y0=np.zeros((1500))

# Solve the ODEs using solve_ivp with VODE method
        
        solution = solve_ivp(DJDT, t_span=t_span, y0=y0, method='LSODA',atol = 1e-8,rtol =1e-8)

# The solution object contains the results, including times and solution values
        times = solution.t
        y_sol = solution.y
        print(f'solution shape is {y_sol.shape}')
        print(f'solution is {y_sol}')
        print(f'evaluated at the times {times}')
        self.jt = y_sol
        
        self.jt = self.jt.reshape((25,1500,1))
        self.jt = list(self.jt)
        self.jc = np.zeros(ogshape)
        # set initial critical current density
        # calulate the jc[t] looping over time frames
        self.jc = list(self.jc)

        # recalculate self.jc
        for inc, time_run in enumerate(times):
                
                if inc ==0:
                    self.jc[inc] = np.ones((self.count_ele, 1)) * self.tape_type.jc
                else:
                # Calculate the magnetic filed at each element and corresponding jc
                    bx = np.matmul(self.Mbx, self.jt[inc]) + self._cal_b(time_run) * self.posi_y.reshape((self.count_ele, 1))
                    by = np.matmul(self.Mby, self.jt[inc])
                    jc = self.tape_type.jcB(by, bx)
                    self.jc[inc] = jc.copy()

        #plt.figure(figsize=(8, 6))
        #plt.plot(solver.t, solver.y[0], label='Rosenbrock Method')
        #plt.xlabel('t')
        #plt.ylabel('y(t)')
        #plt.title('Solution of the ODE using Rosenbrock Method')
        #plt.legend()
        #plt.grid(True)
        #plt.show()    
        #print(time.asctime(time.localtime(time.time())) + ': Finish solving.')
    
    
    
    @nb.jit(nopython=False,forceobj=True)
    def _build_matrix_car(self):
        """
        Build the matrixes used in Cartesian coordinate system
        :return: None
        """
        miu0 = 4 * np.pi * 1e-7
        dx = self.dx
        dy = self.dy
        r1_x, r2_x = np.meshgrid(self.posi_x, self.posi_x)
        #print (f'self.posi_x {self.posi_x}')
        #print(f'r1_x {r1_x}')
        #print(f'r2_x {r2_x}')
        #print (f'self.posi_y {self.posi_y}')
        #print(f'r1_y {r1_y}')
        #print(f'r2_y {r2_y}')
        r1_y, r2_y = np.meshgrid(self.posi_y, self.posi_y)
        eps = 0.005 * dx * dy
        # Self.L is a matrix containing the relation between the element number and its turn number
        self.L = np.zeros((self.count_turns, self.count_ele))
        for _, i in enumerate(range(self.count_turns)):
            self.L[i, i * self.ny: (i + 1) * self.ny] = np.ones(self.ny)
        Qij = -0.5 * np.log((r1_x - r2_x) ** 2 + (r1_y - r2_y) ** 2 + eps) / 2 / np.pi
        print(Qij)
        self.Mbx = miu0 * dx * dy * (r2_y - r1_y) / ((r1_x - r2_x)**2 + (r1_y - r2_y)**2 + eps) / 2 / np.pi
        self.Mby = miu0 * dx * dy * (r1_x - r2_x) / ((r1_x - r2_x)**2 + (r1_y - r2_y)**2 + eps) / 2 / np.pi
        dic_wym = {1: 'symmetry', -1: 'anti- symmetry'}
        if abs(self.sym[0]) == 1:
            
            Qij += -self.sym[0] * 0.5 * np.log((r1_x - r2_x) ** 2 + (r1_y + r2_y) ** 2 + eps) / 2 / np.pi
            self.Mbx += self.sym[0] * miu0 * dx * dy * (r2_y + r1_y) \
                        / ((r1_x - r2_x) ** 2 + (r1_y + r2_y) ** 2 + eps) / 2 / np.pi
            self.Mby += self.sym[0] * miu0 * dx * dy * (r1_x - r2_x) \
                        / ((r1_x - r2_x) ** 2 + (r1_y + r2_y) ** 2 + eps) / 2 / np.pi
        if abs(self.sym[1]) == 1:
            
            Qij += -self.sym[1] * 0.5 * np.log((r1_x + r2_x) ** 2 + (r1_y - r2_y) ** 2 + eps) / 2 / np.pi
            self.Mbx += self.sym[1] * miu0 * dx * dy * (r2_y - r1_y) \
                        / ((r1_x + r2_x) ** 2 + (r1_y - r2_y) ** 2 + eps) / 2 / np.pi
            self.Mby += self.sym[1] * miu0 * dx * dy * (-r1_x - r2_x) \
                        / ((r1_x + r2_x) ** 2 + (r1_y - r2_y) ** 2 + eps) / 2 / np.pi
        if abs(self.sym[0] * self.sym[1]) == 1:
            Qij += -self.sym[0] * self.sym[1] * 0.5 * np.log((r1_x + r2_x) ** 2 + (r1_y + r2_y) ** 2 + eps) \
                   / 2 / np.pi
            self.Mbx += self.sym[0] * self.sym[1] * miu0 * dx * dy * (r2_y + r1_y) \
                        / ((r1_x + r2_x)**2 + (r1_y + r2_y)**2 + eps) / 2 / np.pi
            self.Mby += self.sym[0] * self.sym[1] * miu0 * dx * dy * (-r1_x - r2_x) \
                        / ((r1_x + r2_x)**2 + (r1_y + r2_y)**2 + eps) / 2 / np.pi
        #del r1_x, r2_x, r1_y, r2_y
        #Qij = cholesky(Qij)
        self.Qij_inv = -1 / miu0 / dx / dy * np.linalg.inv(Qij)  # inverse kernel
        #del Qij
        self.M = np.matmul(self.Qij_inv, self.L.transpose())  # To calculate the second part of dj
        self.LM_inv = np.linalg.inv(np.matmul(self.L, self.M))
        #self.LM_inv = cholesky(self.LM_inv)  # solution of the linear equations
       
        
    #@nb.jit(nopython=False)
    def _build_matrix_cyl(self):
        """
         Build the matrixes used in Cylindrical coordinate system
        :return: None
        """
        tc1 = time.perf_counter()
        miu0 = 4 * np.pi * 1e-7
        dx = self.dx
        dy = self.dy
        r1_x, r2_x = np.meshgrid(self.posi_x, self.posi_x)
        r1_y, r2_y = np.meshgrid(self.posi_y, self.posi_y)
        eps = 0.005 * dx * dy

        #print (f'self.posi_x {self.posi_x}')
        #print(f'r1_x {r1_x}')
        #(f'r2_x {r2_x}')
        def calAphi(r_s, r_t, z_s, z_t):
            k2 = (4 * r_s * r_t) / ((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps)
            Aphi = (np.sqrt(r_s / r_t) * (sps.ellipk(k2) * (1 - k2 / 2) - sps.ellipe(k2))) / (np.pi *np.sqrt(k2))
            #Aphi = np.abs(Aphi )
            #phi = np.arctan2(z_s,r_s)
            # represent A_phi in caresian coordinates 
            #Aphi_carx = - Aphi* np.sin(phi)
            # = Aphi* np.cos(phi)
            #Aphi = np.sqrt(Aphi_carx**2 + Aphi_cary**2)
            del k2
            return Aphi
        
        def NumericalcalAphi(phi,r_s,r_t,z_s,z_t):
            # From EH Brandt 1996 Superconductors of finite thickness in a perpendicular magnetic field: Strips and slab
            # page 54 
            #phi = np.arctan(z_t/r_t )
            eta= (z_s - z_t)
            return (r_s * np.cos(phi *np.ones((r_t.shape[0],r_t.shape[1]))))/(2*np.pi*(eta**2 + r_s**2 + r_t **2 + 2*r_s*r_t*np.cos(phi) + eps)**(1/2))


        self.L = np.zeros((self.count_turns, self.count_ele))
        for _, i in enumerate(range(self.count_turns)):
            self.L[i, i * self.ny: (i + 1) * self.ny] = np.ones(self.ny)
        Qij = -calAphi(r1_x, r2_x, r1_y, r2_y)
        #if desired, the integral kernel may be ex- pressed in terms of elliptic 
        # integrals, but for computational purposes it is more convenient to evaluate the integral 
        # numerically.
        #epsabs =1e-250
        #epsrel =1e-09
        otarg = (r1_x, r2_x, r1_y, r2_y)
        #Qij, error = quad_vec(NumericalcalAphi,0,np.pi,args=otarg,epsabs =epsabs,epsrel = epsrel )
        #Qij = -Qij
        dis = min(dx, dy) /4
        print(Qij)
        
        def kprime(k):
            return sps.ellipe(k)/(k-k**3) - sps.ellipk(k)/k
        
        def eprime(k):
            return (sps.ellipe(k) - sps.ellipk(k))/k

        def calAphidr(r_s, r_t, z_s, z_t):
            dkdr= (4*r_s  - 2* (r_s +r_t))/(((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps))**2
            k2 = 4 * r_s * r_t / ((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps)
            term1 = (1* dkdr * np.sqrt(r_s* r_t) * sps.ellipk(k2))/(np.pi *-2 * (k2)**(3/2))  + \
            (1* np.sqrt(r_s/r_t) * sps.ellipk(k2))/(np.pi * 2* np.sqrt(k2))  + \
            (1* np.sqrt(r_s* r_t) * kprime(k2) * dkdr)/(np.pi * np.sqrt(k2)) 
            term2 =   (1* np.sqrt(r_s *r_t)* sps.ellipk(k2) * dkdr) / (4* np.pi*np.sqrt(k2)) -\
            (1* np.sqrt(r_s/r_t)* sps.ellipk(k2) *np.sqrt(k2)) / (np.pi*4) -\
            (1* np.sqrt(r_s*r_t)* kprime(k2) *dkdr *np.sqrt(k2)) / (np.pi*2) 
            term3 = (1*dkdr* np.sqrt(r_s * r_t) * sps.ellipe(k2)) / (np.pi *-2*(k2)**(3/2))  -\
            (1* np.sqrt(r_s/r_t) * sps.ellipe(k2)) / (np.pi*2* np.sqrt(k2))  -\
            (1* np.sqrt(r_s*r_t) * eprime(k2) *dkdr) / (np.pi *np.sqrt(k2)) 
            return term1 + term2 + term3

        def calAphidz(r_s, r_t, z_s, z_t):
            dkdz = ( (-2)* (z_s - z_t)*(-1))/ (((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps))**2
            k2 = 4 * r_s * r_t / ((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps)
            term1 = (1*dkdz * np.sqrt(r_s / r_t) * (sps.ellipk(k2))) / (np.pi * -2* (k2)**(3/2))  +\
            (1* np.sqrt(r_s / r_t) * kprime(k2)* dkdz) / (np.pi * np.sqrt(k2))
            term2 = (1* dkdz* np.sqrt(r_s / r_t) * (sps.ellipk(k2)))/ (np.pi *2* np.sqrt(k2))  -\
            (1* np.sqrt(k2)* np.sqrt(r_s / r_t) * (kprime(k2)* dkdz) )/ (np.pi) 
            term3 = (1* np.sqrt(r_s / r_t) * sps.ellipe(k2))/ (np.pi * -2*(k2)**(3/2))  -\
            (1* np.sqrt(r_s / r_t) * eprime(k2)* dkdz)/ (-np.pi *np.sqrt(k2)) 
            return term1 + term2 + term3
        
        #k2 = 4 * r1_x * r2_x / ((r2_x + r1_x) ** 2 + (r2_y - r1_y) ** 2 + eps)
        #self.Mbx = -miu0 * dx * dy * (calAphi(r1_x, r2_x, r1_y, r2_y + dis) - calAphi(r1_x, r2_x, r1_y, r2_y - dis)) / dis/2
        #otargx1= (r1_x, r2_x, r1_y, r2_y + dis)
        #otargx2= (r1_x, r2_x, r1_y, r2_y - dis)
       

        #Qijx1, _ = quad_vec(NumericalcalAphi,0,np.pi,args=otargx1,epsabs =epsabs,epsrel = epsrel )
        #print(Qijx1)
        #Qijx2, _ = quad_vec(NumericalcalAphi,0,np.pi,args=otargx2,epsabs =epsabs,epsrel = epsrel )
        #print(Qijx2)

        #Qijx = -Qijx
        #self.Mbx = -miu0 * dx * dy * (Qijx1 - Qijx2) / dis/2

        #del Qijx1,Qijx2
       
       # self.Mbx2 = -miu0 * dx * dy * calAphidz(r1_x, r2_x, r1_y, r2_y)/2
        #self.Mby = miu0 * dx * dy * ((r2_x + dis) * calAphi(r1_x, r2_x + dis, r1_y, r2_y) - (r2_x - dis) * calAphi(r1_x, r2_x - dis, r1_y, r2_y)) / r2_x / dis / 2
       # self.Mby2 = miu0 * dx * dy * calAphidr(r1_x, r2_x, r1_y, r2_y)/r2_x/2
        
        
        # FROM PAGE 14 of  Numerical Modeling of Superconducting Applications by: Bertrand Dutoit Francesco Grilli Frédéric Sirois
        # Provides solution given parameterized k^2  in the (r,z) plane representing a cross section with a current in the Phi direction
        #self.Mbx= (-miu0*dx*dy * (r2_y - r1_y))/(2*np.pi*r2_x*np.sqrt((r1_x + r2_x)**2 + (r2_y - r1_y)**2)) *\
        #           (sps.ellipk(np.sqrt(k2)) - sps.ellipe(np.sqrt(k2)) *(r1_x**2 + r2_x**2 + (r2_y-r1_y)**2)/((r1_x - r2_x)**2 + (r2_y - r1_y)**2 + eps))
        #self.Mbx = self.Mbx/2
        
        #self.Mby = (miu0 * dx*dy)/(2*np.pi*np.sqrt((r1_x + r2_x)**2 + (r2_y - r1_y)**2))*\
        #            (sps.ellipk(np.sqrt(k2)) + sps.ellipe(np.sqrt(k2))*(r1_x**2 - r2_x**2 -(r2_y - r1_y)**2)/((r1_x - r2_x)**2 + (r2_y - r1_y)**2+ eps))
        #self.Mby = self.Mby/r2_x/2
        #otargy1 = (r1_x, r2_x + dis, r1_y, r2_y)
        #otargy2 = (r1_x, r2_x - dis, r1_y, r2_y)


        #Qijy1, _ = quad_vec(NumericalcalAphi,0,np.pi,args=otargy1,epsabs =epsabs,epsrel =epsrel )
        #print(Qijy1)
        #Qijy2, _ = quad_vec(NumericalcalAphi,0,np.pi,args=otargy2,epsabs =epsabs,epsrel = epsabs )
        #print(Qijy2)
        #self.Mby = miu0 * dx * dy * ((r2_x + dis) * Qijy1 -
        #                              (r2_x - dis) *  Qijy2) / r2_x / dis / 2



        self.Mbx = -miu0 * dx * dy * (calAphi(r1_x, r2_x, r1_y, r2_y + dis) - calAphi(r1_x, r2_x, r1_y, r2_y - dis)) / dis / 2
        self.Mby = miu0 * dx * dy * ((r2_x + dis) * calAphi(r1_x, r2_x + dis, r1_y, r2_y) -
                                      (r2_x - dis) * calAphi(r1_x, r2_x - dis, r1_y, r2_y)) / r2_x / dis / 2
        #del Qijy1,Qijy2
        dic_wym = {1: 'symmetry', -1: 'anti- symmetry'} 
        if abs(self.sym[0]) == 1:
            otarg = (r1_x, r2_x, -r1_y, r2_y)
            Qij += -self.sym[0] * calAphi(r1_x, r2_x, -r1_y, r2_y)
            #Qij2, _ = quad_vec(NumericalcalAphi,0,np.pi,args=otarg,epsabs =epsabs,epsrel = epsrel )
            #Qij2 = -Qij2

            self.Mbx += - self.sym[0] * miu0 * dx * dy * (calAphi(r1_x, r2_x, -r1_y, r2_y + dis) -
                                                          calAphi(r1_x, r2_x, -r1_y, r2_y - dis)) / dis / 2
            self.Mby += self.sym[0] * miu0 * dx * dy * ((r2_x + dis) * calAphi(r1_x, r2_x + dis, -r1_y, r2_y) -
                                      (r2_x - dis) * calAphi(r1_x, r2_x - dis, -r1_y, r2_y)) / r2_x / dis / 2

            #Qij+= Qij2
            
            
            #otargx1s= (r1_x, r2_x, -r1_y, r2_y + dis)
            #otargx2s= (r1_x, r2_x, -r1_y, r2_y - dis)
            

            #Qijx1, _ = quad_vec(NumericalcalAphi,0,np.pi,args=otargx1s,epsabs =epsabs,epsrel = epsrel )
            #print(Qijx1)
            #Qijx2, _ = quad_vec(NumericalcalAphi,0,np.pi,args=otargx2s,epsabs =epsabs,epsrel = epsrel )
            #print(Qijx2)


            #Qijx = -Qijx
            #self.Mbx += -miu0 * dx * dy * (Qijx1 - Qijx2) / dis/2
            
        
    
            #otargy1s = (r1_x, r2_x + dis, -r1_y, r2_y)
            #otargy2s = (r1_x, r2_x - dis, -r1_y, r2_y)


            #Qijy1, _ = quad_vec(NumericalcalAphi,0,np.pi,args=otargy1s,epsabs =epsabs,epsrel =epsrel )
            #print(Qijy1)
            #Qijy2, _ = quad_vec(NumericalcalAphi,0,np.pi,args=otargy2s,epsabs =epsabs,epsrel = epsabs )
            #print(f'the last Q is {Qijy2}')
            #self.Mby += miu0 * dx * dy * ((r2_x + dis) * Qijy1 -
                                        #(r2_x - dis) *  Qijy2) / r2_x / dis / 2
            
            #k2 = 4 * r1_x * r2_x / ((r2_x + r1_x) ** 2 + (r2_y + r1_y) ** 2 + eps) 
            #self.Mbx4 =  (self.sym[0] * -miu0 *dx*dy * (r2_y + r1_y))/(2*np.pi*r2_x*np.sqrt((r1_x + r2_x)**2 + (r2_y + r1_y)**2 )) *\
            #(sps.ellipk(np.sqrt(k2)) - sps.ellipe(np.sqrt(k2)) *(r1_x**2 + r2_x**2 + (r2_y+r1_y)**2)/((r1_x - r2_x)**2 + (r2_y + r1_y)**2 + eps))
            #self.Mbx4 = self.Mbx4/2
            #self.Mbx += self.Mbx4

            #self.Mby4 = (self.sym[0]* miu0 * dx*dy)/(2*np.pi*np.sqrt((r1_x + r2_x)**2 + (r2_y + r1_y)**2 ))*\
            #(sps.ellipk(np.sqrt(k2)) + sps.ellipe(np.sqrt(k2))*(r1_x**2 - r2_x**2 -(r2_y + r1_y)**2)/((r1_x - r2_x)**2 + (r2_y + r1_y)**2 + eps))
            #self.Mby4 = self.Mby4/r2_x/2
            #self.Mby += self.Mby4


        del r1_x, r2_x, r1_y, r2_y
        #Qij = cholesky(Qij)
        self.Qij_inv = 1 / miu0 / dx / dy * np.linalg.inv(Qij)
        del Qij
        #print(f'Mbx {self.Mbx}')
        #print(f'diff of Mbx {self.Mbx- self.Mbx2}')
        #print(f'diff of Mby {self.Mby-self.Mby2}') 
        #print(f'Mby2 {self.Mby2}')
        self.M = np.matmul(self.Qij_inv, self.L.transpose())
        self.LM_inv = np.linalg.inv(np.matmul(self.L, self.M))
        tc2= time.perf_counter()
        tc3 = tc2 - tc1
        #self.LM_inv = cholesky(self.LM_inv)
        print(f'Time taken for cylindrical Kernel and Magnetic field solution generation {tc3}')

    #@nb.jit(nopython=False)
    def _build_matrix_cyl_corrected(self):
        """
        Build the matrixes used in Cylindrical coordinate system
        This corrected version calculates the partial derivatives analytically
        instead of using finite difference which is sensitive to parameter tuning and functional space
        Resolving the Kernel Q_ij is also a priority it does not span solution space numerically
        (E_a(\Omega_{j},J) obtained via Euler time integration meanwhile the B is obtained from Biot Savart Law integral 
        :return: None
        """
        miu0 = 4 * np.pi * 1e-7
        dx = self.dx
        dy = self.dy
        r1_x, r2_x = np.meshgrid(self.posi_x, self.posi_x)
        r1_y, r2_y = np.meshgrid(self.posi_y, self.posi_y)
        eps = 0.005 * dx * dy

        #print (f'self.posi_x {self.posi_x}')
        #print(f'r1_x {r1_x}')
        #(f'r2_x {r2_x}')

        # define subfunctions to be called
        # vectorized sub functions
        def calAphi(r_s, r_t, z_s, z_t):
            k2 = 4 * r_s * r_t / ((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps)
            Aphi = (np.sqrt(r_s / r_t) * (sps.ellipk((k2)) * (1 - k2 / 2) - sps.ellipe(k2))) / (np.pi *np.sqrt(k2))
            del k2
            return Aphi
        # Definition of derivatives 
        def kprime(k):
            return sps.ellipe(k)/(k-k**3) - sps.ellipk(k)/k
        
        def eprime(k):
            return sps.ellipk(k)/(k- k**3)

        def calAphidr(r_s, r_t, z_s, z_t):
            dkdr= (4*r_s  - 2* (r_s +r_t))/(((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps))**2
            k2 = 4 * r_s * r_t / ((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps)
            term1 = (1* dkdr * np.sqrt(r_s* r_t) * sps.ellipk(k2))/(np.pi *-2 * (k2)**(3/2))  + \
            (1* np.sqrt(r_s/r_t) * sps.ellipk(k2))/(np.pi * 2* np.sqrt(k2))  + \
            (1* np.sqrt(r_s* r_t) * kprime(k2) * dkdr)/(np.pi * np.sqrt(k2)) 
            term2 =   (1* np.sqrt(r_s *r_t)* sps.ellipk(k2) * dkdr) / (4* np.pi*np.sqrt(k2)) -\
            (1* np.sqrt(r_s/r_t)* sps.ellipk(k2) *np.sqrt(k2)) / (np.pi*4) -\
            (1* np.sqrt(r_s*r_t)* kprime(k2) *dkdr *np.sqrt(k2)) / (np.pi*2) 
            term3 = (1*dkdr* np.sqrt(r_s * r_t) * sps.ellipe(k2)) / (np.pi *2*(k2)**(3/2))  -\
            (1* np.sqrt(r_s/r_t) * sps.ellipe(k2)) / (np.pi*2* np.sqrt(k2))  -\
            (1* np.sqrt(r_s*r_t) * eprime(k2) *dkdr) / (np.pi *np.sqrt(k2)) 
            return term1 + term2 + term3

        def calAphidz(r_s, r_t, z_s, z_t):
            dkdz = ( -2* (z_s - z_t)*(-1))/ (((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps))**2
            k2 = 4 * r_s * r_t / ((r_s + r_t) ** 2 + (z_s - z_t) ** 2 + eps)
            term1 = (1*dkdz * np.sqrt(r_s / r_t) * (sps.ellipk(k2))) / (np.pi * -2* (k2)**(3/2))  +\
            (1* np.sqrt(r_s / r_t) * kprime(k2)* dkdz) / (np.pi * np.sqrt(k2))
            term2 = (1)/ (-np.pi *2* np.sqrt(k2)) * dkdz* np.sqrt(r_s / r_t) * (sps.ellipk(k2)) +\
            (1* np.sqrt(k2)* np.sqrt(r_s / r_t) * (kprime(k2)* dkdz) )/ (-np.pi) 
            term3 = (1* np.sqrt(r_s / r_t) * (sps.ellipe)(k2))/ (-np.pi * -2*(k2)**(3/2))  +\
            (1* np.sqrt(r_s / r_t) * eprime(k2)* dkdz)/ (-np.pi *np.sqrt(k2)) 
            return term1 + term2 + term3
        
        self.L = np.zeros((self.count_turns, self.count_ele))
        for _, i in enumerate(range(self.count_turns)):
            self.L[i, i * self.ny: (i + 1) * self.ny] = np.ones(self.ny)
        Qij = -calAphi(r1_x, r2_x, r1_y, r2_y)
        dis = min(dx, dy) / 5
        self.Mbx = -miu0 * dx * dy * (calAphi(r1_x, r2_x, r1_y, r2_y + dis) - calAphi(r1_x, r2_x, r1_y, r2_y - dis)) / dis / 2
        #self.Mbx = -miu0 * dx * dy * calAphidz(r1_x, r2_x, r1_y, r2_y)
        #self.Mby = miu0 * dx * dy * ((r2_x + dis) * calAphi(r1_x, r2_x + dis, r1_y, r2_y) - (r2_x - dis) * calAphi(r1_x, r2_x - dis, r1_y, r2_y)) / r2_x / dis / 2
        self.Mby = miu0 * dx * dy * calAphidr(r1_x, r2_x, r1_y, r2_y)/r2_x

        
        dic_wym = {1: 'symmetry', -1: 'anti- symmetry'}
        # -r_1y is imposing vertical symmetry as in polar (r)
        if abs(self.sym[0]) == 1:
            Qij += -self.sym[0] * calAphi(r1_x, r2_x, -r1_y, r2_y)
            self.Mbx += - self.sym[0] * miu0 * dx * dy * calAphidz(r1_x, r2_x, -r1_y, r2_y)
            self.Mby += self.sym[0] * miu0 * dx * dy *calAphidr(r1_x, r2_x, -r1_y, r2_y)
        # conserve RAM del objects in funtional scope will loop over and over again
        del r1_x, r2_x, r1_y, r2_y
        self.Qij_inv = 1 / miu0 / dx / dy * np.linalg.inv(Qij)
        del Qij
        # NUMPY matrix inversion
        self.M = np.matmul(self.Qij_inv, self.L.transpose())
        self.LM_inv = np.linalg.inv(np.matmul(self.L, self.M))
       
    #@nb.jit(nopython=False)
    @nb.jit(nopython=False,forceobj=True)
    def plot_resultj(self, timestep_n0, use_sym=False, plot3D=False):
        """
        plot the current density distribution at timestep_no
        :param timestep_n0: time step to show the result, which should be defind in function "set_time"
        :param use_sym: True: plot the symmetrical parts False: only plot calculated part
        :param plot3D: True: show 3D effect  False: 2D plot
        :return: None
        """
        if timestep_n0 >= len(self.save_time):
            print( ': Result at timestep %i is not saved.' % timestep_n0)
            return
        
        #print(f'here is the shape of jt {self.jt.shape}')
        #self.jc = np.array(self.jc)
        #self.jc = np.squeeze(self.jc,axis =2)
        #self.jc=self.jc.reshape((1500,13))
        #self.jc = self.jc[:,0:12]
        #print(f'here is the shape of jc {self.jc.shape}')

        
        j = (self.jt[timestep_n0] / self.jc[timestep_n0]).reshape((self.count_turns, self.ny))
        #j = StandardScaler().fit_transform((self.jt[timestep_n0] ).reshape((self.count_turns, self.ny)))
        print(f'shape of j{j.shape}')
        #print(f' j is {j}')
        x = self.posi_x.reshape((self.count_turns, self.ny))
        y = self.posi_y.reshape((self.count_turns, self.ny))
        f = plt.figure()
        if plot3D:
            ax = f.gca(projection='3d')
            ax.set_zlim([-1.1, 1.1])
            plotme = ax.plot_surface
        else:
            ax = f.add_subplot(111)
            plotme = ax.contourf
        k0 = 0
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'J/Jc @(T={self.cumdta[timestep_n0].round(4)})')
        for i in range(len(self.coils)):
            k1 = k0 + self.coils[i].count_turns
            surf = plotme(x[k0: k1, :], y[k0: k1, :], j[k0: k1, :], cmap=cm.coolwarm)
            k0 = k1
        cb = f.colorbar(surf)
        # cb.ax.set_ylabel('J/Jc')
        # cb.ax.annotate('J/Jc', (0.1,1.05))
        if use_sym:
            if abs(self.sym[0]) == 1:
                k0 = 0
                for _, i in enumerate(range(len(self.coils))):
                    k1 = k0 + self.coils[i].count_turns
                    surf = plotme(x[k0: k1, :], -y[k0: k1, :], self.sym[0] * j[k0: k1, :], cmap=cm.coolwarm, vmin=-1
                                  , vmax=1) 
                    k0 = k1
            if abs(self.sym[1]) == 1:
                k0 = 0
                for _, i in enumerate(range(len(self.coils))):
                    k1 = k0 + self.coils[i].count_turns
                    surf = plotme(-x[k0: k1, :], y[k0: k1, :], self.sym[1] * j[k0: k1, :], cmap=cm.coolwarm, vmin=-1
                                  , vmax=1)
                    k0 = k1
            if abs(self.sym[0] * self.sym[1]) == 1:
                k0 = 0
                for _, i in enumerate(range(len(self.coils))):
                    k1 = k0 + self.coils[i].count_turns
                    surf = plotme(-x[k0: k1, :], -y[k0: k1, :],  self.sym[0] * self.sym[1] * j[k0: k1, :],
                                  cmap=cm.coolwarm, vmin=-1, vmax=1)
                    k0 = k1 

        f.savefig(f'/Users/teacher/Downloads/J%20model_v1/{len(self.coils)}Coils{timestep_n0}CurrentDensityPlot.png')
        
    
    #@nb.jit(nopython=False)
    @nb.jit(nopython=False,forceobj=True)
    def plot_resultb(self, timestep_n0,  use_sym=False):
        """
        plot the normal magnetic field distribution at timestep_no
        :param timestep_n0: time step to show the result, which should be defind in function "set_time"
        :param use_sym: True: plot the symmetrical parts False: only plot calculated part
        :param plot3D: True: show 3D effect  False: 2D plot
        :return: None
        """
        if timestep_n0 >= len(self.save_time):
            print('Result at timestep %i is not saved.' % timestep_n0)
            return
        bx = np.matmul(self.Mbx, self.jt[timestep_n0])
        by = np.matmul(self.Mby, self.jt[timestep_n0])
        b = (np.sqrt(bx**2 + by**2)).reshape((self.count_turns, self.ny))
        x = self.posi_x.reshape((self.count_turns, self.ny))
        y = self.posi_y.reshape((self.count_turns, self.ny))
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'B @(T={self.cumdta[timestep_n0].round(4)})')
        k0 = 0
        for i in range(len(self.coils)):
            k1 = k0 + self.coils[i].count_turns
            surf = ax.contourf(x[k0: k1, :], y[k0: k1, :], b[k0: k1, :], cmap=cm.coolwarm)
            k0 = k1
        cb = f.colorbar(surf)
        if use_sym:
            if abs(self.sym[0]) == 1:
                k0 = 0
                for _, i in enumerate(range(len(self.coils))):
                    k1 = k0 + self.coils[i].count_turns
                    surf = ax.contourf(x[k0: k1, :], -y[k0: k1, :], b[k0: k1, :], cmap=cm.coolwarm)
                    k0 = k1
            if abs(self.sym[1]) == 1:
                k0 = 0
                for _, i in enumerate(range(len(self.coils))):
                    k1 = k0 + self.coils[i].count_turns
                    surf = ax.contourf(-x[k0: k1, :], y[k0: k1, :], b[k0: k1, :], cmap=cm.coolwarm)
                    k0 = k1
            if abs(self.sym[0] * self.sym[1]) == 1:
                k0 = 0
                for _, i in enumerate(range(len(self.coils))):
                    k1 = k0 + self.coils[i].count_turns
                    surf = ax.contourf(-x[k0: k1, :], -y[k0: k1, :], b[k0: k1, :], cmap=cm.coolwarm)
                    k0 = k1
        plt.savefig(f'/Users/teacher/Downloads/J%20model_v1/{len(self.coils)}Coils{timestep_n0}BFieldPlot.png')
        
        


    def post_cal_power(self, timestep0, timestep1, amp=1):
        """
        Calculate the AC loss between timestep0 and timestep1
        :param timestep0: step time to start
        :param timestep1:  step time to end
        :param amp: can set a amp paramter to compare different frequency
        :return: Average AC loss
        """
        time1 = self.save_time_step[timestep0]
        time2 = self.save_time_step[timestep1]
        power_avg = np.zeros(self.count_turns)
        energy = 0
        for _, i in enumerate(range(self.count_turns)):
            power = np.array([j[i] for j in self.power][time1: time2]) * self.dx * self.dy
            dt = np.array(self.dta[time1:time2])
            
            assert(len(power)==len(self.dta[time1:time2]))
            power_avg[i] = (power * dt).sum() / dt.sum()
           # power_avg[i] = (power * dt).sum() / dt.sum() # energy/T_period = power_avg
            #energy += (power*dt).sum()# energy integral         
        f = plt.figure()
        axe = f.add_subplot(111)
        axe.semilogy(np.arange(1, self.count_turns + 1), power_avg*amp)
        axe.set_xlabel('Turn number')
        axe.set_ylabel('Average loss per cycle (W/m)')
        axe.grid(which='both')

        f = plt.figure()
        axe = f.add_subplot(111)
        startnum = 0
        for _, i in enumerate(range(len(self.coils))):
            endnum = startnum + self.coils[i].count_turns
            axe.semilogy(np.arange(1, self.coils[i].count_turns + 1), power_avg[startnum: endnum] * amp)
            startnum = endnum
        axe.set_xlabel('Turn number')
        axe.set_ylabel('Average loss (W/m)')
        axe.grid(which='both')
        #heq = HeatEq()
        #heq.Solve(energy)
        #plt.show()
        #print(time.asctime(time.localtime(time.time())) + ': Total average loss: %5.4f W/m.' % (power_avg.sum()*amp))
        return power_avg

    def save_result(self, filename='./ACloss_YBCO2D.pkl'):
        """
        save the calculated results
        :param filename: filename
        :return: None
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_result(filename):
        """
        load saved result
        :param filename: filename
        :return: (AClossFT2D) loaded results
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
        


tape = YBCOtape()

# using Cartesian coordinate system (stack), and using x and y symmetry
solver = AClossFT2D(tape, csys=0, sym=(1, 1))

# adding a coil (inner diameter, outter diameter, bottom y position, number of turns)


solver.add_coil(150e-5,  275e-4 , 0, 100)
solver.add_coil(150e-5, 275e-4, 0.0044,100)
solver.add_coil(150e-5,275e-4, 0.0044*2, 100)
solver.add_coil(150e-5, 275e-4, 0.0044*3, 100)
solver.add_coil(150e-5, 275e-4, 0.0044*4, 100)

# divide each tape into 15 elements along the y direction equally.
solver.mesh(15) 
tf= 110
save_time = np.linspace(0, 1,tf, endpoint=True)

# minimum and maximum step time to use
timestep_min = 1e-6
timestep_max = 1e-4
solver.set_time(save_time, timestep_min, timestep_max, 1)

# amplitude and frequency of transpose current
# ramping function
# imax 200 freq =1/200
Imax= 11
fre =50
 
# set the transpose current
solver.set_current(Imax, fre)
#solver.set_current(Imax, fre)
#solver.add_coil(150e-6, 150e-6 + 300e-6,0,25)
#Bmax = 40e-3
#fre = 2 
#solver.set_field(Bmax, fre, 0)
 
# solve
solver.run_rosenbrock() 

use_sym = False

# plot the current density distribution at the (9+1)th recorded result, here it is 9/12 s.

#solver.plot_resultj(8, use_sym=use_sym)
times = np.linspace(0,tf,1)
#for t in times:
#    solver.plot_resultj(int(t),use_sym=use_sym)
#    solver.plot_resultb(int(t),use_sym=use_sym)

solver.plot_resultj(5,use_sym=use_sym)
solver.plot_resultj(92,use_sym=use_sym)
solver.plot_resultj(94,use_sym=use_sym)
solver.plot_resultj(96,use_sym=use_sym)
solver.plot_resultj(98,use_sym=use_sym)
solver.plot_resultj(100,use_sym=use_sym)
solver.plot_resultb(90,use_sym=use_sym)
solver.plot_resultb(92,use_sym=use_sym)
solver.plot_resultb(94,use_sym=use_sym)
solver.plot_resultb(96,use_sym=use_sym)
solver.plot_resultb(98,use_sym=use_sym)
solver.plot_resultb(100,use_sym=use_sym)
#solver.plot_resultj(190,use_sym=use_sym)
#power_avg = solver.post_cal_power(1,int(tf/2))
#print(f'power avg {power_avg}')

plt.show()
