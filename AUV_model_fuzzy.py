import numpy as np
import sys
import math

# 定系为以岸边起点为原点的 东北地坐标系，
# 其中，x[0]>0向前（北），x[1]>0向右（东）,x[2]>0向下
# delta[0]垂直舵，控制航向——psi  delta[0]为正时，AUV受相对自身向左的力矩
# delta[2]水平舵，控制俯仰——theta  delta[2]为正时，AUV受相对自身下沉的力矩
# 动系为AUV以自身重心为原点的 前右下坐标系

class AuvModel:
    '''
    Define a AUV dynamic model class
    Input:
    x：State vector in fixed coordinate system，ndarray(6,1)
    u: Velocity vector in motion coordinate system，ndarray(6,1)
    delta: Rudder angle vector
    f: propeller
    '''
    p_water=1070#1000#density水密度
    L=2.38#length AUV长度 米

    m=167#mass AUV质量 kg
    g=9.81# gravity acceleration 重力加速度
    # position of the mass:重心坐标，一般定为运动坐标原点
    xg=0
    yg=0
    zg=0
    # position of the buoyant:浮心坐标，注意z方向与重心不重合
    xb=0
    yb=0
    zb=-0.06 # 浮心在重心之下-0.06

    #Inertia：刚体转动惯量项
    Ix=4.73 #  % 绕x轴的转动惯量
    Iy=70.27# % 绕x轴的转动惯量
    Iz=71.38# % 绕x轴的转动惯量

    # 以下是水动力系数，一大堆，看不懂，不过应该与具体型号的AUV有关，这些数值应该就是作者在实验中得来的
    x_u_=-4.8e-3
    y_r_=-9.67e-4
    y_v_=-3.66E-2
    y_r=-4.8E-3
    y_v=-0.0569
    y_delta_r=- 0.0215  #3.76e-4
    z_q_=-1.86e-4
    z_w_=-6.8e-2
    z_q=-0.0761
    z_w=-0.0830
    z_delta_s=-0.0330 # 5.76e-4
    k_vq=0
    k_wr=0
    m_q_=-4.1e-3
    m_w_=-1.82E-4
    m_q=-0.0207
    m_w=0.0372
    m_delta_s= 0.0213 # 3.72e-4 # 无因次流体动力阻尼系数
    n_r_=-2.4E-3
    n_v_=9.54E-4
    n_r=-9.67E-4
    n_v=-0.0186
    n_delta_r= 0.0099 # 无因次流体动力阻尼系数
    x_uu = -0.0093

    THRUST_MIN_AUV = 0  # 最小推力 
    THRUST_MAX_AUV = 500  # 最大推力
    RUDDER_MAX_AUV = 25*2*np.pi/360  # 舵角最大位置 (角度->弧度)


    def __init__(self,x,u,delta,f,t,random_factor_array):
        self.x = x
        self.u = u
        self.delta = delta
        self.f = f
        self.step = t*10

        self.phi=x[3].item()   #p
        self.theta=x[4].item() #q
        self.psi=x[5].item()  #r
        self.delta_r=delta[0].item()#产生Y，N delta_r--psi，航向，垂直舵
        self.delta_s=delta[2].item()#产生Z，M delta_s--theta，纵倾，水平舵
        

        self.u_=u[0].item() # 速度向量u的第1项，表示运动坐标系 航行（surge）线速度
        self.v=u[1].item()  # 速度向量u的第2项，表示运动坐标系 横向（sway）线速度
        self.w=u[2].item()  # 速度向量u的第3项，表示运动坐标系 生沉（heave）线速度
        self.p=u[3].item()  # 速度向量u的第4项，表示运动坐标系 横摇（roll）角速度
        self.q=u[4].item()  # 速度向量u的第5项，表示运动坐标系 俯仰（pitch）角速度
        self.r=u[5].item() # 速度向量u的第6项，表示运动坐标系 偏航（yaw）角速度
        self.tau = f

        self.random_factor_array = random_factor_array

        # self.p_water = self.p_water * self.random_factor_array[0]#density水密度
        
        if round(self.step) % 20 == 0:
            self.m = self.m * np.random.uniform(0.7, 1.3)
            # print(self.m/167, round(self.step))

        if round(self.step) % 100 == 0:
            exp = round(self.step) / 100
            self.p_water = self.p_water * (0.99**exp)
            # print(self.p_water, self.m, self.random_factor_array, exp, round(self.step))

        # 以下是水动力系数，一大堆，看不懂，不过应该与具体型号的AUV有关，这些数值应该就是作者在实验中得来的
        self.x_u_ = self.x_u_ * self.random_factor_array[1]
        self.y_r_ = self.y_r_ * self.random_factor_array[2]
        self.y_v_ = self.y_v_ * self.random_factor_array[3]
        self.y_r= self.y_r    * self.random_factor_array[4]
        self.y_v=self.y_v     * self.random_factor_array[5]
        self.y_delta_r=self.y_delta_r * self.random_factor_array[6]  
        self.z_q_=self.z_q_ * self.random_factor_array[7]
        self.z_w_=self.z_w_ * self.random_factor_array[8]
        self.z_q=self.z_q   * self.random_factor_array[9]
        self.z_w=self.z_w   * self.random_factor_array[10]
        self.z_delta_s=self.z_delta_s * self.random_factor_array[11] 
        self.k_vq=self.k_vq * self.random_factor_array[12]
        self.k_wr=self.k_wr * self.random_factor_array[13]
        self.m_q_=self.m_q_ * self.random_factor_array[14]
        self.m_w_=self.m_w_ * self.random_factor_array[15]
        self.m_q=self.m_q   * self.random_factor_array[16]
        self.m_w=self.m_w   * self.random_factor_array[17]
        self.m_delta_s=self.m_delta_s * self.random_factor_array[18] 
        self.n_r_=self.n_r_ * self.random_factor_array[19]
        self.n_v_=self.n_v_ * self.random_factor_array[20]
        self.n_r=self.n_r   * self.random_factor_array[21]
        self.n_v=self.n_v   * self.random_factor_array[22]
        self.n_delta_r=self.n_delta_r * self.random_factor_array[23]
        self.x_uu =self.x_uu * self.random_factor_array[24]

    def M(self):
        M = np.array([
            [self.m-0.5*self.p_water*(self.L**3)*self.x_u_, 0, 0, 0, 0, 0],
            [0, self.m-0.5*self.p_water*(self.L**3)*self.y_v_, 0, 0, 0, -0.5*self.p_water*(self.L**4)*self.y_r_],
            [0, 0, self.m-0.5*self.p_water*(self.L**3)*self.z_w_, 0, -0.5*self.p_water*(self.L**4)*self.z_q_, 0],
            [0, 0, 0, self.Ix, 0, 0],
            [0, 0, -0.5*self.p_water*(self.L**4)*self.m_w_, 0, self.Iy-0.5*self.p_water*(self.L**5)*self.m_q_, 0],
            [0, -0.5*self.p_water*(self.L**4)*self.n_v_, 0, 0, 0, self.Iz-0.5*self.p_water*(self.L**5)*self.n_r_],
            ])
            # 惯性矩阵，似乎这个矩阵M已经是
            # 刚体质量矩阵M(RB)和附加质量矩阵M(A)的加和
        return M

    def C(self):
        C = np.array([
            [0.5*self.p_water*(self.L**2)*self.x_uu*abs(self.u_), 0, 0, 0, self.m*self.w, -self.m*self.v],
            [0.5*self.p_water*(self.L**3)*self.y_r*self.r+0.5*self.p_water*(self.L**3)*self.y_v*self.v, 0, 0, -self.m*self.w, 0, self.m*self.u_],
            [0.5*self.p_water*(self.L**3)*self.z_q*self.q+0.5*self.p_water*(self.L**3)*self.z_w*self.w, 0, 0, self.m*self.v, -self.m*self.u_, 0],
            [0, 0.5*self.p_water*(self.L**4)*self.k_vq*self.q, 0.5*self.p_water*(self.L**4)*self.k_wr*self.r, 0, self.Iz*self.r, -self.Iy*self.q],
            [0.5*self.p_water*(self.L**4)*self.m_q*self.q+0.5*self.p_water*(self.L**3)*self.m_w*self.w, 0, 0, -self.Iz*self.r, 0, self.Ix*self.p],
            [0.5*self.p_water*(self.L**4)*self.n_r*self.r+0.5*self.p_water*(self.L**4)*self.n_v*self.v, 0, 0, self.Iy*self.q, -self.Ix*self.p, 0],
            ])
        # 科氏向心力矩阵
        return C

    def G(self):
        G = np.array([
            [0],
            [0],
            [0],
            [-self.m*self.g*(self.zg-self.zb)*np.cos(self.theta)*np.sin(self.phi)],
            [-self.m*self.g*(self.zg-self.zb)*np.sin(self.theta)],
            [0]
            ])
        return G
        # gravity重力与浮力的力与力矩矩阵。此仿真中假设了重力与浮力抵消，
        # 重心坐标[xg,yg,zg]、浮心坐标[xb,yb,zb]在水平面重合，在垂直方向不重合，本仿真中浮心在重心下0.06m     
        #tau=f#thruster推进器
    #以下的矩阵不是很明确，虽然rudder的翻译是舵，但看起来像是流体阻尼矩阵，也估计是一个意思，流体阻尼在舵面的作用下为AUV提供了操纵
    def Rudder(self):
        Rudder = np.array([
            [0], 
            [0.5*self.p_water*(self.L**2)*self.y_delta_r*(self.u_**2)*self.delta_r],
            [0.5*self.p_water*(self.L**2)*self.z_delta_s*(self.u_**2)*self.delta_s],
            [0],
            [0.5*self.p_water*(self.L**3)*self.m_delta_s*(self.u_**2)*self.delta_s],
            [0.5*self.p_water*(self.L**3)*self.n_delta_r*(self.u_**2)*self.delta_r]
            ])
        return Rudder

    def get_du(self,x,u,delta,f):
        self.x = x
        self.u = u
        self.delta = delta
        self.f = f

        self.phi=x[3].item()   #p
        self.theta=x[4].item() #q
        self.psi=x[5].item()  #r
        self.delta_r=delta[0].item()#产生Y，N delta_r--psi，航向，垂直舵
        self.delta_s=delta[2].item()#产生Z，M delta_s--theta，纵倾，水平舵
        

        self.u_=u[0].item() # 速度向量u的第1项，表示运动坐标系 航行（surge）线速度
        self.v=u[1].item()  # 速度向量u的第2项，表示运动坐标系 横向（sway）线速度
        self.w=u[2].item()  # 速度向量u的第3项，表示运动坐标系 生沉（heave）线速度
        self.p=u[3].item()  # 速度向量u的第4项，表示运动坐标系 横摇（roll）角速度
        self.q=u[4].item()  # 速度向量u的第5项，表示运动坐标系 俯仰（pitch）角速度
        self.r=u[5].item() # 速度向量u的第6项，表示运动坐标系 偏航（yaw）角速度

        M = self.M() 
        C = self.C() 
        Rudder = self.Rudder() 
        G = self.G()
        du = np.matmul( np.linalg.inv(M) , np.matmul(C,u) + Rudder + f + G )
        return du
  
    def radlimit(self,rad): # this function is used to limit a input rad in [-pi,pi]
        count = 0
        un_converge = False
        while True:         
            if rad > np.pi:                
                rad = rad - 2*np.pi
                count += 1 
                if count >= 1e3:                 
                    #print("The rad doesn't converge")                 
                    #sys.exit()   
                    un_converge = True  
                    break                              
            elif rad < -np.pi:
                rad = rad +2*np.pi
                count += 1 
                if count >= 1e3:                                      
                    #print("The rad doesn't converge")                                      
                    #sys.exit()
                    un_converge = True 
                    break
            else:
                break
        return rad, un_converge
        

class CoordinateTrans:
    def __init__(self,x,u):
        self.x = x
        self.u = u
        self.phi=x[3].item()   #p
        self.theta=x[4].item() #q
        self.psi=x[5].item()  #r

    def mov_to_fix(self,x,u): #   

        self.x = x
        self.u = u
        self.phi = self.x[3].item()   
        self.theta = self.x[4].item() 
        self.psi = self.x[5].item()  

        inv_T = np.array([             
            [np.cos(self.psi)*np.cos(self.theta),np.cos(self.psi)*np.sin(self.theta)*np.sin(self.phi)-np.sin(self.psi)*np.cos(self.phi),np.cos(self.psi)*np.sin(self.theta)*np.cos(self.phi)+np.sin(self.psi)*np.sin(self.phi),0,0,0],             
            [np.sin(self.psi)*np.cos(self.theta),np.sin(self.psi)*np.sin(self.theta)*np.sin(self.phi)+np.cos(self.psi)*np.cos(self.phi),np.sin(self.psi)*np.sin(self.theta)*np.cos(self.phi)-np.cos(self.psi)*np.sin(self.phi),0,0,0], 
            [-np.sin(self.theta),np.cos(self.theta)*np.sin(self.phi),np.cos(self.phi)/np.cos(self.theta),0,0,0],             
            [0,0,0,1,np.sin(self.phi)*np.tan(self.theta),np.cos(self.phi)*np.tan(self.theta)],             
            [0,0,0,0,np.cos(self.phi),-np.sin(self.phi)],             
            [0,0,0,0,np.sin(self.phi)/np.cos(self.theta),np.cos(self.phi)/np.cos(self.theta)] # np无sec()方法，所以*sec()全部被替换为/cos()
            ])         
        return np.matmul(inv_T,self.u)     


'''
u = np.zeros((6,1))
#x = np.zeros((6,1))  
x = np.array([20,0,0,0,0,0]).reshape(6,1)   
delta = np.zeros((3,1)) # 初始舵角      
#f = np.zeros((6,1)) # 初始推进器
f = np.array([20,0,0,0,0,0]).reshape(6,1) 

Auv = AuvModel(x,u,delta,f)
print(Auv.get_du(x,u,delta,f))
print(Auv.coordinate_trans_mov_to_fix(x,u))
'''
class Obstacle:
    def __init__(self):
        # 设计三根管道放置于AUV上浮路径上
        self.pipe_radius = 2
        self.pipe_position = [5,15,25] # 管道
        self.pipe_depth = 10 # 管道深度

        self.flotage_radius = 2
        self.flotage_position = [(10,10,5),(20,20,5),(20,20,15),(15,15,15)]

    def CreatePipeObstacle(self):
        pass
    def CreateFlotageObstacle(self):
        pass