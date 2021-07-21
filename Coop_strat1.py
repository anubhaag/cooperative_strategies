def P_sol(HRES,t,Nps,F_loss,n_MPPT):

    import numpy as np
    import pandas as pd
    from scipy import interpolate
    #import matplotlib.pyplot as plt
    
    df = pd.read_excel (r'D:\MTP\multi-hres_optimization\Varodara.xlsx')
    x = df['Time(IST)'].to_list()
    y = df['G'+HRES].to_list()
    tck = interpolate.splrep(x,y)
    #t = np.linspace(1,42,int(41/Ts + 1))
    G = interpolate.splev(t, tck) #W/m2
    for i in range(t.size):
        if (G[i] < 1.0):
            G[i] = 0
    #plt.plot(x, y, 'o', t, G)

    #Pmp_ref = 49;
    #Imp_ref = 2.88;
    #Vmp_ref = 17;
    Isc_ref = 3.11;
    Voc = 21.8;
    a_ref = 1.2;
    G_ref = 1000;
    Iph_ref = Isc_ref;
    Io_ref = Isc_ref*np.exp(-Voc/a_ref)
    Rp_ref = 310;
    Rs_ref = 0.45;

    dT = t.size
    Iph = np.multiply((Iph_ref/G_ref)*np.ones(dT),G) #Assume the cell temp.= ref_cell_temp. 25C
    Io = Io_ref
    a = a_ref
    Rs = Rs_ref #assumption
    Rp = np.multiply((Rp_ref*G_ref)*np.ones(dT), 1/G)
    P_solar = np.zeros(dT)
    I_solar = np.zeros(dT)
    V_solar = np.zeros(dT)

    from gekko import GEKKO

    for i in range(dT):
        if G[i] > 1.0:
            m = GEKKO(remote = False)
            
            I = m.MV(lb = 0)
            I.STATUS = 1
            
            V = m.MV(lb = 0)
            V.STATUS = 1
            
            P = m.Var()
            
            m.Equation(P == I*V)
            m.Equation(I == Iph[i] - Io*(m.exp((V+I*Rs)/a)  -  1) - (V + I*Rs)/Rp[i])
            m.Obj(-P)
            
            m.options.IMODE = 3
            m.options.SOLVER = 1
            m.options.MAX_ITER = 500
            m.options.OTOL = 0.01#tolerance
            #m.options.RTOL = 0.01
            m.options.NODES = 2
            m.solve(disp = False)
        
            #print(P.value)
    
            [P_solar[i]] = P
            [I_solar[i]] = I
            [V_solar[i]] = V
            del m
        
    P_solar = n_MPPT*F_loss*Nps*P_solar
    #plt.plot(t, P_solar)
    
    return P_solar

def P_wind(HRES,t, N, h):
    import numpy as np
    import pandas as pd
    from scipy import interpolate
    #import matplotlib.pyplot as plt
    a = 0.34 #hellman coef.
    rho = 1.29 #kg/m3
    Cp = 0.4
    D = 4 #m rotor diameter
    Awg = (3.14*D**2)/4 #m2
    Pwgr = 1000 #W
    Vc = 3.5 #m/s
    Vr = 14.0
    Vf = 20
    
    df = pd.read_excel (r'D:\MTP\multi-hres_optimization\Varodara.xlsx')
    x = df['Time(IST)'].to_list()
    y = df['W'+HRES].to_list()
    tck = interpolate.splrep(x,y)
    #t = np.linspace(1,42,int(41/Ts + 1))
    v = interpolate.splev(t, tck) #m/s
    #plt.figure()
    #plt.plot(x,y,'o',t,v)
    v = v*(h/10)**a
    
    P_w = np.zeros(t.size)
    for i in range(t.size):
        if (v[i] < Vr) and (v[i] >= Vc):
            P_w[i] = 0.5*Cp*rho*Awg*v[i]**3
        elif (v[i] < Vf) and (v[i] >= Vr):
            P_w[i] = Pwgr
    #plt.figure()
    #plt.plot(t,P_w)
    P_w = N*P_w
    return P_w
################# Optimization for a single HRES###############################

def Single_HRES(tim,HRES='1',Tr=6,T=24,g_buy=7.5,g_sell=3.75,P_dgr=10,Nps=300,Nw=8,Nh=20,eta_c=120,V_cl2_max=800,V_cl2_min=300,N_fc_s=4,n_fc_max_in=0,n_fc_in=0):
    
    import numpy as np
    import pandas as pd
    from scipy import interpolate
    #import matplotlib.pyplot as plt
    from gekko import GEKKO
    
    m = GEKKO(remote = False)
    C_buy = m.Const(g_buy)  #INR/kwh  buying from the grid
    C_sell = m.Const(g_sell) #INR/Kwh as per net metering  selling to the grid
    
    ### Diesel generator  
    #P_dgr = 10#maximum rated power of generator KW/hr
    A_dg = 0.246 #litre/Kwh
    B_dg = 0.08145 #litre/Kwh
    C_fuel = 65 #Rs/litre

    ###  Solar Panels ##
    #Nps = 300       #no. of cell in parallel  no. of cell in series
    F_loss = 0.95  #loss due to dust
    n_MPPT = 0.95 # MPPT device efficiency
    C_PV = 3.6 # INR/kwh
    P_sol_max = P_sol(HRES,tim,Nps,F_loss,n_MPPT)/1000 #kw/hr
    
    ###  Wind Power  ####
    #Nw = 8 #no. of wind turbines
    ht = 20 #height of the turbine (metres)
    C_wind = 3.0 #INR/kW
    P_wind_max = P_wind(HRES,tim,Nw, ht)/1000  #kw/hr

    ###   Demand function  ####
    #Nh = 20 #no. of househiolds
    df = pd.read_excel (r'D:\MTP\multi-hres_optimization\Gujarat_demand.xlsx')
    x = df['Time(hrs)'].to_list()
    y = df['Load(kwh)'].to_list()
    tck = interpolate.splrep(x,y)
    L = interpolate.splev(tim, tck) #kwh for a single household
    P_l = L*Nh

    ####   Electrolyzer(PHOEBUS)/cell   #######
    #eta_c = 120 #no. of cells in series power range (400 - 600W)
    eta_f = 0.95
    z = 2
    F = 98485 #Cmol-1
    a_el = 3600*eta_f/(z*F)
    Tc = 40 #C  35<= T <=90
    U_rev = 1.184 #V
    A = 0.25 #m^2 electrode area
    ######
    r1 = -3.0071e-004 #omega m^2
    r2 =  1.1211e-005 #omega m^2 C-1
    r = r1 + r2*Tc #omega m^2
    #####
    s1 = 0.2772353 #V
    s2 = -0.0024638 #VC-1
    s = s1 + s2*Tc #V
    t1 = -1.4421e+000 #A-1m^2
    t2 = 5.6183e-002 #A-1m2C
    t3 = -3.9527e-004 #A-1m2C2
    t = t1 + t2*Tc + t3*(Tc**2) #A-1m^2
    ##
    Vol_cl2_max = V_cl2_max/T #(litres) maximum oxygen can stored in a day at Temp = 25C and P=24bar
    Vol_cl2_min = V_cl2_min/T
    n_cl2_max = (24*Vol_cl2_max/(0.082*298))*np.ones(Tr)
    n_cl2_min = (24*Vol_cl2_min/(0.082*298))*np.ones(Tr)
    #C_elec = 4.0 #INR/kW
    #C_cl2 = 1.4 #INR/mol 20 INR/kg
    
    ####   Fuel cell Stack model ##########(200W - 400W)
    #N_fc_s = 4 # no. of fuel cell stacks
    eta_fc = 0.7
    N_cell = 35 #35 cell in series
    a_fc = 3600*eta_fc*N_cell/(z*F)
    U_fc0 = 33.18 #V
    e1 = -0.013 #VC-1
    e2 = -1.57 #V
    I_fc0 = 8.798 #A
    R_fc = -2.04 #ohmC-1
    T_fc = 25 #C
    C_h2 = 75*14 #INR/kg 14 dollars per kg
    C_h2 = C_h2/500; #INR/mol
    C_fc = 3.5 #INR/kW
    ####
         
    P_fc = [m.Var(lb = 0) for i in range(Tr)]
    
    P_grid1 = [m.Var() for i in range(Tr)]
    
    P_grid2 = [m.Var(lb = 0) for i in range(Tr)]

    P_cr1 = [m.Var(lb = 0) for i in range(Tr)]
        
    P_dg1 = [m.Var(lb = 0) for i in range(Tr)]
        
    P_dg2 = [m.Var(lb = 0) for i in range(Tr)]
        
    n_fc2 = [m.Var(lb = 0) for i in range(Tr)]
    
    P_cr2 = [m.Var(0, lb = 0) for i in range(Tr)]
    n_h2 = [m.Var(0, lb = 0) for i in range(Tr)]
    n_fc = [m.Var(0, lb = 0) for i in range(Tr)]
    n_fc_max = [m.Var(0, lb = 0) for i in range(Tr)]
    
        
    P_dg = [None]*Tr
    P_grid = [None]*Tr
    P_fcin = [None]*Tr
    for i in range(Tr):
        P_dg[i] = m.Intermediate(P_dg1[i] + P_dg2[i])
        P_grid[i] = m.Intermediate(P_grid1[i] + P_grid2[i])
        P_fcin[i] = m.Intermediate(P_grid2[i] + P_dg2[i] + P_cr2[i])
    

    m.Equations([P_cr2[i] == - P_cr1[i] + P_sol_max[i] + P_wind_max[i] for i in range(Tr)]) 
    m.Equations([1000*(P_cr2[i]+P_dg2[i]+P_grid2[i])/eta_c ==(n_h2[i]/a_el)*U_rev+ (r/A)*((n_h2[i]/a_el)**2) +(n_h2[i]/a_el)*s*m.log((t/A)*(n_h2[i]/a_el) + 1) for i in range(Tr)])
    m.Equations([P_l[i] == P_fc[i] + P_cr1[i] + P_dg1[i] + P_grid1[i] for i in range(Tr)])
    m.Equations([P_dgr >= P_dg1[i] + P_dg2[i] for i in range(Tr)])
    m.Equation(n_fc_max[0] == n_fc_max_in + eta_c*n_h2[0] -N_fc_s*n_fc_in)
    m.Equations([n_fc_max[i] == n_fc_max[i-1] +eta_c*n_h2[i] - N_fc_s*n_fc[i-1] for i in range(1,Tr)])
    m.Equations([1000*P_fc[i]/N_fc_s == ((n_fc[i]+n_fc2[i])/a_fc)*U_fc0 + ((n_fc[i]+n_fc2[i])/a_fc)*e1*T_fc + ((n_fc[i]+n_fc2[i])/a_fc)*e2*m.log( ((n_fc[i]+n_fc2[i])/a_fc)*(1/I_fc0) + 1) + (R_fc/T_fc)*(((n_fc[i]+n_fc2[i])/a_fc)**2) for i in range(Tr)])
    m.Equations([n_fc[i]*N_fc_s <= n_fc_max[i] for i in range(Tr)])
    m.Equations([n_h2[i]*eta_c <= n_cl2_max[i]   for i in range(Tr)])
    m.Equations([n_h2[i]*eta_c >= n_cl2_min[i] for i in range(Tr)])
    
    C_grid = [None]*Tr
    for i in range(Tr):
        C_grid[i] = m.if2(P_grid1[i]+P_grid2[i],C_sell,C_buy) #Rs/Kwh
     
    cost = [None]*Tr
    for i in range(Tr):
        cost[i] = m.Intermediate((C_fuel*(A_dg*P_dg[i]) + C_PV*P_sol_max[i] +C_wind*P_wind_max[i]+ C_fc*P_fc[i])/(P_sol_max[i] + P_wind_max[i] + P_fc[i] + P_dg[i]) )
    cost_grid1 = np.multiply(C_grid,P_grid1)
    cost_grid2 = np.multiply(C_grid,P_grid2)
    
    m.Obj(np.sum(cost_grid1)+np.sum(cost_grid2)+C_fuel*A_dg*np.sum(P_dg1)+C_fuel*A_dg*np.sum(P_dg2)+C_h2*N_fc_s*np.sum(n_fc2) )    
    
    m.options.IMODE = 3 #control
    m.options.SOLVER = 3#possible values 1,2,3 #3. IPOPT - int3rior point, #1. APOPT
    m.options.MAX_ITER = 1000
    m.options.OTOL = 0.001#tolerance
        
    m.solve(disp = True)
    del m
    
    return P_grid,P_grid1,P_grid2,P_cr1,P_cr2, P_fc, P_dg1, P_dg2, P_dg, n_h2, n_fc, n_fc2, n_fc_max, P_l, P_sol_max, P_wind_max, P_fcin, C_grid, cost

 
def Multi_HRES(x,y,P_mgs,bid,ask,b_grid,s_grid):
    import numpy as np
    
    s = []
    b = []
    Qs = []
    Qs_ind = []
    Qb = []
    Qb_ind = []
    for i in range(P_mgs.size):
        if P_mgs[i] < 0:
            s.append(ask[i])
            Qs.append(-1*P_mgs[i])
            Qs_ind.append(i)
        elif P_mgs[i] > 0:
            Qb.append(P_mgs[i])
            Qb_ind.append(i)
            b.append(bid[i])
            
    Ss = len(Qs)
    Sb = len(Qb)
    s = np.array(s)
    b = np.array(b)
    
    from gekko import GEKKO
    m = GEKKO(remote = False)
    U1 = 4.2 #kV
    U0 = 11 #V
    beta = 0.02 #transformer loss
    rho = 0.2 #ohm/km
    
    L = [[m.MV(lb = 0) for j in range(Sb)] for i in range(Ss)]
    Ls = [m.Var(lb=0) for i in range(Ss)]
    Lb = [m.Var(lb=0) for i in range(Sb)]
    Q = [[m.Var(lb = 0) for j in range(Sb)] for i in range(Ss)]
    Q_s = [m.Var(lb=0) for i in range(Ss)]
    Q_b = [m.Var(lb=0) for i in range(Sb)]
    
    R = np.zeros([Ss,Sb])
    w = np.zeros([Ss,Sb])
    Rs = np.zeros(Ss)
    Rb = np.zeros(Sb)
    for i in range(Ss):
        Rs[i] = rho*(np.sqrt((x[Qs_ind[i]] - 0)**2 + (y[Qs_ind[i]] - 0)**2))
        Q_s[i].value = 1000*((1-beta)*(U0**2))/(2*Rs[i])  #kw
        #Ls[i].STATUS = 1
        for j in range(Sb):
            R[i][j] = rho*(np.sqrt((x[Qs_ind[i]] - x[Qb_ind[j]])**2 + (y[Qs_ind[i]] - y[Qb_ind[j]])**2))
            Q[i][j].value = 1000*(U1**2)/(2*R[i][j])  #kw'
            w[i][j] = (s[i]+b[j])/2
            L[i][j].STATUS = 1
            
    for j in range(Sb):
        Rb[j] = rho*(np.sqrt((x[Qb_ind[j]] - 0)**2 + (y[Qb_ind[j]] - 0)**2))
        Q_b[j].value = 1000*((1-beta)*(U0**2))/(2*Rb[j])  #kw  
        #Lb[j].STATUS = 1
          
     ### in terms of power ###
    Loss_sb = 0.001*np.multiply(np.multiply(L,L),R)/(U1**2)  #loss due to local trading
    Loss_s1 =  0.001*np.multiply(np.multiply(Ls,Ls),Rs)/(U0**2) #transmission loss due to selling to MS
    Loss_s2 = np.multiply(beta*np.ones(Ss),Ls) #transformer loss due to selling to MS
    Loss_b1 =  0.001*np.multiply(np.multiply(Lb,Lb),Rb)/(U0**2) #transmission loss due to buying from MS
    Loss_b2 = np.multiply(beta*np.ones(Sb),Q_b) ##transformer loss due to buying from MS
    
    ### in terms of money ####
    Loss_sbp = 0.001*np.multiply(np.multiply(np.multiply(L,L),R),w)/(U1**2)  #loss due to local trading
    Loss_s1p =  s_grid*0.001*np.multiply(np.multiply(Ls,Ls),Rs)/(U0**2) #transmission loss due to selling to MS
    Loss_s2p = s_grid*np.multiply(beta*np.ones(Ss),Ls) #transformer loss due to selling to MS
    Loss_b1p =  b_grid*0.001*np.multiply(np.multiply(Lb,Lb),Rb)/(U0**2) #transmission loss due to buying from MS
    Loss_b2p = b_grid*np.multiply(beta*np.ones(Sb),Q_b) ##transformer loss due to buying from MS
    
    ub11 = b*Q                             #buyer's utility due to local trading
    us12 = L*s[:, np.newaxis]              #seller's utility due to local trading
    ub0 = np.multiply((b - b_grid*np.ones(Sb)),Q_b)  #buyer's utility due to buying from MS
    us0 = np.multiply(s_grid*np.ones(Ss),Q_s) - np.multiply(s,Ls) #seller's utility due to selling to MS
   
    m.Equations([Qs[i] == np.sum(L,axis =1)[i] + Ls[i] for i in range(Ss)])
    m.Equations([Qb[j]  == np.sum(Q,axis =0)[j] + Q_b[j] for j in range(Sb)])
    m.Equations([[Q[i][j] == L[i][j] - 0.001*(L[i][j]**2)*R[i][j]/(U1**2) for j in range(Sb)] for i in range(Ss)])
    m.Equations([Q_b[j] == Lb[j] - 0.001*(Lb[j]**2)*Rb[j]/(U0**2) - beta*Q_b[j] for j in range(Sb)])
    m.Equations([Q_s[i] == Ls[i] - 0.001*(Ls[i]**2)*Rs[i]/(U0**2) - beta*Ls[i] for i in range(Ss)])
    
    P_loss = m.Intermediate(np.sum(Loss_sb)+np.sum(Loss_s1)+np.sum(Loss_s2)+np.sum(Loss_b1) + np.sum(Loss_b2))
    Total_p = m.Intermediate(np.sum(ub11) - np.sum(us12) + np.sum(ub0)+ np.sum(us0) - (np.sum(Loss_sbp)+np.sum(Loss_s1p)+np.sum(Loss_s2p)+np.sum(Loss_b1p) + np.sum(Loss_b2p) ))
    TMV = m.Intermediate(np.sum(ub11) - np.sum(us12) + np.sum(ub0)+ np.sum(us0))

    m.Obj(np.sum(Loss_sb)+np.sum(Loss_s1)+np.sum(Loss_s2)+np.sum(Loss_b1) + np.sum(Loss_b2) )
    
    m.options.IMODE = 3 #control
    m.options.SOLVER = 1#possible values 1,2,3 #3. IPOPT - int3rior point, #1. APOPT
    m.options.MAX_ITER = 1000
    m.options.OTOL = 0.001#tolerance
    
    m.solve(disp=True)
    P_ds = np.zeros([P_mgs.size,P_mgs.size])
    P_ms = np.zeros(P_mgs.size)
    for i in range(Ss):
        [P_ms[Qs_ind[i]]] = Ls[i].value
        P_ms[Qs_ind[i]] = -1*P_ms[Qs_ind[i]]
        for j in range(Sb):
            [P_ds[Qs_ind[i]][Qb_ind[j]]] = L[i][j].value
            P_ds[Qs_ind[i]][Qb_ind[j]] = -1*P_ds[Qs_ind[i]][Qb_ind[j]]
    for i in range(Sb):
        [P_ms[Qb_ind[i]]] = Q_b[i].value
        for j in range(Ss):
            [P_ds[Qb_ind[i]][Qs_ind[j]]] = Q[j][i].value        
    
    del m
    return P_ds, P_ms, P_loss.value, TMV.value, Total_p.value

def Multi_HRES2(x,y,P_mgs,bid,ask,b_grid,s_grid):
    import numpy as np
    
    s=[]
    b=[]
    Qs = []
    Qs_ind = []
    Qb = []
    Qb_ind = []
    for i in range(P_mgs.size):
        if P_mgs[i] < 0:
            s.append(ask[i])
            Qs.append(-1*P_mgs[i])
            Qs_ind.append(i)
        elif P_mgs[i] > 0:
            b.append(bid[i])
            Qb.append(P_mgs[i])
            Qb_ind.append(i)
    Ss = len(Qs)
    Sb = len(Qb)
    
    from gekko import GEKKO
    m = GEKKO(remote = False)
    U0 = 11 #kV
    beta = 0.02 #transformer loss
    rho = 0.2 #ohm/km
    
    Rs = np.zeros(Ss)
    Rb = np.zeros(Sb)
    for i in range(Ss):
        Rs[i] = rho*(np.sqrt((x[Qs_ind[i]] - 0)**2 + (y[Qs_ind[i]] - 0)**2))
            
    for j in range(Sb):
        Rb[j] = rho*(np.sqrt((x[Qb_ind[j]] - 0)**2 + (y[Qb_ind[j]] - 0)**2))
    
    if (Ss > 0) and (Sb>0):    
        Ls = [m.Var(lb=0) for i in range(Ss)]
        Lb = [m.Var(lb=0) for j in range(Sb)]
        
        ### in terms of power loss ####
        Loss_s1 =  0.001*np.multiply(np.multiply(Qs,Qs),Rs)/(U0**2) #transmission loss due to selling to MS
        Loss_s2 = np.multiply(beta*np.ones(Ss),Qs) #transformer loss due to selling to MS
        Loss_b1 =  0.001**np.multiply(np.multiply(Lb,Lb),Rb)/(U0**2) #transmission loss due to buying from MS
        Loss_b2 = np.multiply(beta*np.ones(Sb),Qb) ##transformer loss due to buying from MS
        
        ### in terms of money ####
        Loss_s1p =  0.001*s_grid*np.multiply(np.multiply(Qs,Qs),Rs)/(U0**2) #transmission loss due to selling to MS
        Loss_s2p = s_grid*np.multiply(beta*np.ones(Ss),Qs) #transformer loss due to selling to MS
        Loss_b1p =  0.001*b_grid*np.multiply(np.multiply(Lb,Lb),Rb)/(U0**2) #transmission loss due to buying from MS
        Loss_b2p = b_grid*np.multiply(beta*np.ones(Sb),Qb) ##transformer loss due to buying from MS
        ub0 = np.multiply((b - b_grid*np.ones(Sb)),Qb)  #buyer's utility due to buying from MS
        us0 = np.multiply(s_grid*np.ones(Ss),Ls) - np.multiply(s,Qs) #seller's utility due to selling to MS
    
        m.Equations([Qb[j] == Lb[j] - 0.001*(Lb[j]**2)*Rb[j]/(U0**2) - beta*Qb[j] for j in range(Sb)])
        m.Equations([Ls[i] == Qs[i] - 0.001*(Qs[i]**2)*Rs[i]/(U0**2) - beta*Qs[i] for i in range(Ss)])
    
        P_loss = m.Intermediate(np.sum(Loss_s1)+np.sum(Loss_s2)+np.sum(Loss_b1)+np.sum(Loss_b2))
        Total_p = m.Intermediate(np.sum(us0)+np.sum(ub0)-(np.sum(Loss_s1p)+np.sum(Loss_s2p)+np.sum(Loss_b1p)+np.sum(Loss_b2p)) ) 
        TMV = m.Intermediate(np.sum(us0)+np.sum(ub0))
        
        m.options.IMODE = 1
        m.options.SOLVER=3
        m.solve(disp=True)
        del m
        return P_loss.value, TMV.value, Total_p.value
    elif (Ss>0) and(Sb==0):
        Ls = np.zeros(Ss)
        for i in range(Ss):
            Ls[i] == Qs[i] - 0.001*(Qs[i]**2)*Rs[i]/(U0**2) - beta*Qs[i]
        
        ### in terms of power
        Loss_s1 = 0.001*np.multiply(np.multiply(Qs,Qs),Rs)/(U0**2) #transmission loss due to selling to MS
        Loss_s2 = np.multiply(beta*np.ones(Ss),Qs) #transformer loss due to selling to MS
        P_loss = np.sum(Loss_s1)+np.sum(Loss_s2)
        
        ### in terms of money
        Loss_s1p =  0.001*s_grid*np.multiply(np.multiply(Qs,Qs),Rs)/(U0**2) #transmission loss due to selling to MS
        Loss_s2p = s_grid*np.multiply(beta*np.ones(Ss),Qs) #transformer loss due to selling to MS
        us0 = np.multiply(s_grid*np.ones(Ss),Ls) - np.multiply(s,Qs) #seller's utility due to selling to MS
       
        Total_p = np.sum(us0)-np.sum(Loss_s1p)-np.sum(Loss_s2p)
        TMV = np.sum(us0)
        
        return [P_loss], [TMV], [Total_p] 
    
    elif (Sb>0) and(Ss==0):
        Lb = [m.Var(lb=0) for j in range(Sb)]
        
        ### in terms of power 
        Loss_b1 =  0.001*np.multiply(np.multiply(Lb,Lb),Rb)/(U0**2) #transmission loss due to buying from MS
        Loss_b2 = np.multiply(beta*np.ones(Sb),Qb) ##transformer loss due to buying from MS
        
        ### in terms of money ####
        Loss_b1p =  0.001*b_grid*np.multiply(np.multiply(Lb,Lb),Rb)/(U0**2) #transmission loss due to buying from MS
        Loss_b2p = b_grid*np.multiply(beta*np.ones(Sb),Qb) ##transformer loss due to buying from MS
        ub0 = np.multiply((b - b_grid*np.ones(Sb)),Qb)  #buyer's utility due to buying from MS
        
        m.Equations([Qb[j] == Lb[j] - 0.001*(Lb[j]**2)*Rb[j]/(U0**2) - beta*Qb[j] for j in range(Sb)])
        P_loss = m.Intermediate(np.sum(Loss_b1)+np.sum(Loss_b2))
        Total_p = m.Intermediate(np.sum(ub0)-np.sum(Loss_b1p)-np.sum(Loss_b2p))
        TMV = m.Intermediate(np.sum(ub0))
        
        m.options.IMODE = 1
        m.options.SOLVER=3
        m.solve(disp=True)
        return P_loss.value, TMV.value, Total_p.value
    else :
        return [0.0],[0.0],[0.0]
        
     
import numpy as np
import random
import matplotlib.pyplot as plt

Ts = 1
T = 24 ## Total time period(in hrs) 01:00 - 24:00
Tr = 6
Ng = 5 #total no. of grid

x = np.zeros(Ng)
y = np.zeros(Ng)
HRES_n = np.zeros(Ng)
x_range = 150 #in kms
y_range = 150 #in kms
for i in range(Ng):
    random.seed(i)
    x[i] = random.uniform(x_range/2-x_range, x_range-x_range/2)
    random.seed(i+Ng)
    y[i] = random.uniform(y_range/2-y_range, y_range-y_range/2)
    HRES_n[i] = i+1

plt.figure()    
plt.plot(x,y,'b*')
plt.title('HRES location')
i=0
for x1, y1 in zip(x, y):
    plt.text(x1, y1, str(HRES_n[i]), color="red", fontsize=10)
    i=i+1

n_fc_i = np.zeros(Ng)
n_fc_max_i = np.zeros(Ng)

P_cr2T = np.zeros([Ng,T+1])
P_fcT = np.zeros([Ng,T+1])
P_cr1T = np.zeros([Ng,T+1])
P_dg1T = np.zeros([Ng,T+1])
P_dg2T = np.zeros([Ng,T+1])
P_dgT = np.zeros([Ng,T+1])
n_h2T = np.zeros([Ng,T+1])
n_fcT = np.zeros([Ng,T+1])
n_fc2T = np.zeros([Ng,T+1])
n_fc_maxT  =np.zeros([Ng,T+1])
P_gridT = np.zeros([Ng,T+1])
P_grid1T = np.zeros([Ng,T+1])
P_grid2T = np.zeros([Ng,T+1])
C_gridT = np.zeros([Ng,T+1])
P_lT    =  np.zeros([Ng,T+1])
P_sol_maxT =  np.zeros([Ng,T+1])
P_wind_maxT = np.zeros([Ng,T+1])
P_fcinT = np.zeros([Ng,T+1])
costT = np.zeros([Ng,T+1])

P_mgs = np.zeros(Ng)

P_buy = np.zeros([Ng,T+1])
P_sell = np.zeros([Ng,T+1])
b_grid = 7.5               # INR buying from the grid
s_grid = 3.75               # INR selling to the grid
P_ds = np.zeros([T+1,Ng,Ng])
P_ds_buy = np.zeros([T+1,Ng,Ng])
P_ds_sell = np.zeros([T+1,Ng,Ng])
P_ms = np.zeros([T+1,Ng])
P_ms_buy = np.zeros([T+1,Ng])
P_ms_sell = np.zeros([T+1,Ng])

P_loss = np.zeros(T+1)
P_loss_un = np.zeros(T+1)
Total_c = np.zeros(T+1)
Total_c_un = np.zeros(T+1)
TMV = np.zeros(T+1)
TMV_un = np.zeros(T+1)

P_loss[:] = np.nan
Total_c[:] = np.nan
TMV[:] = np.nan


o_charg = 0.25 #INR/Kwh overhead charges
ask = np.array([3.56, 3.42, 3.69, 3.06, 3.29]) + o_charg*np.ones(Ng)
spread = 1.5
bid = ask + spread*np.ones(Ng)
for h in range(1,T+1):
    print(h)
    tim= np.linspace(h,Tr+h-1,Tr)
    
    ####  HRES 1 ############
    P_grid,P_grid1,P_grid2,P_cr1,P_cr2, P_fc, P_dg1, P_dg2, P_dg, n_h2, n_fc, n_fc2, n_fc_max, P_l, P_sol_max, P_wind_max, P_fcin, C_grid, cost= Single_HRES(HRES='1',tim=tim,Tr=Tr,T=T,Nh=100,Nps=4000,Nw=15,eta_c=1000,N_fc_s=30,P_dgr=25,V_cl2_max=10000,V_cl2_min=8000, n_fc_in=n_fc_i[0], n_fc_max_in =n_fc_max_i[0],g_buy=b_grid,g_sell=s_grid)
    
    [P_cr2T[0][h]] = P_cr2[0].value
    [P_fcT[0][h]] = P_fc[0].value
    [P_cr1T[0][h]] = P_cr1[0].value
    [P_dg1T[0][h]] = P_dg1[0].value
    [P_dg2T[0][h]] = P_dg2[0].value
    [P_dgT[0][h]] = P_dg[0].value
    [n_h2T[0][h]] = n_h2[0].value
    [n_fcT[0][h]] = n_fc[0].value
    [n_fc2T[0][h]] = n_fc2[0].value
    [n_fc_maxT[0][h]]  = n_fc_max[0].value
    [P_gridT[0][h]] = P_grid[0].value
    [P_grid1T[0][h]] = P_grid1[0].value
    [P_grid2T[0][h]] = P_grid2[0].value
    P_lT[0][h] = P_l[0]
    P_sol_maxT[0][h] = P_sol_max[0]
    P_wind_maxT[0][h] = P_wind_max[0]
    [P_fcinT[0][h]] = P_fcin[0]
    [C_gridT[0][h]] = C_grid[0]
    [costT[0][h]] = cost[0]
    [P_mgs[0]] = P_grid[0].value
    #####   initial values   ###
    [n_fc_i[0]] = n_fc[0].value
    [n_fc_max_i[0]] = n_fc_max[0].value
    
    ####  HRES 2 ############
    P_grid,P_grid1,P_grid2,P_cr1,P_cr2, P_fc, P_dg1, P_dg2, P_dg, n_h2, n_fc, n_fc2, n_fc_max, P_l, P_sol_max, P_wind_max, P_fcin, C_grid, cost= Single_HRES(HRES='2',tim=tim,Tr=Tr,T=T,Nh=100,Nps=2000,Nw=25,eta_c=1000,N_fc_s=30,P_dgr=25,V_cl2_max=10000,V_cl2_min=8000, n_fc_in=n_fc_i[1], n_fc_max_in =n_fc_max_i[1],g_buy=b_grid,g_sell=s_grid)
    
    [P_cr2T[1][h]] = P_cr2[0].value
    [P_fcT[1][h]] = P_fc[0].value
    [P_cr1T[1][h]] = P_cr1[0].value
    [P_dg1T[1][h]] = P_dg1[0].value
    [P_dg2T[1][h]] = P_dg2[0].value
    [P_dgT[1][h]] = P_dg[0].value
    [n_h2T[1][h]] = n_h2[0].value
    [n_fcT[1][h]] = n_fc[0].value
    [n_fc2T[1][h]] = n_fc2[0].value
    [n_fc_maxT[1][h]]  = n_fc_max[0].value
    [P_gridT[1][h]] = P_grid[0].value
    [P_grid1T[1][h]] = P_grid1[0].value
    [P_grid2T[1][h]] = P_grid2[0].value
    P_lT[1][h] = P_l[0]
    P_sol_maxT[1][h] = P_sol_max[0]
    P_wind_maxT[1][h] = P_wind_max[0]
    [P_fcinT[1][h]] = P_fcin[0]
    [C_gridT[1][h]] = C_grid[0]
    [costT[1][h]] = cost[0]
    [P_mgs[1]] = P_grid[0].value
    #####   initial values   ###
    [n_fc_i[1]] = n_fc[0].value
    [n_fc_max_i[1]] = n_fc_max[0].value
    
    ####  HRES 3 ############
    P_grid,P_grid1,P_grid2,P_cr1,P_cr2, P_fc, P_dg1, P_dg2, P_dg, n_h2, n_fc, n_fc2, n_fc_max, P_l, P_sol_max, P_wind_max, P_fcin, C_grid, cost= Single_HRES(HRES='3',tim=tim,Tr=Tr,T=T,Nh=100,Nps=1000,Nw=50,eta_c=1000,N_fc_s=30,P_dgr=25,V_cl2_max=10000,V_cl2_min=8000, n_fc_in=n_fc_i[2], n_fc_max_in =n_fc_max_i[2],g_buy=b_grid,g_sell=s_grid)
    
    [P_cr2T[2][h]] = P_cr2[0].value
    [P_fcT[2][h]] = P_fc[0].value
    [P_cr1T[2][h]] = P_cr1[0].value
    [P_dg1T[2][h]] = P_dg1[0].value
    [P_dg2T[2][h]] = P_dg2[0].value
    [P_dgT[2][h]] = P_dg[0].value
    [n_h2T[2][h]] = n_h2[0].value
    [n_fcT[2][h]] = n_fc[0].value
    [n_fc2T[2][h]] = n_fc2[0].value
    [n_fc_maxT[2][h]]  = n_fc_max[0].value
    [P_gridT[2][h]] = P_grid[0].value
    [P_grid1T[2][h]] = P_grid1[0].value
    [P_grid2T[2][h]] = P_grid2[0].value
    P_lT[2][h] = P_l[0]
    P_sol_maxT[2][h] = P_sol_max[0]
    P_wind_maxT[2][h] = P_wind_max[0]
    [P_fcinT[2][h]] = P_fcin[0]
    [C_gridT[2][h]] = C_grid[0]
    [costT[2][h]] = cost[0]
    [P_mgs[2]] = P_grid[0].value
    #####   initial values   ###
    [n_fc_i[2]] = n_fc[0].value
    [n_fc_max_i[2]] = n_fc_max[0].value
    
    ####  HRES 4 ############
    P_grid,P_grid1,P_grid2,P_cr1,P_cr2, P_fc, P_dg1, P_dg2, P_dg, n_h2, n_fc, n_fc2, n_fc_max, P_l, P_sol_max, P_wind_max, P_fcin, C_grid, cost= Single_HRES(HRES='4',tim=tim,Tr=Tr,T=T,Nh=100,Nps=1000,Nw=50,eta_c=1000,N_fc_s=30,P_dgr=25,V_cl2_max=10000,V_cl2_min=8000, n_fc_in=n_fc_i[3], n_fc_max_in =n_fc_max_i[3],g_buy=b_grid,g_sell=s_grid)
    
    [P_cr2T[3][h]] = P_cr2[0].value
    [P_fcT[3][h]] = P_fc[0].value
    [P_cr1T[3][h]] = P_cr1[0].value
    [P_dg1T[3][h]] = P_dg1[0].value
    [P_dg2T[3][h]] = P_dg2[0].value
    [P_dgT[3][h]] = P_dg[0].value
    [n_h2T[3][h]] = n_h2[0].value
    [n_fcT[3][h]] = n_fc[0].value
    [n_fc2T[3][h]] = n_fc2[0].value
    [n_fc_maxT[3][h]]  = n_fc_max[0].value
    [P_gridT[3][h]] = P_grid[0].value
    [P_grid1T[3][h]] = P_grid1[0].value
    [P_grid2T[3][h]] = P_grid2[0].value
    P_lT[3][h] = P_l[0]
    P_sol_maxT[3][h] = P_sol_max[0]
    P_wind_maxT[3][h] = P_wind_max[0]
    [P_fcinT[3][h]] = P_fcin[0]
    [C_gridT[3][h]] = C_grid[0]
    [costT[3][h]] = cost[0]
    [P_mgs[3]] = P_grid[0].value
    #####   initial values   ###
    [n_fc_i[3]] = n_fc[0].value
    [n_fc_max_i[3]] = n_fc_max[0].value
    
     ####  HRES 5 ############
    P_grid,P_grid1,P_grid2,P_cr1,P_cr2, P_fc, P_dg1, P_dg2, P_dg, n_h2, n_fc, n_fc2, n_fc_max, P_l, P_sol_max, P_wind_max, P_fcin, C_grid, cost= Single_HRES(HRES='5',tim=tim,Tr=Tr,T=T,Nh=100,Nps=4000,Nw=15,eta_c=1000,N_fc_s=30,P_dgr=25,V_cl2_max=10000,V_cl2_min=8000, n_fc_in=n_fc_i[4], n_fc_max_in =n_fc_max_i[4],g_buy=b_grid,g_sell=s_grid)
    
    [P_cr2T[4][h]] = P_cr2[0].value
    [P_fcT[4][h]] = P_fc[0].value
    [P_cr1T[4][h]] = P_cr1[0].value
    [P_dg1T[4][h]] = P_dg1[0].value
    [P_dg2T[4][h]] = P_dg2[0].value
    [P_dgT[4][h]] = P_dg[0].value
    [n_h2T[4][h]] = n_h2[0].value
    [n_fcT[4][h]] = n_fc[0].value
    [n_fc2T[4][h]] = n_fc2[0].value
    [n_fc_maxT[4][h]]  = n_fc_max[0].value
    [P_gridT[4][h]] = P_grid[0].value
    [P_grid1T[4][h]] = P_grid1[0].value
    [P_grid2T[4][h]] = P_grid2[0].value
    P_lT[4][h] = P_l[0]
    P_sol_maxT[4][h] = P_sol_max[0]
    P_wind_maxT[4][h] = P_wind_max[0]
    [P_fcinT[4][h]] = P_fcin[0]
    [C_gridT[4][h]] = C_grid[0]
    [costT[4][h]] = cost[0]
    [P_mgs[4]] = P_grid[0].value
    #####   initial values   ###
    [n_fc_i[4]] = n_fc[0].value
    [n_fc_max_i[4]] = n_fc_max[0].value
    
    P_mgs = np.round_(P_mgs, decimals=3)
    [P_loss_un[h]],[TMV_un[h]], [Total_c_un[h]] = Multi_HRES2(x,y,P_mgs,bid,ask, b_grid, s_grid)
    Ss=0
    Sb=0
    for i in range(Ng):
        if P_mgs[i] >0:
            Sb = Sb+1
        elif P_mgs[i] <0:
            Ss = Ss+1
    if (Ss>0) and (Sb>0):
        P_ds[h], P_ms[h], [P_loss[h]],[TMV[h]], [Total_c[h]] = Multi_HRES(x,y,P_mgs,bid,ask, b_grid, s_grid)
    else:
        P_ms[h] = P_mgs
        
P_gridT = np.round_(P_gridT, decimals=3)
P_ds = np.round_(P_ds, decimals=3)
P_ms = np.round_(P_ms, decimals=3)

for i in range(T+1):
    if P_loss[i]>=P_loss_un[i]:
        P_loss[i] = None
        TMV[i] = None
        Total_c[i] = None
        P_ms[i] = P_gridT[:,i]
        P_ds[i] = np.zeros([Ng,Ng])

for i in range(Ng):
    for g in range(T+1):
        if P_gridT[i][g] > 0:
            P_buy[i][g] = P_gridT[i][g]
        elif P_gridT[i][g] < 0 :
            P_sell[i][g] = -1*P_gridT[i][g]

for i in range(T+1):
    for g in range(Ng):
        if P_ms[i][g] > 0:
            P_ms_buy[i][g] = P_ms[i][g]
        elif P_ms[i][g] < 0 :
            P_ms_sell[i][g] = -1*P_ms[i][g]


for h in range(1,T+1):
    for i in range(Ng):
        for j in range(Ng):
            if P_ds[h][i][j]>0:
                P_ds_buy[h][i][j] = P_ds[h][i][j]
            elif P_ds[h][i][j]<0:
                P_ds_sell[h][i][j] = -1*P_ds[h][i][j]
    

Time = np.transpose(np.linspace(1,T,T))
#for i in range(Ng):
 #   plt.figure() 
  #  plt.plot(Time, P_fcT[i][1:T+1],'-g', Time, P_dgT[i][1:T+1],'-r',Time,P_sol_maxT[i][1:T+1],'-y',Time, P_wind_maxT[i][1:T+1],'-b',Time,P_lT[i][1:T+1],'-k')

 #   plt.figure()
  #  plt.plot(Time, P_buy[i][1:T+1],'-k',Time, P_sell[i][1:T+1])
    
   # plt.figure()
    #plt.plot(Time, costT[i][1:T+1])

   
plt.figure()
plt.plot(Time, P_loss[1:T+1],'gD',Time,P_loss_un[1:T+1],'--r*')

plt.figure()
plt.plot(Time, TMV[1:T+1],'gD',Time,TMV_un[1:T+1],'--r*')

plt.figure()
plt.plot(Time, Total_c[1:T+1],'gD',Time,Total_c_un[1:T+1],'--r*')
