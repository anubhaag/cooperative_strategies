###########Do this and then export ###############
import numpy as np
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
##############################################################

import matplotlib.pyplot as plt

markers =['o', 'v','+','*','D']
colors = ['r', 'b','g','c', 'm']


####################       Data plots(1*3)    #######
import pandas as pd
import numpy as np
G = np.zeros([Ng,42])
W = np.zeros([Ng,42])
df = pd.read_excel (r'D:\MTP\coaliation\Varodara.xlsx')
t = df['Time(IST)'].to_list()

fig,axs = plt.subplots(1,3,figsize = (14,4))

for i in range(Ng):
    G[i] = df['G'+str(i+1)].to_list()
    axs[0].plot(t[0:T],G[i][0:T], linestyle='--', marker=markers[i], color=colors[i],markersize=5,linewidth=1, label='HRES'+str(i+1))
axs[0].set_ylabel('Solar Irradiance(W/m2)',fontsize=14)
axs[0].set_xlabel('Time(hrs)',fontsize=14)
axs[0].legend(loc='best',markerscale=1.0)
axs[0].set_title('(a)')

for i in range(Ng):
    W[i] = df['W'+str(i+1)].to_list()
    axs[1].plot(t[0:T],W[i][0:T], linestyle='--', marker=markers[i], color=colors[i], markersize = 5,linewidth=1)
axs[1].set_ylabel('Wind speed(m/s)',fontsize=14)
axs[1].set_xlabel('Time(hrs)',fontsize=14)
axs[1].set_title('(b)')

for i in range(Ng):
    axs[2].plot(Time,P_lT[i][1:T+1], linestyle='--', marker=markers[i], color=colors[i], markersize = 5,linewidth=1)
axs[2].set_ylabel('Load(Kw)',fontsize=14)
axs[2].set_xlabel('Time(hrs)',fontsize=14)
axs[2].set_title('(c)')

############  Single HRES(5x3)  ###########
fig,axs = plt.subplots(5,3,figsize = (14,16),sharex='col',gridspec_kw={'hspace': 0.22})

axs[0,0].stackplot(Time, P_fcT[0][1:T+1], P_dgT[0][1:T+1],P_sol_maxT[0][1:T+1], P_wind_maxT[0][1:T+1],labels=['Fuel cells','Diesel generator','Solar','Wind'])
axs[0,0].legend(loc='best',fontsize=9,ncol=2)
axs[0,0].set_title('(a(i)) HRES 1')

axs[0,1].plot(Time, P_buy[0][1:T+1],'r--o',markersize=5,linewidth=1.0,label='Buy from Grid')
axs[0,1].plot(Time, P_sell[0][1:T+1],'b--D',markersize=5,linewidth=1.0,label='Sell to grid')
axs[0,1].set_title('(a(ii))')

axs[0,2].plot(Time, costT[0][1:T+1], linestyle='--', marker='D', color='b',markersize=5,linewidth=1.0)
axs[0,2].set_title('(a(iii))')

axs[1,0].stackplot(Time, P_fcT[1][1:T+1], P_dgT[1][1:T+1],P_sol_maxT[1][1:T+1], P_wind_maxT[1][1:T+1],labels=['fuel cells','diesel','Solar','Wind'])
axs[1,0].set_title('(b(i)) HRES 2')

axs[1,1].plot(Time, P_buy[1][1:T+1],'r--o',markersize=5,linewidth=1.0,label='Buy from Grid')
axs[1,1].plot(Time, P_sell[1][1:T+1],'b--D',markersize=5,linewidth=1.0,label='Sell to grid')
axs[1,1].set_title('(b(ii))')

axs[1,2].plot(Time, costT[1][1:T+1], linestyle='--', marker='D', color='b',markersize=5,linewidth=1.0)
axs[1,2].set_title('(b(iii))')
axs[1,2].set_ylabel('Electricity Price(INR/KWh)',fontsize=14)

axs[2,0].stackplot(Time, P_fcT[2][1:T+1], P_dgT[2][1:T+1],P_sol_maxT[2][1:T+1], P_wind_maxT[2][1:T+1],labels=['fuel cells','diesel','Solar','Wind'])
axs[2,0].set_title('(c(i)) HRES 3')
axs[2,0].set_ylabel('Electricity Generation(KWh)',fontsize=14)

axs[2,1].plot(Time, P_buy[2][1:T+1],'r--o',markersize=5,linewidth=1.0,label='Buy from Grid')
axs[2,1].plot(Time, P_sell[2][1:T+1],'b--D',markersize=5,linewidth=1.0,label='Sell to grid')
axs[2,1].set_ylabel('Electricity(KWh)',fontsize=14)
axs[2,1].legend(loc='best',fontsize=9)
axs[2,1].set_title('(c(ii))')

axs[2,2].plot(Time, costT[2][1:T+1], linestyle='--', marker='D', color='b',markersize=5,linewidth=1.0)
axs[2,2].set_title('(c(iii))')

axs[3,0].stackplot(Time, P_fcT[3][1:T+1], P_dgT[3][1:T+1],P_sol_maxT[3][1:T+1], P_wind_maxT[3][1:T+1],labels=['fuel cells','diesel','Solar','Wind'])
axs[3,0].set_title('(d(i)) HRES 4')

axs[3,1].plot(Time, P_buy[3][1:T+1],'r--o',markersize=5,linewidth=1.0,label='Buy from Grid')
axs[3,1].plot(Time, P_sell[3][1:T+1],'b--D',markersize=5,linewidth=1.0,label='Sell to grid')
axs[3,1].set_title('(d(i))')

axs[3,2].plot(Time, costT[3][1:T+1], linestyle='--', marker='D', color='b',markersize=5,linewidth=1.0)
axs[3,2].set_title('(d(iii))')

axs[4,0].stackplot(Time, P_fcT[4][1:T+1], P_dgT[4][1:T+1],P_sol_maxT[4][1:T+1], P_wind_maxT[4][1:T+1],labels=['fuel cells','diesel','Solar','Wind'])
axs[4,0].set_title('(e(i)) HRES 5')
axs[4,0].set_xlabel('Time(hrs)',fontsize=14)

axs[4,1].plot(Time, P_buy[4][1:T+1],'r--o',markersize=5,linewidth=1.0,label='Buy from Grid')
axs[4,1].plot(Time, P_sell[4][1:T+1],'b--D',markersize=5,linewidth=1.0,label='Sell to grid')
axs[4,1].set_xlabel('Time(hrs)',fontsize=13)
axs[4,1].set_title('(e(ii))')

axs[4,2].plot(Time, costT[4][1:T+1], linestyle='--', marker='D', color='b',markersize=5,linewidth=1.0)
axs[4,2].set_xlabel('Time(hrs)',fontsize=14)
axs[4,2].set_title('(e(iii))')
fig.savefig("D:\\MTP\\coaliation\\hres_12345.svg")


############  Single HRES(3x2)  ###########
fig,axs = plt.subplots(3,2,figsize = (10,8),sharex='col',gridspec_kw={'hspace': 0.1})

axs[0,0].stackplot(Time, P_fcT[3][1:T+1], P_dgT[3][1:T+1],P_sol_maxT[3][1:T+1], P_wind_maxT[3][1:T+1],labels=['fuel cells','diesel','Solar','Wind'])
axs[0,0].set(ylabel = 'Electricity Generation(KWh)')
axs[0,0].set_title('(a) HRES 4')

axs[1,0].plot(Time, P_buy[3][1:T+1],'r--o',markersize=5,linewidth=1.0,label='Buy from Grid')
axs[1,0].plot(Time, P_sell[3][1:T+1],'b--D',markersize=5,linewidth=1.0,label='Sell to grid')
axs[1,0].legend(loc='best',fontsize=8)
axs[1,0].set(ylabel = 'Electricity(KWh)')

axs[2,0].plot(Time, costT[3][1:T+1], linestyle='--', marker='D', color='b',markersize=5,linewidth=1.0)
axs[2,0].set(ylabel = 'Electricity Price(INR/KWh)')
axs[2,0].set(xlabel = 'Time(hrs)')

axs[0,1].stackplot(Time, P_fcT[4][1:T+1], P_dgT[4][1:T+1],P_sol_maxT[4][1:T+1], P_wind_maxT[4][1:T+1],labels=['fuel cells','diesel','Solar','Wind'])
axs[0,1].set_title('(b) HRES 5')
axs[0,1].legend(loc='best',fontsize=8)

axs[1,1].plot(Time, P_buy[4][1:T+1],'r--o',markersize=5,linewidth=1.0,label='Buy from Grid')
axs[1,1].plot(Time, P_sell[4][1:T+1],'b--D',markersize=5,linewidth=1.0,label='Sell to grid')

axs[2,1].plot(Time, costT[4][1:T+1], linestyle='--', marker='D', color='b',markersize=5,linewidth=1.0)
axs[2,1].set(xlabel = 'Time(hrs)')


########   HRES locations  ###########
colors = ['r', 'b','g','c', 'm']
plt.figure()
plt.plot(0,0,'D',color='black',markersize = 25,fillstyle='none')
plt.text(5,5,'Macro station',color='r',fontsize=13)
for i in range(Ng):    
    plt.plot(x[i],y[i],'^',color=colors[i],markersize=10)
plt.ylabel('y-coordinate(Kms)',fontsize=14)
plt.xlabel('x-coordinate(Kms)',fontsize=14)
plt.grid(b='True',linestyle='--')
i=0
for x1, y1 in zip(x, y):
    plt.text(x1, y1, str(i+1), color="black", fontsize=10)
    i=i+1

######  Individual trading results(5x2)  ###########
import matplotlib.pyplot as plt

markers =['o', 'v','+','*','D']
colors = ['k', 'y','g','c', 'm']
nu1 = ['(a(i))','(b(i))','(c(i))', '(d(i))','(e(i))' ]
nu2 = ['(a(ii))','(b(ii))','(c(ii))', '(d(ii))','(e(ii))' ]

fig,axs = plt.subplots(5,2,figsize = (10,12),sharex='col',gridspec_kw={'hspace': 0.22})
for j in range(Ng):
    for i in range(Ng):
        if i!=j:
            axs[j,0].plot(Time, P_ds_buy[1:T+1,j,i],linestyle='--',color=colors[i],marker=markers[i],markersize=5,linewidth=1.0,label=str(i+1))
    axs[j,0].plot(Time,P_ms_buy[1:T+1,j],'b--1',markersize=5,linewidth=1.0,label='MS')
    axs[j,0].plot(Time,P_buy[j,1:T+1],'r--2',markersize=5,linewidth=1.0,label='NC')
    axs[j,0].legend(loc='best',fontsize=10,ncol=2) 
    axs[j,0].set_title(nu1[j] +' HRES '+str(j+1))
    for i in range(Ng):
        if i!=j:
            axs[j,1].plot(Time, P_ds_sell[1:T+1,j,i],linestyle='--',color=colors[i],marker=markers[i],markersize=5,linewidth=1.0,label=str(i+1))
    axs[j,1].plot(Time,P_ms_sell[1:T+1,j],'b--1',markersize=5,linewidth=1.0,label='MS')
    axs[j,1].plot(Time,P_sell[j,1:T+1],'r--2',markersize=5,linewidth=1.0,label='NC')
    axs[j,1].set_title(nu2[j])

axs[1,0].set_ylabel('Energy deficit(KWh)',fontsize=12)
axs[1,1].set_ylabel('Energy surplus(KWh)',fontsize=12)
axs[4,0].set_xlabel('Time(hrs)',fontsize=12)
axs[4,1].set_xlabel('Time(hrs)',fontsize=12)
fig.savefig("D:\\MTP\\coaliation\\trading_12345.svg")   

###### Overall Power loss graphs  #############
P_loss1 = P_loss
TMV1 = TMV
P_loss2 = P_loss
TMV2 = TMV
P_loss3 = P_loss
TMV3 = TMV

import matplotlib.pyplot as plt
fig,axs = plt.subplots(1,2,figsize = (12,5))
axs[0].plot(Time, P_loss1[1:T+1],'Dg',markersize=5,linewidth=1.0,label='Cooperative Stratergy 1')
axs[0].plot(Time, P_loss2[1:T+1],'b*',markersize=5,linewidth=1.0,label='Cooperative Stratergy 2')
axs[0].plot(Time, P_loss3[1:T+1],'mx',markersize=5,linewidth=1.0,label='Cooperative Stratergy 3')
axs[0].plot(Time,P_loss_un[1:T+1],'o--r',markersize=5,linewidth=1.0,label='Non-Cooperative')
axs[0].set_ylabel('Power loss(Kwh)',fontsize=14)
axs[0].set_xlabel('Time(hrs)',fontsize=14)
axs[0].set_title("(i)",fontsize=13)
axs[0].legend(loc='upper center',fontsize=12)

###### Overall Market utikity graphs  #############
axs[1].plot(Time, TMV1[1:T+1],'Dg',markersize=5,linewidth=1.0,label='Cooperative Stratergy 1')
axs[1].plot(Time, TMV2[1:T+1],'b*',markersize=5,linewidth=1.0,label='Cooperative Stratergy 2')
axs[1].plot(Time, TMV3[1:T+1],'mx',markersize=5,linewidth=1.0,label='Cooperative Stratergy 3')
axs[1].plot(Time,TMV_un[1:T+1],'o--r',markersize=5,linewidth=1.0,label='Non-Cooperative')
axs[1].set_ylabel('Total market utility(INR)',fontsize=14)
axs[1].set_xlabel('Time(hrs)',fontsize=14)
axs[1].set_title("(ii)",fontsize=13)
####### average electricity cost ####
for i in range(Ng):
    print(np.round_(np.average(costT[i,1:T+1]),decimals=2))

#####   Daily profits   ######
C_e = 4.0 #INR/kW
C_cl2 = 1.4 #INR/mol 20 INR/kg
eta_c = 1000
for i in range(Ng):
    print(np.round_((C_e*np.sum(P_fcinT[i,1:T+1])-eta_c*C_cl2*np.sum(n_h2T[i,1:T+1]) +np.sum(np.multiply(P_gridT[i,1:T+1],C_gridT[i,1:T+1]))+np.sum(np.multiply(costT[i,1:T+1],(P_sol_maxT[i,1:T+1] + P_wind_maxT[i,1:T+1] + P_fcT[i,1:T+1] + P_dgT[i,1:T+1]))))/100) )

print(np.round_((b_grid*np.sum(P_lT[0,1:T+1]))/100))

### Energy exchange #########
import numpy as np
for i in range(Ng):
    print(np.round_(np.sum(P_ds_buy[6:18,i,:])+np.sum(P_ds_sell[6:18,i,:]), decimals=2))
        
for i in range(Ng):
    print(np.round_(np.sum(P_ms_buy[6:18,i])+np.sum(P_ms_sell[6:18,i]), decimals=2))

for i in range(Ng):
    print(np.round_(np.sum(P_ds_buy[1:6,i,:])+np.sum(P_ds_buy[18:24,i,:])+np.sum(P_ds_sell[1:6,i,:])+np.sum(P_ds_sell[18:24,i,:]), decimals=2))
        
for i in range(Ng):
    print(np.round_(np.sum(P_ms_buy[1:6,i])+np.sum(P_ms_buy[18:24,i])+np.sum(P_ms_sell[1:6,i])+np.sum(P_ms_sell[18:24,i]), decimals=2))

### Energy exchange (buy and sell seperately)#########
import numpy as np
for i in range(Ng):
    print(np.round_(np.sum(P_ds_buy[6:18,i,:]), decimals=2))
        
for i in range(Ng):
    print(np.round_(np.sum(P_ds_sell[6:18,i,:]), decimals=2))
            
for i in range(Ng):
    print(np.round_(np.sum(P_ms_buy[6:18,i]), decimals=2))

for i in range(Ng):
    print(np.round_(np.sum(P_ms_sell[6:18,i]), decimals=2))

for i in range(Ng):
    print(np.round_(np.sum(P_ds_buy[1:6,i,:])+np.sum(P_ds_buy[18:24,i,:]), decimals=2))

for i in range(Ng):
    print(np.round_(np.sum(P_ds_sell[1:6,i,:])+np.sum(P_ds_sell[18:24,i,:]), decimals=2))
        
for i in range(Ng):
    print(np.round_(np.sum(P_ms_buy[1:6,i])+np.sum(P_ms_buy[18:24,i]), decimals=2))

for i in range(Ng):
    print(np.round_(np.sum(P_ms_sell[1:6,i])+np.sum(P_ms_sell[18:24,i]), decimals=2))

##### total power loss ###########(Only considering the trading instants)
import math
un_loss=0
for i in range(1,T+1):
    if math.isnan(P_loss[i])==False:
        un_loss = un_loss + P_loss_un[i] 
print(np.nansum(P_loss[1:T+1]))

un_tmv=0
for i in range(1,T+1):
    if math.isnan(TMV[i])==False:
        un_tmv = un_tmv + TMV_un[i] 
print(np.nansum(TMV[1:T+1]))

##########counting of local trade instances #######
np.count_nonzero(~np.isnan(P_loss))

##### total power loss ###########
import numpy as np
import math
print(np.sum(P_loss_un[1:T+1]))
loss=0
for i in range(1,T+1):
    if math.isnan(P_loss[i])==True:
        loss = loss + P_loss_un[i] 
print(np.nansum(P_loss[1:T+1])+loss)

print(np.sum(TMV_un[1:T+1]))
tmv=0
for i in range(1,T+1):
    if math.isnan(TMV[i])==True:
        tmv = tmv + TMV_un[i] 
print(np.nansum(TMV[1:T+1])+tmv)
