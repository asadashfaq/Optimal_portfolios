#!/usr/bin/python
import numpy as np
import heapq as hq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure, show


######################################################################################
############################## Loading data ##########################################
######################################################################################
"""
# Files[24:26] is AA
Files = list('Data_files/ISET_country_AA.npz')
Country_names = ['AT','BA','BE','BG','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LU','LV','NL','NO','PL','PT','RO','RS','SE','SI','SK']
# This for-loop is used to replace AA with the country codes, and then combine the list into a string
for x in range(len(Country_names)):
	Files[24:26] = Country_names[x]
	Filename = ''.join(Files)
	Country_names[x] = np.load(Filename)
"""
######################################################################################
############## Calculating key properties for Wind, Solar and Load ###################
######################################################################################

rolando_alphas = [0.701074, 0.660791, 0.690674, 0.695068, 0.659473, 0.812915, 0.650977, 0.646582, 0.731543, 0.716455, 0.641748, 0.849023, 0.674268, 0.639551, 0.70708, 0.68291, 0.696533, 0.755273, 0.713086, 0.787354, 0.706787, 0.662549, 0.795557, 0.789551, 0.68862, 0.716016, 0.739453, 0.810791, 0.817676, 0.754102]
# We load the wind-data from the ISET-files and calculate the standard deviation of the individual countries.
Wind = np.load('windsy.npy')
Wind_std = [np.std(Wind[x]) for x in range(30)]
Wind_cf = (np.load('cf_wind.npy'))

Solar = np.load('solarsy.npy')
Solar_std = [np.std(Solar[x]) for x in range(30)]
Solar_cf = (np.load('cf_solar.npy'))

# We load the load-data from the ISET-files and calculate the mean load
Load = np.load('loadsy.npy')
Mean_load = np.array([np.mean(Load[j]) for j in range(30)])
EU_L =sum(Load)

Std_wind_with_load = np.load('total_std_with_load.npy')
Std_wind_without_load = np.load('total_std.npy') 
Cf_wind= np.load('total_cf.npy')
Gamma_wind = np.load('gammas_wind.npy')

Std_solar_with_load = np.load('total_std_solar_with_load.npy')
Std_solar_without_load = np.load('total_solar_std.npy') 
Cf_solar= np.load('total_solar_cf.npy')
Gamma_solar = np.load('gamma_solar.npy')

######################################################################################
####################### Loading optimal gammas and alphas    #########################
######################################################################################'

Optimal_gamma_solar = np.load('Optimal_gamma_solar.npy')
Optimal_std_solar = np.load('Optimal_std_solar.npy')
Optimal_cf_solar = np.load('Optimal_cf_solar.npy')
Optimal_gamma_solar_with_load = np.load('Optimal_gamma_solar_with_load.npy')
Optimal_std_solar_with_load = np.load('Optimal_std_solar_with_load.npy')
Optimal_cf_solar_with_load = np.load('Optimal_cf_solar_with_load.npy')
Optimal_gamma_wind = np.load('Optimal_gamma_wind.npy')
Optimal_std_wind = np.load('Optimal_std_wind.npy')
Optimal_cf_wind = np.load('Optimal_cf_wind.npy')
Optimal_gamma_wind_with_load = np.load('Optimal_gamma_wind_with_load.npy')
Optimal_std_wind_with_load = np.load('Optimal_std_wind_with_load.npy')
Optimal_cf_wind_with_load = np.load('Optimal_cf_wind_with_load.npy')

alphas = np.load('GENERATIONS_5/total_population_0.8.npz')['alpha']
balancing = np.load('GENERATIONS_5/total_population_0.8.npz')['sd']
p = []
for x in range(1010):
	p.append([alphas[x][y]*Mean_load[y]/EU_L.mean() for y in range(30)])
	
total_alpha = []
for x in range(1010):
	total_alpha.append(sum(p[x])) 
	
alphas2 = np.load('GENERATIONS_QUANTILES/total_population_0.8.npz')['alpha']
quantiles = np.load('GENERATIONS_QUANTILES/total_population_0.8.npz')['sd']
p = []
for x in range(1010):
	p.append([alphas2[x][y]*Mean_load[y]/EU_L.mean() for y in range(30)])
	
total_alpha2 = []
for x in range(1010):
	total_alpha2.append(sum(p[x]))
	
alphas3 = np.load('GENERATIONS_WITH_LOAD/total_population_0.8.npz')['alpha']
balancing2 = np.load('GENERATIONS_WITH_LOAD/total_population_0.8.npz')['sd']
p = []
for x in range(1010):
	p.append([alphas3[x][y]*Mean_load[y]/EU_L.mean() for y in range(30)])
	
total_alpha3 = []
for x in range(1010):
	total_alpha3.append(sum(p[x])) 
alphas4 = np.load('GENERATIONS_QUANTILES_WITH_LOAD/total_population_0.8.npz')['alpha']
quantiles2 = np.load('GENERATIONS_QUANTILES_WITH_LOAD/total_population_0.8.npz')['sd']
p = []
for x in range(1010):
	p.append([alphas4[x][y]*Mean_load[y]/EU_L.mean() for y in range(30)])
	
total_alpha4 = []
for x in range(1010):
	total_alpha4.append(sum(p[x]))

Std_file = list('GENERATIONS_4/total_population_AAA.npz')
Std_optim = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
Cf_optim = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
Gamma_optim = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
Alphas = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']

for x in range(11):
	Std_file[31:34] = Std_optim[x]
	Filename_std = ''.join(Std_file)
	Std_optim[x] = np.load(Filename_std)['sd']
	Cf_optim[x] = np.load(Filename_std)['cf']
	Gamma_optim[x] = np.load(Filename_std)['gammas']

######################################################################################
##### Making gamma; calculating Standard deviation and Capacity factor ###############
######################################################################################

#This function makes random numbers, and then add a correction, beta, to match the constraint that gamma_eu = 1
def make_gamma():
	gamma_0 = np.random.random(30)*2
	#beta = sum(gamma_0*Mean_load)/EU_L.mean() - 1
	gamma_1 = sum(gamma_0*Mean_load)/EU_L.mean()
	gamma = gamma_0/gamma_1
	return gamma

def corrected(gamma):
	gamma_1 = sum(gamma*Mean_load)/EU_L.mean()
	gamma_corrected = gamma/gamma_1
	return gamma_corrected
	
def make_alpha():
	alpha = np.random.rand(30)
	return alpha
	
def make_alpha_1():
	alpha_1 = np.ones(30)
	return alpha_1
	
def make_alpha_0():
	alpha_0 = np.zeros(30)
	return alpha_0
	
#This function makes the reference point where gamma = 1 in each country
def make_gamma_1():
	gamma_1 = np.ones(30)
	return gamma_1

def total_std_mismatch(gammas,alphas):
	vres = [Mean_load[x]*gammas[x]*alphas[x]*Wind[x]+(1-alphas[x])*Mean_load[x]*gammas[x]*Solar[x]-Load[x] for x in range(30)]
	std = sum(vres).std()/1000
	return std

def total_std(gammas,alphas):
	vres = [Mean_load[x]*gammas[x]*alphas[x]*Wind[x]+(1-alphas[x])*Mean_load[x]*gammas[x]*Solar[x] for x in range(30)]
	std = sum(vres).std()/1000
	return std

def total_cf(gammas,alphas):
	installed_c = sum(gammas*alphas*Mean_load*Wind_cf) + sum(gammas*(1-alphas)*Mean_load*Solar_cf)
	capfac = EU_L.mean()/(installed_c)
	return capfac

######################################################################################
####################### Calculating Mismatch and Balancing energy ####################
######################################################################################

def mismatch_energy_2(gamma_wind,gamma_solar,alpha):
	mismatch = []
	for x in range(30):
		mismatch.append((gamma_wind[x]*alpha*Wind[x]*Mean_load[x]+gamma_solar[x]*(1-alpha)*Solar[x]*Mean_load[x])-Load[x])
	total_mismatch = (np.sum(mismatch,axis=0))
	return total_mismatch

def mismatch_energy_just_for_rolandos_alphas(gamma_wind,gamma_solar,alpha):
	mismatch = []
	for x in range(30):
		mismatch.append((gamma_wind[x]*alpha[x]*Wind[x]*Mean_load[x]+gamma_solar[x]*(1-alpha[x])*Solar[x]*Mean_load[x])-Load[x])
	total_mismatch = (np.sum(mismatch,axis=0))
	return total_mismatch	

def balancing_single(gamma_wind,gamma_solar,alpha):
	balancing = []
	for x in range(30):
		mismatch = (gamma_wind[x]*alpha[x]*Wind[x]*Mean_load[x]+gamma_solar[x]*(1-alpha[x])*Solar[x]*Mean_load[x]-Load[x])
		negative_mismatch_index = ([mismatch[y] < 0 for y in range(len(mismatch))])
		negative_mismatch = ([negative_mismatch_index[z] for z in range(len(mismatch))]*mismatch)
		balancing.append(np.average(negative_mismatch)*(-1))
	return balancing
	
def balancing_energy_2(gamma_wind,gamma_solar,alpha):
	mismatch = mismatch_energy_2(gamma_wind,gamma_solar,alpha)
	negative_mismatch_index = ([mismatch[y] < 0 for y in range(len(mismatch))])
	negative_mismatch = ([negative_mismatch_index[z] for z in range(len(mismatch))]*mismatch)
	average = np.average(negative_mismatch)*(-1)/EU_L.mean()
	return average
	
def balancing_energy_just_for_rolandos_alphas(gamma_wind,gamma_solar,alpha):
	mismatch = mismatch_energy_just_for_rolandos_alphas(gamma_wind,gamma_solar,alpha)
	negative_mismatch_index = ([mismatch[y] < 0 for y in range(len(mismatch))])
	negative_mismatch = ([negative_mismatch_index[z] for z in range(len(mismatch))]*mismatch)
	average = np.average(negative_mismatch)*(-1)/EU_L.mean()
	return average	
	
######################################################################################
############################# Calculating 1 % quantiles ##############################
######################################################################################

def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx

def integration(gamma,alpha):
	b = mismatch_energy_2(gamma,alpha)
	i = []; h = []	
	#c = min(b[x])
	#d = max(b[x])
	#for x in range(len(b)):
	e = np.percentile(b,99)
	f = np.percentile(b,1)
	#g = list(np.sort(b[x]))
	h.append(find_nearest(b,e))
	i.append(find_nearest(b,f))
	#j = len(g[0:g.index(i)])
	#k = len(g[g.index(h):-1])
	return i, h

######################################################################################
################## Finding optimal gamma values ######################################
######################################################################################

starting_points = [405,359,314,271,230,190,155,124,106,110,134]
intervals = [5,4,3,2.5,2,2,2.5,2,2.5,3,3]
	
def find_optimal_layouts():
	new_gamma = []; new_std = []; new_cf = []; old_gamma = []
	for z in range(11):
		for x in range(3):
			over = (Std_optim[z]>starting_points[z]+intervals[z]*x).nonzero()
			under = (Std_optim[z]<(starting_points[z]+intervals[z])+intervals[z]*x).nonzero()
			indices = np.intersect1d(over[0],under[0])
			cf = list(Cf_optim[z])
			list_cf = list(np.unique(Cf_optim[z][indices]))
			top_values = ([cf.index(hq.nlargest(15,list_cf)[y]) for y in range(15)])
			average_gamma = (np.average(Gamma_optim[z][top_values], axis=0))
			average_cf = (np.average(Cf_optim[z][top_values], axis=0))
			average_std = (np.average(Std_optim[z][top_values], axis=0))
			new_cf.append(total_cf(average_gamma,np.ones(30)*(z/10.0)))
			new_std.append(total_std_mismatch(average_gamma,np.ones(30)*(z/10.0)))
			#new_cf.append(average_cf)
			#new_std.append(average_std)
			new_gamma.append(average_gamma)
		#old_gamma.append(np.ones(30))
		#new_gamma.append(np.ones(30))
		new_std.append(total_std_mismatch(np.ones(30),np.ones(30)*(z/10.0)))
		new_cf.append(total_cf(np.ones(30),np.ones(30)*(z/10.0)))
	return new_gamma,new_std,new_cf

######################################################################################
############ Plotting the balancing energy as a function of alpha ####################
######################################################################################

"""
total_rolando_alpha = np.sum([rolando_alphas[y]*Mean_load[y]/EU_L.mean() for y in range(30)])

#plt.plot(total_alpha,balancing,'bo',alpha = 0.1)
plt.plot(total_alpha[1005],balancing[1005]/EU_L.mean(),'b*',markersize=8)
#plt.plot(total_alpha2,quantiles,'bo',alpha = 0.1)
plt.plot(total_alpha2[932],balancing_energy_just_for_rolandos_alphas(Optimal_gamma_wind[0],Optimal_gamma_solar[0],alphas2[932]),'r*',markersize=8)

plt.plot(total_rolando_alpha,np.sum(balancing_single(Optimal_gamma_wind[0],Optimal_gamma_solar[0],np.array(rolando_alphas)))/EU_L.mean(),'y*',markersize=8)
plt.plot(total_rolando_alpha,balancing_energy_just_for_rolandos_alphas(Optimal_gamma_wind[0],Optimal_gamma_solar[0],np.array(rolando_alphas)),'g*',markersize=8)
"""
"""
#plt.plot(total_alpha,balancing,'bo',alpha = 0.1)
plt.plot(total_alpha3[996],balancing2[996]/EU_L.mean(),'bs',markersize=8)
#plt.plot(total_alpha2,quantiles,'bo',alpha = 0.1)
plt.plot(total_alpha4[953],balancing_energy_just_for_rolandos_alphas(Optimal_gamma_wind_with_load[0],Optimal_gamma_solar_with_load[0],alphas4[953]),'rs',markersize=8)
plt.plot(total_rolando_alpha,np.sum(balancing_single(Optimal_gamma_wind_with_load[0],Optimal_gamma_solar_with_load[0],np.array(rolando_alphas)))/EU_L.mean(),'ys',markersize=8)
plt.plot(total_rolando_alpha,balancing_energy_just_for_rolandos_alphas(Optimal_gamma_wind_with_load[0],Optimal_gamma_solar_with_load[0],np.array(rolando_alphas)),'gs',markersize=8)

x2 = np.arange(0,41,1)
a = []; b = []; c = []; d = []; e = []; a1 = []; b1 = []; c1 = []; d1 = []; e1 = []
for x in range(41):
	a.append(balancing_energy_2(Optimal_gamma_wind[0],Optimal_gamma_solar[0],x/40.0))
	b.append(balancing_energy_2(Optimal_gamma_wind[1],Optimal_gamma_solar[0],x/40.0))
	c.append(balancing_energy_2(Optimal_gamma_wind[2],Optimal_gamma_solar[0],x/40.0))
	e.append(balancing_energy_2(np.ones(30),np.ones(30),x/40.0))
	#a1.append(balancing_energy_2(Optimal_gamma_wind_with_load[0],Optimal_gamma_solar_with_load[0],x/40.0))
	#b1.append(balancing_energy_2(Optimal_gamma_wind_with_load[1],Optimal_gamma_solar_with_load[0],x/40.0))
	#c1.append(balancing_energy_2(Optimal_gamma_wind_with_load[2],Optimal_gamma_solar_with_load[0],x/40.0))
plt.plot(x2[0]/40.0,a[0],'go')
plt.plot(x2[0]/40.0,b[0],'ro')
plt.plot(x2[0]/40.0,c[0],'yo')
plt.plot(x2[0]/40.0,e[0],'mo')
#plt.plot(x2[0]/40.0,a1[0],'g^')
#plt.plot(x2[0]/40.0,b1[0],'r^')
#plt.plot(x2[0]/40.0,c1[0],'y^')
"""
"""
plt.legend(['Layout found by minimizing balancing','Layout found by minimizing 99% balancing quantiles',
'layout from rolandos alphas with each balancing calculated individually','layouts from rolandos alphas','Layout found by minimizing balancing (with load)','Layout found by minimizing 99 % balancing quantiles (with load)',
'layout from rolandos alphas with balancing calculated individually (with load)','layout from rolandos alphas (with load)','wind layout 1 + solar layout','wind layout 2 + solar layout','wind layout 3 + solar layout',
'reference layout + reference layout','wind layout 1 + solar layout (with load)','wind layout 2 + solar layout (with load)' ,'wind layout 3 + solar layout (with load)'],prop={'size':9})
"""
"""
plt.legend(['Layout found by minimizing balancing (without load)','Layout found by minimizing 99 % balancing quantiles (without load)','layout from rolandos alphas with balancing calculated individually (without load)',
'layout from rolandos alphas (without load)','wind layout 1 + solar layout (without load)','wind layout 2 + solar layout (without load)','wind layout 3 + solar layout (without load)','reference layout + reference layout'],prop={'size':9})


plt.plot(x2/40.0,a,'go')
plt.plot(x2/40.0,b,'ro')
plt.plot(x2/40.0,c,'yo')
plt.plot(x2/40.0,e,'mo')
#plt.plot(x2/40.0,a1,'g^')
#plt.plot(x2/40.0,b1,'r^')
#plt.plot(x2/40.0,c1,'y^')
plt.plot(total_rolando_alpha,balancing_energy_just_for_rolandos_alphas(Optimal_gamma_wind[0],Optimal_gamma_solar[0],np.array(rolando_alphas)),'g*',markersize=8)
#plt.plot(total_rolando_alpha,balancing_energy_just_for_rolandos_alphas(Optimal_gamma_wind_with_load[0],Optimal_gamma_solar_with_load[0],np.array(rolando_alphas)),'gs',markersize=8)
plt.title('Balancing energy for different layout combinations as function of total alpha without load')
plt.ylabel('Balancing energ/Total load')
plt.xlabel('Total alpha')

plt.show()
"""
"""
one = []
nine = []
optimal_values = np.loadtxt('Data_files/optimal_points',dtype=int)

#b = find_optimal_layouts()	
for x in range(11):
	gamma = optimal_values[x*3:(x+1)*3]
	for y in range(3):
		c = (integration(Gamma_optim[x][gamma[y]],(x/10.0)))
		d = (integration(np.ones(30),(x/10.0)))
		one.append(np.round(mismatch_energy_2(Gamma_optim[x][gamma[y]],(x/10.0))[c[0]][0]/1000,2))
		nine.append(np.round(mismatch_energy_2(Gamma_optim[x][gamma[y]],(x/10.0))[c[1]][0]/1000,2))
	one.append(np.round(mismatch_energy_2(np.ones(30),(x/10.0))[d[0]][0]/1000,2))
	nine.append(np.round(mismatch_energy_2(np.ones(30),(x/10.0))[d[1]][0]/1000,2))
"""
"""
optimal_values = np.loadtxt('Data_files/optimal_points',dtype=int)

#b = find_optimal_layouts()[0]
a = []
alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for x in range(11):
	for y in range(3):
		c = optimal_values[x*3:(1+x)*3]
		a.append(balancing_energy_2(Gamma_optim[x][c[y]],alpha[x]))

		
np.save('Balancing_energy_w_load',a)

a = np.load('Balancing_energy_w_load.npy')
"""
"""
f = open('quantile_1',"a")
a = np.array(one)
for x in range(11):
	b = np.round(a[x*4:(x+1)*4],2)
	f.write(str(b[0]) + ' ' + str(b[1]) + ' ' + str(b[2]) + ' ' + str(b[3]) + '\n')
	
f.close()
"""
"""
alpha = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
b = find_optimal_layouts()
colors = ['yo','ro','co','mo']
for x in range(11):	
	plt.figure()
	c = b[1][4*x:(1+x)*4]	
	d = b[2][4*x:(1+x)*4]
	for y in range(4):
		plt.plot(c[y],d[y],colors[y],markersize=9)
	plt.legend(['layout 1','layout 2','layout 3','reference layout'])
	#for y in range(4):
		#plt.plot((x/10.0),balancing_energy_2(c[y],(x/10.0))/1000,colors[y],markersize=9)
		#plt.legend(['layout 1','layout 2','layout 3','layout 4','reference'])
	for y in range(len(Gamma_optim[1])):
		plt.plot(total_std_mismatch(Gamma_optim[x][y],np.ones(30)*(x/10.0)),total_cf(Gamma_optim[x][y],np.ones(30)*(x/10.0)),'bo')
	for y in range(4):
		plt.plot(c[y],d[y],colors[y],markersize=9)
		plt.axvline(starting_points[x]+y*intervals[x],0,1,linewidth=2,color='r')
	
	#for y in range(3):
		#e = optimal_values[x*3:(x+1)*3]
		#plt.plot(Std_optim[x][c[y]],Cf_optim[x][e[y]],'ro',markersize=9)
		#plt.plot((x/10.0),balancing_energy_2(Gamma_optim[x][e[y]],(x/10.0))/1000,colors[y],markersize=9)
	#plt.plot((x/10.0),balancing_energy_2(np.ones(30),(x/10.0))/1000,'ro',markersize=9)
	#plt.legend(['layout 1','layout 2','layout 3','reference layout'])
	#plt.plot(total_std_mismatch(np.ones(30),np.ones(30)*(x/10.0)),total_cf(np.ones(30),np.ones(30)*(x/10.0)),'ro',markersize=9)
		
	plt.title('Scatterplot with optimal and reference layout for alpha = ' + str(alpha[x]))
	plt.xlabel('Standard Deviation')
	plt.ylabel('Capacity factor')
	plt.savefig('Plots/Optimiz_new_std_alpha_' + str(alpha[x]) + '.png')
#plt.show()
"""
"""
alpha = [0.7,0.8,0.9]
alpha_nr = [8,9,10]
layout_1 = [Gamma_optim[alpha_nr[x]][optimal_values[::3][-3:][x]] for x in range(3)]
for x in range(3):
	plt.figure()
	plt.hist(mismatch_energy_2(layout_1[x],alpha[x])/1000,100,range=(-400,600),alpha=0.3,color='b')
	plt.hist(mismatch_energy_2(np.ones(30),alpha[x])/1000,100,range=(-400,600),alpha=0.3,color='r')
	plt.legend(['layout 1','reference layout'])
	plt.xlabel('Mismatch (GW)')
	plt.title('Mismatch probability distribution for ' + str(alpha[x]))
"""

######################################################################################
################ Calculating variability, plotting variability and 99 % quantiles ####
######################################################################################

#Calculates the variability, the difference in mismatches at certain time intervals. Returns dDelta, the variability and std, the variance. dDelta is a 70128*16 entrance list, so please don't open it.
def variability(gamma,alpha):
	mismatch = mismatch_energy_2(gamma,alpha)/1000
	positive_mismatch = np.where(mismatch > 0, mismatch, 0)
	negative_mismatch = np.where(mismatch < 0, mismatch, 0)
	#np.sum(x for x in mismatch[0] if x > 0)
	#negative_mismatch = []
	#negative_mismatch.append([(mismatch[x] < 0) for x in range(len(mismatch))]*mismatch)
	dDelta = []; variance = []; positive_var = []; negative_var = [];
	for x in [1,2,3,6]:
		for y in range(len(mismatch)):
			if y < 70128-x:
				positive_var.append(positive_mismatch[x+y]-positive_mismatch[y])
			else:
				pass
	for x in [1,2,3,6]:
		for y in range(len(mismatch)):
			if y < 70128-x:
				negative_var.append(negative_mismatch[y+x]-negative_mismatch[y])
			else:
				pass
	a = np.array(negative_var[0:70128-1])+np.array(positive_var[0:70128-1])
	b = np.array(negative_var[70128-1:70128*2-3])+np.array(positive_var[70128-1:70128*2-3])
	c = np.array(negative_var[70128*2-3:70128*3-6])+np.array(positive_var[70128*2-3:70128*3-6])
	d = np.array(negative_var[70128*3-6:70128*4-12])+np.array(positive_var[70128*3-6:70128*4-12])
	return a,b,c,d
"""
alpha = [0.7,0.8,0.9]
alpha_nr = [7,8,9]
layout_1 = [Gamma_optim[alpha_nr[x]][optimal_values[::3][-3:][x]] for x in range(3)]

time = [1,2,3,6]
for x in range(3):
	plt.figure()
	for y in range(4):
		steps = y
		plt.subplot(221+steps)
		a = variability(layout_1[x],alpha[x])[y]
		b = mismatch_energy_2(layout_1[x],alpha[x])[::-1]/1000
		plt.plot(b[0:70128-time[y]],a,'bo')
		plt.title('Time step: ' + str(time[y]) + 'h')
		plt.suptitle('Alpha = ' + str(alpha[x]),size=20)
		plt.xlabel('Mismatch energy (GW)')
		plt.ylabel('Variability Energy (GW)')
"""	

"""
step = [1,2,3,6]
f = open('var_99',"a")
g = open('var_1',"a")
for x in range(3):
	a = [np.percentile(variability(layout_1[x],alpha[x])[y],99) for y in range(4)]
	b = [np.percentile(variability(layout_1[x],alpha[x])[y],1) for y in range(4)]
	c = [np.percentile(variability(np.ones(30),alpha[x])[y],99) for y in range(4)]
	d = [np.percentile(variability(np.ones(30),alpha[x])[y],1) for y in range(4)]
	f.write(str(a[0]) + ' ' + str(c[0]) + '\n' + str(a[1]) + ' ' + str(c[1]) + '\n' + str(a[2]) + ' ' + str(c[2]) + '\n' + str(a[3]) + ' ' + str(c[3]) + '\n')
	g.write(str(b[0]) + ' ' + str(d[0]) + '\n' + str(b[1]) + ' ' + str(d[1]) + '\n' + str(b[2]) + ' ' + str(d[2]) + '\n' + str(b[3]) + ' ' + str(d[3]) + '\n')
f.close()
g.close()
"""

"""
a = variability(Gamma_wind[Optimal_wind_gamma_index[0]],Gamma_solar[Optimal_solar_gamma_index[0]])
titles = ['1h','2h','3h','6h']
#g = open('b',"a")
for x in range(4):
	b = a[70128*x:70128*(x+1)]
	c = np.percentile(b,1)
	d = np.percentile(b,99)
	e = find_nearest(b,c)
	f = find_nearest(b,d)
	minimum = round(min(b),2)
	maximum = round(max(b),2)
	diffmin = round(b[e]-min(b),2)
	diffmax = round(max(b)-b[f],2)
	balance = round(b[e],2)
	curtail = round(b[f],2)
	steps = 221+x
	plt.subplot(steps)
	plt.title(titles[x],size=20)
	plt.hist(b,bins=50,alpha=0.2)
	plt.vlines(minimum,0,ymax=500)
	plt.vlines(maximum,0,ymax=500)
	plt.vlines(balance,0,ymax=500)
	plt.vlines(curtail,0,ymax=500)
	plt.xlabel('Mismatch (GW)',size=20)
	plt.tick_params(labelsize='large')
	plt.annotate('minimum \n' + str(minimum),(minimum-14,500+40))
	plt.annotate('interval \n' + str(diffmin),((minimum+balance)/2-18,500-50))
	plt.annotate('1 % quantile \n' + str(balance),(balance-19,500+40))
	plt.annotate('maximum \n' + str(maximum),(maximum-14,500+40))
	plt.annotate('interval \n' + str(diffmax),((maximum+curtail)/2-18,500-50))
	plt.annotate('99 % quantile \n' + str(curtail),(curtail-19,500+40))
	plt.axis([-400,300,0,40000])
	
plt.suptitle('Wind optimal layout 01 vs Solar optimal layout, alpha = 0.8',size=40)
plt.show()
#g.write(str(balance) + '	' + str(diffmin) + '	' + str(minimum) + '	' + str(curtail) + '	' + str(diffmax) + '	' + str(maximum) + '\n')	
#g.close()
"""
"""
#Plotting the variance of the variability for different time intervals
titles = ['layout 1','layout 2','layout 3','layout 4','ref.point']
shapes = ['ro','g<','bs','ys','ro','r>']
for x in range(5):
	a, b = variability(e[x])
	steps = 231+x
	plt.subplot(steps)	
	plt.semilogy() #make sure the y-axis is logarithmic		
	for y in range(6):
		plt.plot(b,shapes[y])
	plt.legend(('24h','50h','100h','200h','300h','400h'))	
	plt.xlabel('alpha')
	plt.ylabel('variance (GW^2)')
	plt.title(titles[x])
plt.show()
"""

"""
#Plotting the variability histogram in 5 different figures (one for each layout) with the different time intervals. 
plt.ion()
titles = ['1h','2h','3h','6h']
titles2 = ['layout 1','layout 2','layout 3','layout 4','ref.point']
for y in range(4):
	plt.figure(y)
	for x in range(4):
		steps = 231+x
		plt.subplot(steps)
		#plt.axis([-400,400,0,30000])
		plt.title(titles[x])
		plt.xlabel('Mismatch power (GW)')
		a, b = variability(Gamma_wind[Optimal_wind_gamma_index[y]],Gamma_solar[Optimal_solar_gamma_index[0]])
		plt.hist(a[(x*70128+0*70128):((x+1)*70128+0*70128)],bins=100) #to use alpha = 0.8
	plt.suptitle(titles2[y] + ' alpha = 0.0', fontsize=16)
	plt.show()
"""
######################################################################################
############# Finding 1 % quantiles in mismatch distribution##########################
######################################################################################
"""
plt.suptitle('Alpha = 0.0',size=30)
#titles = ['Optimal wind layout 1 vs optimal solar layout','Optimal wind layout 2 vs optimal solar layout','Optimal wind layout 3 vs optimal solar layout','Optimal wind layout 4 vs optimal solar layout']
#titles = ['Optimal wind layout 1 vs solar reference layout','Optimal wind layout 2 vs solar reference layout','Optimal wind layout 3 vs solar reference layout','Optimal wind layout 4 vs solar reference layout']
titles = ['Wind reference layout vs solar reference layout']
for x in range(2):
	a = mismatch_energy(make_gamma_1(),make_gamma_1())[0]
	steps = 211+x
	plt.subplot(steps)
	plt.title(titles[x],size=20)
	low_quan, high_quan = integration(make_gamma_1(),make_gamma_1())
	#high_quan = integration(Gamma_wind[Optimal_wind_gamma_index[x]],Gamma_solar[Optimal_solar_gamma_index[0]])[1]
	minpoint = a[low_quan[0]]
	maxpoint = a[high_quan[0]]
	histplot = plt.hist(a,bins=100,alpha=0.2)
	histploty = histplot[0]
	histplotx = histplot[1]
	low_height = histploty[find_nearest(histplotx,minpoint)]
	high_height = histploty[find_nearest(histplotx,maxpoint)]
	plt.vlines(minpoint,0,ymax=low_height)
	plt.vlines(maxpoint,0,ymax=high_height)
	plt.vlines(min(a),0,ymax=low_height)
	plt.vlines(max(a),0,ymax=high_height)
	plt.xlabel('Mismatch (GW)',size=20)
	plt.tick_params(labelsize='large')
	maxa = round(max(a),2)
	diffmax = round(max(a)-maxpoint,2)
	quan_max = round(maxpoint,2)
	mina = round(min(a),2)
	diffmin = round(minpoint-min(a),2)
	quan_min = round(minpoint,2)
	plt.annotate('minimum \n' + str(mina),(mina-14,low_height+40))
	plt.annotate('interval \n' + str(diffmin),((mina+minpoint)/2-18,low_height-50))
	plt.annotate('1 % quantile \n' + str(quan_min),(minpoint-19,low_height+40))
	plt.annotate('maximum \n' + str(maxa),(maxa-14,high_height+40))
	plt.annotate('interval \n' + str(diffmax),((maxa+maxpoint)/2-18,high_height-50))
	plt.annotate('99 % quantile \n' + str(quan_max),(maxpoint-19,high_height+40))
	plt.axis([-500,600,0,2500])
"""
######################################################################################
############################## Comparing gammas ######################################
######################################################################################

"""
f = []
a = rolando_alphas
for x in range(30):
	#for y in range(1):
	f.append(a[x])
		
#plt.ion()
#alpha = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]		
my_colors = 'rb' #A very brilliant way of coloring every 4 bars red and blue
ind = np.arange(30)*1 + 0.25

#for x in range(1):
#	plt.bar(range(3), f[0:90],  width=0.5, bottom=0,color=my_colors)

for x in range(1):
	plt.figure()
	plt.bar(range(30), f[0:30], width=0.5, bottom=0, color=my_colors)
	#plt.xticks(ind, ('AT','BA','BE','BG','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LU','LV','NL','NO','PL','PT','RO','RS','SE','SI','SK'))
	plt.xticks(ind, ('DE','FR','GB','IT','ES','SE','PL','NO','NL','BE','FI','CZ','AT','GR','RO','BG','PT','CH','HU','DK','RS','IE','BA','SK','HR','LT','EE','SI','LV','LU'),size=8)
	plt.suptitle('Rolandos alpha values',size=20)
	plt.ylabel('Alpha of individual countries')
	#plt.savefig('Plots/Wind_only_without_load.png')
"""


######################################################################################
############################### Balancing capacity	##################################
######################################################################################
"""
f = open('Balancing Capacities quantiles 2',"a")
for x in range(4):
	mismatch = mismatch_energy(Gamma_wind[Optimal_wind_gamma_index[x]],Gamma_solar[Optimal_solar_gamma_index[0]])
	a = integration(Gamma_wind[Optimal_wind_gamma_index[x]],Gamma_solar[Optimal_solar_gamma_index[0]])
	f.write(str((mismatch[0][a[0][0]])) + '	' + str((mismatch[1][a[0][1]])) + '	' + str((mismatch[2][a[0][2]])) + '	' + str((mismatch[3][a[0][3]])) + '	' + str((mismatch[4][a[0][4]])) + '	' + str((mismatch[5][a[0][5]])) + '	' + str((mismatch[6][a[0][6]])) + '	' + str((mismatch[7][a[0][7]])) + '	' + str((mismatch[8][a[0][8]])) + '	' + str((mismatch[9][a[0][9]])) + '	' + str((mismatch[10][a[0][10]])) + '\n')

for x in range(4):	
	mismatch = mismatch_energy(Gamma_wind[Optimal_wind_gamma_index[x]],make_gamma_1())
	a = integration(Gamma_wind[Optimal_wind_gamma_index[x]],make_gamma_1())
	f.write(str((mismatch[0][a[0][0]])) + '	' + str((mismatch[1][a[0][1]])) + '	' + str((mismatch[2][a[0][2]])) + '	' + str((mismatch[3][a[0][3]])) + '	' + str((mismatch[4][a[0][4]])) + '	' + str((mismatch[5][a[0][5]])) + '	' + str((mismatch[6][a[0][6]])) + '	' + str((mismatch[7][a[0][7]])) + '	' + str((mismatch[8][a[0][8]])) + '	' + str((mismatch[9][a[0][9]])) + '	' + str((mismatch[10][a[0][10]])) + '\n')
	
mismatch = mismatch_energy(make_gamma_1(),Gamma_solar[Optimal_solar_gamma_index[0]])
a = integration(make_gamma_1(),Gamma_solar[Optimal_solar_gamma_index[0]])
f.write(str((mismatch[0][a[0][0]])) + '	' + str((mismatch[1][a[0][1]])) + '	' + str((mismatch[2][a[0][2]])) + '	' + str((mismatch[3][a[0][3]])) + '	' + str((mismatch[4][a[0][4]])) + '	' + str((mismatch[5][a[0][5]])) + '	' + str((mismatch[6][a[0][6]])) + '	' + str((mismatch[7][a[0][7]])) + '	' + str((mismatch[8][a[0][8]])) + '	' + str((mismatch[9][a[0][9]])) + '	' + str((mismatch[10][a[0][10]])) + '\n')

mismatch = mismatch_energy(make_gamma_1(),make_gamma_1())
a = integration(make_gamma_1(),make_gamma_1())
f.write(str((mismatch[0][a[0][0]])) + '	' + str((mismatch[1][a[0][1]])) + '	' + str((mismatch[2][a[0][2]])) + '	' + str((mismatch[3][a[0][3]])) +'	' + str((mismatch[4][a[0][4]])) + '	' + str((mismatch[5][a[0][5]])) + '	' + str((mismatch[6][a[0][6]])) + '	' + str((mismatch[7][a[0][7]])) + '	' + str((mismatch[8][a[0][8]])) + '	' + str((mismatch[9][a[0][9]])) + '	' + str((mismatch[10][a[0][10]])) + '\n')

f.close()
"""

######################################################################################
###################### Plotting scatterplots with optimal layouts ####################
######################################################################################

"""
colors = ['ro','yo','go','wo']
plt.plot(Optimal_std_wind_with_load[0],Optimal_cf_wind_with_load[0],colors[0],markersize=9)
plt.plot(Optimal_std_wind_with_load[1],Optimal_cf_wind_with_load[1],colors[1],markersize=9)
plt.plot(Optimal_std_wind_with_load[2],Optimal_cf_wind_with_load[2],colors[2],markersize=9)
plt.plot(Optimal_std_wind_with_load[3],Optimal_cf_wind_with_load[3],colors[3],markersize=9)
colors2 = ['co','wo']
plt.plot(Optimal_std_solar_with_load[0],Optimal_cf_solar_with_load[0],colors2[0],markersize=9)
plt.plot(Optimal_std_solar_with_load[1],Optimal_cf_solar_with_load[1],colors2[1],markersize=9)
plt.legend(['Wind Layout 1','Wind Layout 2','Wind layout 3','Wind Reference layout','Solar Layout','Solar reference layout'])
plt.plot(Std_solar_with_load,Cf_solar,'bo')
plt.plot(Std_wind_with_load,Cf_wind,'bo')
colors = ['ro','yo','go','wo']
plt.plot(Optimal_std_wind_with_load[0],Optimal_cf_wind_with_load[0],colors[0],markersize=9)
plt.plot(Optimal_std_wind_with_load[1],Optimal_cf_wind_with_load[1],colors[1],markersize=9)
plt.plot(Optimal_std_wind_with_load[2],Optimal_cf_wind_with_load[2],colors[2],markersize=9)
plt.plot(Optimal_std_wind_with_load[3],Optimal_cf_wind_with_load[3],colors[3],markersize=9)
colors2 = ['co','wo']
plt.plot(Optimal_std_solar_with_load[0],Optimal_cf_solar_with_load[0],colors2[0],markersize=9)
plt.plot(Optimal_std_solar_with_load[1],Optimal_cf_solar_with_load[1],colors2[1],markersize=9)
plt.title('Comparison between the scatterplots for only wind and for only solar (with load)')
plt.xlabel('Standard deviation of mismatch (GW)')
plt.ylabel('Capacity factor')
plt.show()
"""
