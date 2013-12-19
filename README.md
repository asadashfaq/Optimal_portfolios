Optimal_portfolios
==================

My master thesis project - portfolio theory

I use three different files for the most part (Main_wind, Main_solar and Main_wind_and_solar) and just out-comment things I don't need. 
I was given the ISET data-files. From those I calculated standard deviation, capacity factors, gamma values, 1 % quantiles and variance of variabilities for solar and wind and saved these to text-files

#############################################################################################
##################################### Codes #################################################
#############################################################################################

Main_wind: 
- Focuses on Wind (duh!)
- Loads ISET-data and the calculated gamma-values
- Calculates Standard deviation and capacity factor
- Finds the optimal layouts in the interval
- Calculates mismatch and variability for wind
- Plots variability histogram, variability variance and balancing energy as function of alpha.
- Plots balancing energy and compare gamma values
- Saves standard deviation, capacity factor and gamma values

Main_solar: 
- Focuses on Solar
- Loads ISET-data and the calculated gamma-values
- Calculates standard deviation and capacity factor
- Find the optimal layouts in the interval
- Calculates mismatch and variability for wind
- Plots variability histogram, variability variance and balancing energy as function of alpha
- Plots balancing energy and compare gamma values
- Saves standard deviation, capacity factor and gamma values

Main_wind_and_solar:
- Focuses on Solar and Wind combined
- Loads ISET-data and the calculated gamma-values
- Calculates standard deviation and capacity factor
- Find the optimal layouts in the interval
- Calculates mismatch and variability for wind and solar
- Plots variability histogram, variability variance and balancing energy as function of alpha
- Plots balancing energy and compare gamma values
- Saves standard deviation, capacity factor and gamma values

Code to pick out points: 
- Loads the gamma-values and plots them
- Can press them to get the gamma-values for an individual point.

Loading_stuff: 
- Loads the calculated gamma-values
- Plots things (this code is mostly used if I want to load+plot something without having to change one of the main codes

#############################################################################################
##################################### Data-files ############################################
#############################################################################################

Capacity_values02:
- Installed wind capacity for gamma between 0 and 2 (250.000 lines)

Gamma_values02:
- Gamma values for wind for gamma between 0 and 2 (7.500.000 lines)

Optimal_values02:
- Every second line is standard deviation for wind and every second line is capacity factor for wind for gamma between 0 and 2 (500.000 lines) 

Solar_Capacity_values:
- Installed solar capacity for gamma between 0 and 2 (250.000 lines)

Solar_Gamma_values:
- Gamma values for solar for gamme between 0 and 2 (7.500.000 lines)

Solar_Optimal_values:
- Every second line is standard deviation for solar and every second line is capacity factor for solar for gamme between 0 and 2 (500.000 lines)

Optimal_quantiles:
- 1 % quantiles for wind (250.000 lines)

Optimal_quantiles_solar:
- 1 % quantiles for solar (250.000 lines)

var1,var2,var3,var6:
- Variance of the variability for wind (250.000 lines)

sol_var1,sol_var2,sol_var3,sol_var6:
- Variance of the variability for solar (250.000 lines)

Solar_optimal_gammas:
- The index of the optimal layouts for solar (5 lines)

Wind_optimal_gammas:
- The index of the optimal layouts for wind (4 lines)



