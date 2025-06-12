# Discovery_of_soild_consititutive
The data and codes for the paper "Graphic-symbolic regression for constitutive law discovery in solids: towards generalizable and interpretable data-driven modeling"

# Environments
python==3.12.4   
pandas==2.2.2  
numpy==1.26.4 
pytorch==2.3.1 
scikit-learn==1.4.2  
scipy==1.13.1 
matplotlib==3.8.4
sympy==1.13.3
openpyxl==3.1.5
torch-geometric==2.6.1


# The dataset utilized in this work 
* [data_DIF]: data for DIF constitutive model, including 40 different materials under varying strain rates were compiled from 18 published studies.  
  -In each data, the strain rate and DIF is recorded  
* [data_strain_stress]: data for strain hardening constitutive model. In each data, the true plastic strain and true plastic stress is recorded.
* [saved_data_hardening_strain_rate]: data for combined DIF and strain hardening for the discovery of integrated model.
  

# How to reproduce our work?
1. Install relevant packages (~20 min)  
2. [Discovery of DIF constitutive model] Run the Discovery_of_DIF.py  
   [Discovery of strain hardening constitutive model] Run the Discovery_of_strain_harderning.py  
   -model=='Train': Discover underlying equation from experimental data by graphic-symbolic regression, about 40 min for parallel computing and several hours for not parallel computing      
   Note: set self.use_parallel_computing=True for parallel computing.    
   -model=='Valid': Test the performance of the discovered equation  
   -model=='Valid discovered': Show the optimization process of graphic-symbolic regression with the top-5 equations in each epoch.  
3. [Discovery of integrated constitutive model] Run the Discovery_of_hardening_strain_rate.py to test the integrated consititutive model.  
   -model=='Valid': use the integrated model or modified model to predict the strain-stress curve under different strain rates.


# Expected outputs
* All outputs will be saved in the dir result_save  
* The discovered equations and corresponding awards are saved in result save, including best_fitness.pkl, best_graph.pkl which records the best 5 PDE structures in each optimization epoch.
  (The results is provided in this repository)
