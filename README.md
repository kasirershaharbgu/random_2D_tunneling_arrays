
# random_2D_tunneling_arrays 
This repository includes python code for running Gillespie (Kinetic Monte-Carlo) simulations for 2D random arrays. 
Also included:  - result examples
                - example running scripts on Slurm and Qsub servers
                - Python script for fast results processing
                
This code is free to use, you are more than welcom to use this code for any reason (see LICENSE.md).

Running instructions
--------------------
Usage: random_2d_array_simulation.py [options]


Options:

  -h, --help           
  show this help message and exit
  
  -T T, --temperature=T
                       
                        Environment temperature (in units of
                        planckConstant/timeUnits) [default: 0]
                        
  --temperature-gradient=TEMPERATURE_GRADIENT
                        
                        Temperature gradient (in units of
                        planckConstant/(timeUnits*Lattice constant)) [default:
                        0]
                        
  --gap=GAP             
                        
                        superconducting gap (in units of
                        planckConstant/timeUnits)[default: 0]
                        
  -M M, --height=M      
                        
                        number of lines in the array [default: 1]
  
  -N N, --width=N       
                        
                        number of columns in the array [default: 1]
  
  --vr=VR               
                        
                        right electrode voltage (in units of
                        planckConstant/electronCharge*timeUnits) [default: 0]
                        
  --vu=VU               
                        
                        upper electrode voltage (in units of
                        planckConstant/electronCharge*timeUnits) [default: 0]
                        
  --vd=VD               
                        
                        lower electrode voltage (in units of
                        planckConstant/electronCharge*timeUnits) [default: 0]
                        
  --right-electrode=RIGHTELECTRODE
                        
                        Location of right electrode in form of a binary array,
                        i.e. if the array height is 7 and the electrode is
                        connected in the second and fifth rows then
                        [0,1,0,0,1,0,0] [default: connected to all rows]
                        
  --left-electrode=LEFTELECTRODE
                        
                        Location of left electrode in form of a binary array,
                        i.e. if the array height is 7 and the electrode is
                        connected in the second and fifth rows then
                        [0,1,0,0,1,0,0] default: connected to all rows]
                        
  --up-electrode=UPELECTRODE
                        
                        Location of upper electrode in form of a binary array,
                        i.e. if the array width is 5 and the electrode is
                        connected in the second and fifth rows then
                        [0,1,0,0,1] [default: connected to all columns]
                        
  --down-electrode=DOWNELECTRODE
                       
                       Location of lower electrode in form of a binary array,
                        i.e. if the array width is 5 and the electrode is
                        connected in the second and fifth rows then
                        [0,1,0,0,1] [default: connected to all columns]
                        
  --vmin=VMIN           
                        
                        minimum external voltage  (in units of
                        planckConstant/electronCharge*timeUnits) [default: 0]
                        
  --vmax=VMAX           
                        
                        maximum external voltage  (in units of
                        planckConstant/electronCharge*timeUnits) [default: 10]
                        
  --vstep=VSTEP         
                        
                        size of voltage step  (in units of
                        planckConstant/electronCharge*timeUnits)[default: 1]
                        
  --symmetric-v         
                        
                        Voltage raises symmetric on VR and VL[default: False]
  
  --repeats=REPEATS     
                        
                        how many times to run calculation for averaging
                        [default: 1]
                        
  --file-name=FILENAME  
                      
                      optional output files name
  
  --distribution=DIST   
                      
                      probability distribution to use [Default:uniform]
  
  --full                
                      
                      if true the results n and Q will be also saved
                        [Default:False]
                        
  --graph               
                        
                        if true a simulation using graph solution for master
                        equationwill be used [Default:False]
                        
  --current-map         
                        
                        if true fraes for clip of current distribution during
                        simulation will be created and saved [Default:False]
                        
  --plot-current-map    
                      
                        if true clip of current distribution will be plotted
                        using former saved frames (from a former run with same
                        file name and location and the flag --current-map
                        [Default:False]
                        
  --plot-binary-current-map
                        
                        if true a binary clip of current distribution will be
                        plotted using former saved frames (from a former run
                        with same file name and location and the flag
                        --current-map [Default:False]
                        
  --frame-norm          
                      
                      if true the clip of current distribution will be
                        normalized per frame [Default:False]
                        
  --dbg                 
                    
                    Avoids parallel running for debugging [Default:False]
  
  --resume              
                  
                  Resume failed run from last checkpoint [Default:False]
  
  --superconducting     
                  
                  use superconducting array [Default:False]
  
  --tau-leaping         
                  
                  use tau leaping approximation [Default:False]
  
  --const-q             
                
                calc current using const Q method [Default:False]
  
  --variable-ef         
                  
                  if true Fermi energy level for each island will be
                        changed according to constant density of states
                        assumption, else it will be assumed constant (infinite
                        density of states [Default:False]
                        
  --double-time         
                    
                    if true each simulation step will run twice as long as
                        the default time [Default:False]
                        
  --double-loop         
                      
                      if true the voltage would be raised and lowered twice
                        [Default:False]
                        
  --calc-it             
                  
                  Instead of calculating IV curve calculates current as
                        a function of the temperature, in this case Vmax,
                        Vstep would be used as Tmax, Tstep instead
                        [Default:False]
                        
  -o OUTPUT_FOLDER, --output-folder=OUTPUT_FOLDER
                        
                        Output folder [default: current folder]
                        
  -i PARAMS_PATH, --load-from-file=PARAMS_PATH
                        
                        If a parameters file is given all the array parameters
                        would be loaded from that file. The file should be in
                        the same format as the resulted parameter file for a
                        run. Ignores all other array related parameters, so
                        the array owuld be exactly as specified in the given
                        file. [default: '']
                        
  --vg-avg=VG_AVG       
                    
                    Gate voltage average  (in units of
                        planckConstant/electronCharge*timeUnits) [default: 1]
                        
  --vg-std=VG_STD       
                    
                    Gate voltage std  (in units of
                        planckConstant/electronCharge*timeUnits) [default: 0]
                        
  --c-avg=C_AVG         
                    
                    capacitance of junctions average (in units of
                        timeUnits*electronCharge^2/planckConstant) [default:
                        1]
                        
  --c-std=C_STD         
                      
                      capacitance of junctions std (in units of
                        timeUnits*electronCharge^2/planckConstant) [default:
                        0]
                        
  --cg-avg=CG_AVG       
                      
                      Gate Capacitors capacitance average (in units of
                        timeUnits*electronCharge^2/planckConstant) [default:
                        1]
                        
  --cg-std=CG_STD      
                      
                      Gate Capacitors capacitance std (in units of
                        timeUnits*electronCharge^2/planckConstant) [default:
                        0]
                        
  --r-avg=R_AVG         
                      
                      junctions resistance average (in units of
                        planckConstant/electronCharge^2) [default: 1]
                        
  --r-std=R_STD         
                        
                        junctions resistance std (in units of
                        planckConstant/electronCharge^2) [default: 0]
                        
  --custom-rh=CUSTOM_RH
                        
                        list of r horizontal values ordered as numpy array.
                        Overrides random r parameters [default: ]
                        
  --custom-rv=CUSTOM_RV
                        
                        list of r vertical values ordered as numpy array.
                        Overrides random r parameters [default: ]
                        
  --custom-ch=CUSTOM_CH
                        
                        list of c horizontal values ordered as numpy array.
                        Overrides random c parameters [default: ]
                        
  --custom-cv=CUSTOM_CV
                        
                        list of c vertical values ordered as numpy array.
                        Overrides random c parameters [default: ]
                        
  --rg-avg=RG_AVG       
                        
                        Gate Resistors resistance average (in units of
                        planckConstant/electronCharge^2) [default: 1]
                        
  --rg-std=RG_STD       
                        
                        Gate Resistors resistance std (in units of
                        planckConstant/electronCharge^2) [default: 0]
                        
  --n-avg=N0_AVG        
                        
                        initial number of electrons on each dot average
                        [default:0]
                        
  --n-std=N0_STD       
                        
                        initial number of electrons on each dot std
                        [default:0]
                        
  --q-avg=Q0_AVG        
                        
                        initial charge on gate capacitors average (in units of
                        electronCharge) [default:0]
                        
  --q-std=Q0_STD          
                        
                        initial charge on gate capacitors std (in units of
                        electronCharge) [default:0]
