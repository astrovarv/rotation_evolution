# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 2020

@author: Varvara Semenova

"""
import Evolution

def collect_test_data(masses, period, period_crit, b_crit):
    M, Pinit, Bcrit, Pcrit, Myr150, Myr650, Myr3000 = [], [], [], [], [], [], []
    Myr150_contr, Myr650_contr, Myr3000_contr= [], [], []
    n = 0
    for i in range(len(masses)):
        for j in range(len(period)):
            for k in range(len(period_crit)):
                for l in range(len(b_crit)):
                    
                    print('M:', masses[i]/M_sun, 'Pinit:', period[j], \
                          'Pcrit:', period_crit[k], 'Bcrit:', b_crit[l])
                        
                    #initialise the evolution
                    ev = Full_Evolution(masses[i], period[j], period_crit[k], \
                                        b_crit[l])
                    ev.evolve()
                    
                    Myr150.append(ev.get_period_at(150*10**6))
                    Myr650.append(ev.get_period_at(650*10**6))
                    Myr3000.append(ev.get_period_at(3000*10**6))
                    
                    Myr150_contr.append(ev.get_period_at(150*10**6, True))
                    Myr650_contr.append(ev.get_period_at(650*10**6, True))              
                    Myr3000_contr.append(ev.get_period_at(3000*10**6, True))              
                    
                    M.append(ev.mass0/M_sun)
                    Pinit.append(ev.period0)
                    Bcrit.append(ev.B_crit)
                    Pcrit.append(ev.P_crit)
                    
                    print('Period (only contraction)', ev.get_period_at(3000*10**6, True))
                    print('Period (with outflow)', ev.get_period_at(3000*10**6))
                    
                    n += 1
                    print(n/18*100, '%')
         
    data = {'M/Ms' : M,
            'P_init' : Pinit,
            'B_crit' : Bcrit,
            'P_crit' : Pcrit,
            'P_after150Myr' : Myr150,
            'P_after650Myr' : Myr650,
            'P_after3000Myr' : Myr3000,
            'P_after150Myr_contr' : Myr150_contr,
            'P_after650Myr_contr' : Myr650_contr,
            'P_after3000Myr_contr' : Myr3000_contr,
            }
    
    df = pd.DataFrame(data)
    
    return df


def example_plot_const_B(B_crit):
    
    df = pd.read_csv('samples_n10^12.csv')
    
    fig, ax = plt.subplots(3,3, sharey = True, sharex = True, figsize = (9,9))
    ax[0,0].set_title('5 Myr')
    ax[0,1].set_title('650 Myr')
    ax[0,2].set_title('3000 Myr')
    ax[0,0].invert_xaxis()   
    
    for i in range(len(b_crit)):
        
        df1 = df[(df['B_crit'] == B_crit) & (df['P_crit'] == period_crit[i])]
        
        ax[i,0].text(1,10.8, 'Bcrit = {}kG \nPcrit = {}days'.format(B_crit/1000, period_crit[i]))
        ax[i,0].semilogy(df1['M/Ms'], df1['P_init'], '.k', alpha = 0.5)
        ax[i,0].set_xlabel(r'Mass $(M_{\odot})$')
        ax[i,0].set_ylabel('Period (days)')
        ax[i,0].set_ylim((0.01, 100))
        
        ax[i,1].semilogy(df1['M/Ms'], df1['P_after650Myr'], '.k', alpha = 0.5)
        ax[i,1].semilogy(df1['M/Ms'], df1['P_after650Myr_contr'], '.r', \
                         alpha = 0.5, label = 'Contraction only')
        ax[i,1].set_xlabel(r'Mass $(M_{\odot})$')
        
        ax[i,2].semilogy(df1['M/Ms'], df1['P_after3000Myr'], '.k')
        ax[i,1].semilogy(df1['M/Ms'], df1['P_after3000Myr_contr'], '.r', \
                         alpha = 0.5, label = 'Contraction only')
        ax[i,2].set_xlabel(r'Mass $(M_{\odot})$')
    
    plt.show()
    
    
masses = [0.3*M_sun, 1*M_sun]
period = [1,10]
period_crit = [3, 7, 8.5] # days
b_crit = [10**4] # Gauss


df = collect_test_data(masses, period, period_crit, b_crit)
df.to_csv('samples_n10^12.csv', index = False)
