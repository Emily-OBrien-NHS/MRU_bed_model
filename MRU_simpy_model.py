import simpy
import random
import math
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

class default_params():
    run_name = '6hr MRU LoS, 22 beds'
    #run times and iterations
    run_time = 525600
    run_days = int(run_time/(60*24))
    iterations = 10
    occ_sample_time = 60
    #arrivals
    sdmart_engine = create_engine('mssql+pyodbc://@SDMartDataLive2/InfoDB?'\
                              'trusted_connection=yes&driver=ODBC+Driver+17'\
                              '+for+SQL+Server')
    arrivals_query = """
    select [Ward Stay Start Datetime] = sstay_start_dttm,
    [Ward Stay End Datetime]	= sstay_end_dttm,
    [ED Arrival Datetime]		= ArrivalDateTime,
    [ED Departure Datetime]	= DischargeDateTime
		
    from [pimsmarts].[dbo].[ip_movements] as IP_MO
    Inner join [infodb].[dbo].[vw_ipdc_fces_pfmgt] as INPAT on INPAT.prvsp_refno = IP_MO.prvsp_refno --Inner join, as I only want the matches
    Left join [PiMSMarts].[dbo].[cset_admet] as ADMET on ADMET.identifier = INPAT.admet	--Just to get the admet description
    Left join [CL3-data].[DataWarehouse].[ed].[vw_EDAttendance]	as EDATN on EDATN.admitprvsprefno = IP_MO.prvsp_refno

    Where sstay_start_dttm >=	'01/08/2024' --Start of window
    and sstay_start_dttm <=	'31/03/2025 23:59:59' --End of window
    and move_reason_sp = 'S' --Ward stay not consultant change
    and INPAT.admit_dttm = IP_MO.sstay_start_dttm --Ensuring 1st ward
    and INPAT.admit_dttm = INPAT.start_dttm	--Ensuring vw_ipdc_fces_pfmgt is reduced down to spell level rather than episode level
    and	IP_MO.sstay_ward_code in ('RK950116', 'RK950AMW', 'RK950MAU') --Thrushel - MAU, Zone A, Level 6
    and	(INPAT.admit_dttm >= ArrivalDateTime or ArrivalDateTime is null) --Bringing in the ED view introduced duplicates, as per the validation below. 
    """
    arrivals = pd.read_sql(arrivals_query, sdmart_engine)
    #Get ED inter arrival time by hour of day
    arrivals['ED Hour'] = arrivals['ED Arrival Datetime'].dt.hour
    arrivals['ED Date'] = arrivals['ED Arrival Datetime'].dt.date
    mean_ED_arr = (60 /
                   (arrivals.groupby([arrivals['ED Date'], arrivals['ED Hour']],
                                     dropna=True, as_index=False)
                                     ['ED Arrival Datetime'].count()
                            .groupby(arrivals['ED Hour'])['ED Arrival Datetime']
                            .mean()))
    #Get External inter arrival time by hour of day
    arrivals['EXT Hour'] = arrivals['Ward Stay Start Datetime'].dt.hour
    arrivals['EXT Date'] = arrivals['Ward Stay Start Datetime'].dt.date
    mean_EXT_arr = (60 /
                    (arrivals.loc[arrivals['ED Arrival Datetime'].isna()]
                    .groupby(['EXT Date', 'EXT Hour'], as_index=False)
                             ['Ward Stay Start Datetime'].count()
                    .groupby('EXT Hour')['Ward Stay Start Datetime'].mean()))

    #LoS
    mean_ED_los = 120
    mean_MRU_los = 60*6
    #resources
    no_MRU_beds = 22#np.inf
    #lists for storing results
    pat_res = []
    occ_res = []

class spawn_patient:
    def __init__(self, p_id):
        self.id = p_id
        self.arrival = ''
        self.ED_arrival_time = np.nan
        #self.ED_leave_time = np.nan
        self.MRU_wait_start_time = np.nan
        self.MRU_arrival_time = np.nan
        self.MRU_leave_time = np.nan

class mru_model:
    def __init__(self, run_number, input_params):
        self.patient_results = []
        self.mru_occupancy_results = []
        #start environment, set patient counter to 0 and set run number
        self.env = simpy.Environment()
        self.input_params = input_params
        self.patient_counter = 0
        self.run_number = run_number
        #establish resources
        self.MRU_bed = simpy.Resource(self.env,
                                      capacity=input_params.no_MRU_beds)
        self.ED = simpy.Resource(self.env, capacity = np.inf)
    
    ########################ARRIVALS################################
    def ED_arrivals(self):
        while True:
            #up patient counter and spawn a new walk-in patient
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter)
            p.arrival = 'ED'
            #begin patient ED process
            self.env.process(self.ED_to_MRU_journey(p))
            #randomly sample the time until the next patient arrival
            time_of_day = math.floor(self.env.now % (60*24) / 60)
            ED_arr = self.input_params.mean_ED_arr[time_of_day]
            sampled_interarrival = random.expovariate(1.0 / ED_arr)
            yield self.env.timeout(sampled_interarrival)
    
    def External_arrivals(self):
        while True:
            #up patient counter and spawn a new walk-in patient
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter)
            p.arrival = 'External'
            #begin patient ED process
            self.env.process(self.MRU_journey(p))
            #randomly sample the time until the next patient arrival
            time_of_day = math.floor(self.env.now % (60*24) / 60)
            EXT_arr = self.input_params.mean_EXT_arr[time_of_day]
            sampled_interarrival = random.expovariate(1.0 / EXT_arr)
            yield self.env.timeout(sampled_interarrival)
    
    ##################ED TO MRU PROCESS #########################

    def MRU_journey(self, patient):
        #Enter MRU
        patient.MRU_wait_start_time = self.env.now
        with self.MRU_bed.request() as req:
            yield req
            patient.MRU_arrival_time = self.env.now
            sampled_MRU_time = random.expovariate(1.0
                                                / self.input_params.mean_MRU_los) 
            yield self.env.timeout(sampled_MRU_time)
        patient.MRU_leave_time = self.env.now
        self.store_patient_results(patient)

    def ED_to_MRU_journey(self, patient):
        #Patient comes into ed
        patient.ED_arrival_time = self.env.now 
        with self.ED.request() as req:
            yield req
            #randomly sample the time spent in ED
            sampled_ED_time = min(random.expovariate(1.0
                                                / self.input_params.mean_ED_los), 240)
            yield self.env.timeout(sampled_ED_time)
        #patient.ED_leave_time = self.env.now

        #Enter MRU
        patient.MRU_wait_start_time = self.env.now
        with self.MRU_bed.request() as req:
            yield req
            patient.MRU_arrival_time = self.env.now
            sampled_MRU_time = random.expovariate(1.0
                                                / self.input_params.mean_MRU_los) 
            yield self.env.timeout(sampled_MRU_time)
        patient.MRU_leave_time = self.env.now
        self.store_patient_results(patient)
###################RECORD RESULTS####################
    def store_patient_results(self, patient):
        self.patient_results.append([self.run_number, patient.id,
                                     patient.arrival,
                                     patient.ED_arrival_time,
                                     #patient.ED_leave_time,
                                     patient.MRU_wait_start_time,
                                     patient.MRU_arrival_time,
                                     patient.MRU_leave_time])
    
    def store_occupancy(self):
        while True:
            self.mru_occupancy_results.append([self.run_number,
                                               self.ED._env.now,
                                               self.ED.count,
                                               len(self.MRU_bed.queue),
                                               self.MRU_bed.count])
            yield self.env.timeout(self.input_params.occ_sample_time)
########################RUN#######################
    def run(self):
        self.env.process(self.ED_arrivals())
        self.env.process(self.External_arrivals())
        self.env.process(self.store_occupancy())
        self.env.run(until = self.input_params.run_time)
        default_params.pat_res += self.patient_results
        default_params.occ_res += self.mru_occupancy_results
        return self.patient_results, self.mru_occupancy_results

def export_results(run_days, pat_results, occ_results):
    patient_df = pd.DataFrame(pat_results,
                              columns=['Run', 'Patient ID', 'Arrival Method',
                                       'ED Arrival Time', #'ED Leave Time', 
                                       'MRU Wait Start Time',
                                       'MRU Arrival Time', 'MRU Leave Time'])
    patient_df['Simulation Arrival Time'] = (patient_df['ED Arrival Time']
                                             .fillna(patient_df['MRU Wait Start Time']))
    patient_df['Simulation Arrival Day'] = pd.cut(
                           patient_df['Simulation Arrival Time'], bins=run_days,
                           labels=np.linspace(1, run_days, run_days))
    patient_df['Wait for MRU Bed Time'] = (patient_df['MRU Arrival Time']
                                           - patient_df['MRU Wait Start Time'])
    patient_df['Simulation Leave Day'] = pd.cut(
                                    patient_df['MRU Leave Time'], bins=run_days,
                                    labels=np.linspace(1, run_days, run_days))
    patient_df['Simulation Leave Hour'] = (patient_df['MRU Leave Time']
                                           / 60).round().astype(int)

    
    occupancy_df = pd.DataFrame(occ_results,
                                columns=['Run', 'Time', 'ED Occupancy',
                                'MRU Bed Queue', 'MRU Occupancy'])
    occupancy_df['day'] = pd.cut(occupancy_df['Time'], bins=run_days,
                                 labels=np.linspace(1, run_days, run_days))
    return patient_df, occupancy_df

def run_the_model(input_params):
    #run the model for the number of iterations specified
    #for run in stqdm(range(input_params.iterations), desc='Simulation progress...'):
    for run in range(input_params.iterations):
        print(f"Run {run+1} of {input_params.iterations}")
        model = mru_model(run, input_params)
        model.run()
    patient_df, occ_df = export_results(input_params.run_days,
                                        input_params.pat_res,
                                        input_params.occ_res)
    return patient_df, occ_df

###############################################################################
#Run and save results
pat, occ = run_the_model(default_params)
pat.to_csv(f'C:/Users/obriene/Projects/MRU/Simpy Model/Results/Full Outputs/patients - {default_params.run_name}.csv')
occ.to_csv(f'C:/Users/obriene/Projects/MRU/Simpy Model/Results/Full Outputs/occupancy - {default_params.run_name}.csv')

####MRU leavers plot
font_size = 24
MRU_discharges = (pat.groupby(['Run', 'Simulation Leave Hour'], as_index=False)
                  ['Patient ID'].count()
                  .groupby('Simulation Leave Hour').mean()
                  ['Patient ID'])
MRU_discharges.columns = ['Hour', 'Patients Leaving MRU']
daily_av = MRU_discharges.groupby((MRU_discharges.index/24).round()).mean()
daily_av.index = daily_av.index * 24

fig, axs = plt.subplots(figsize=(25, 10))
axs.plot(MRU_discharges.index, MRU_discharges, color='grey', alpha=0.3, label='Hourly Leavers')
axs.plot(daily_av.index, daily_av, 'r-', label='Daily Average Hourly Leavers')
axs.set_title(f'Patients Leaving MRU per Hour - {default_params.run_name}', fontsize=font_size)
axs.set_xlabel('Hour', fontsize=font_size)
axs.set_ylabel('Patients Leaveing MRU', fontsize=font_size)
axs.legend()
axs.tick_params(axis='both',  which='major', labelsize=font_size)
plt.savefig(f'C:/Users/obriene/Projects/MRU/Simpy Model/Results/Patients Leaving MRU - {default_params.run_name}.svg', bbox_inches='tight', dpi=1200)
plt.close()

####Occupancy plot
mean_occupancy = (occ.groupby(['Run', 'Time'], as_index=False)
                 [['ED Occupancy', 'MRU Occupancy']].mean().groupby('Time')
                 [['ED Occupancy', 'MRU Occupancy']].mean()).round().astype(int)
mean_occupancy.columns = ['Mean ED Occupancy', 'Mean MRU Occupancy']
max_occupancy = (occ.groupby(['Run', 'Time'], as_index=False)
                 [['ED Occupancy', 'MRU Occupancy']].max().groupby('Time')
                 [['ED Occupancy', 'MRU Occupancy']].max())
max_occupancy.columns = ['Max ED', 'Max MRU']
min_occupancy = (occ.groupby(['Run', 'Time'], as_index=False)
                 [['ED Occupancy', 'MRU Occupancy']].min().groupby('Time')
                 [['ED Occupancy', 'MRU Occupancy']].min())
min_occupancy.columns = ['Min ED', 'Min MRU']
occupancy = min_occupancy.join(mean_occupancy).join(max_occupancy)

####ED
fig, axs = plt.subplots(figsize=(25, 10))
axs.plot(occupancy.index, occupancy['Mean ED Occupancy'], '-r')
axs.fill_between(occupancy.index, occupancy['Min ED'], occupancy['Max ED'], color='grey', alpha=0.2)
axs.set_title(f'Average Number of Patients in ED to go to MRU - {default_params.run_name}', fontsize=font_size)
axs.set_xlabel('Time (Mins)', fontsize=font_size)
axs.set_ylabel('No. Patients', fontsize=font_size)
axs.tick_params(axis='both',  which='major', labelsize=font_size)
plt.savefig(f'C:/Users/obriene/Projects/MRU/Simpy Model/Results/ED Occupancy - {default_params.run_name}.svg', bbox_inches='tight', dpi=1200)
plt.close()

#MRU
fig, axs = plt.subplots(figsize=(25, 10))
axs.plot(occupancy.index, occupancy['Mean MRU Occupancy'], '-r')
axs.fill_between(occupancy.index, occupancy['Min MRU'], occupancy['Max MRU'], color='grey', alpha=0.2)
axs.set_title(f'Average Number of Patients in MRU - {default_params.run_name}', fontsize=font_size)
axs.set_xlabel('Time (Mins)', fontsize=font_size)
axs.set_ylabel('No. Patients', fontsize=font_size)
axs.tick_params(axis='both',  which='major', labelsize=font_size)
plt.savefig(f'C:/Users/obriene/Projects/MRU/Simpy Model/Results/MRU Occupancy - {default_params.run_name}.svg', bbox_inches='tight', dpi=1200)
plt.close()

#### MRU Bed queue
queue = occ.groupby('day')['MRU Bed Queue'].mean()
fig, axs = plt.subplots(figsize=(25, 10))
axs.plot(queue.index, queue)
axs.set_title(f'Average Number of Patients in MRU Queue - {default_params.run_name}', fontsize=font_size)
axs.set_xlabel('Simulation Day', fontsize=font_size)
axs.set_ylabel('No. Patients', fontsize=font_size)
axs.tick_params(axis='both',  which='major', labelsize=font_size)
plt.savefig(f'C:/Users/obriene/Projects/MRU/Simpy Model/Results/Average Number of Patients in MRU Queue - {default_params.run_name}.svg', bbox_inches='tight', dpi=1200)
plt.close()

#### MRU Bed Wait Time
wait_for_bed = pat.groupby('Simulation Arrival Day')['Wait for MRU Bed Time'].mean() / 60
fig, axs = plt.subplots(figsize=(25, 10))
axs.plot(wait_for_bed.index, wait_for_bed)
axs.set_title(f'Average Hours Waiting for MRU Bed - {default_params.run_name}', fontsize=font_size)
axs.set_xlabel('Simulation Day', fontsize=font_size)
axs.set_ylabel('Hours Waited', fontsize=font_size)
axs.tick_params(axis='both',  which='major', labelsize=font_size)
plt.savefig(f'C:/Users/obriene/Projects/MRU/Simpy Model/Results/Average Wait for Bed Time - {default_params.run_name}.svg', bbox_inches='tight', dpi=1200)
plt.close()