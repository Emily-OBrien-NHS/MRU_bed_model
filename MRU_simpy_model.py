import re
import simpy
import random
import math
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from uhpt_population_projection import age_bands


class default_params():
    run_name = 'MRU new build'
    #run times and iterations
    start_year = datetime.today().year
    run_years = 10
    run_time = run_years * (365 * (60*24))
    #run_time = 525600
    iterations = 100
    occ_sample_time = 60
    #arrivals
    sdmart_engine = create_engine('mssql+pyodbc://@SDMartDataLive2/InfoDB?'\
                                  'trusted_connection=yes&driver=ODBC+Driver+17'\
                                  '+for+SQL+Server')
    start_dttm = '01/08/2024'
    end_dttm = '31/03/2025 23:59:59'
    arrivals_query = f"""SET NOCOUNT ON
    SELECT *
    INTO #MRU
    FROM
        (SELECT IPMOV.prvsp_refno, sstay_start_dttm, sstay_end_dttm, [pat_age_on_admit],
            ROW_NUMBER() OVER (PARTITION BY IPMOV.prvsp_refno ORDER BY IPMOV.sstay_start_dttm ASC) AS rn
        FROM  SDMARTDATALIVE2.pimsmarts.dbo.ip_movements   AS IPMOV
        LEFT JOIN SDMARTDATALIVE2.[infodb].[dbo].[vw_ipdc_fces_pfmgt] AS INPAT ON INPAT.prvsp_refno = IPMOV.prvsp_refno
        LEFT JOIN SDMARTDATALIVE2.PiMSMarts.dbo.cset_admet AS ADMET ON ADMET.identifier = INPAT.admet
        WHERE IPMOV.move_reason_sp  = 'S' --Ward stay not consultant change
        AND INPAT.last_episode_in_spell = '1'
        AND IPMOV.sstay_end_dttm IS NOT NULL 
        AND IPMOV.sstay_ward_code IN ('RK950116', 'RK950AMW', 'RK950MAU')  --Medical Receiving Unit - Tamar Ward, Zone B, Level 6
        AND sstay_start_dttm >=	'{start_dttm}' --Start of window
        AND sstay_start_dttm <=	'{end_dttm}') MRU
    WHERE rn = 1

    SELECT [Ward Stay Start Datetime] = sstay_start_dttm,
    [Ward Stay End Datetime] = sstay_end_dttm,
    [pat_age_on_admit] AS Age,
    [ED Arrival Datetime] = ArrivalDateTime,
    [ED Departure Datetime]	= DischargeDateTime
    FROM #MRU
    LEFT JOIN [CL3-data].[DataWarehouse].[ed].[vw_EDAttendance]	AS EDATN ON EDATN.admitprvsprefno = prvsp_refno"""
    arrivals = pd.read_sql(arrivals_query, sdmart_engine).sort_values(by='Ward Stay Start Datetime')
    arrivals['Date'] = arrivals['Ward Stay Start Datetime'].dt.date
    arrivals['ED Date'] = arrivals['ED Arrival Datetime'].dt.date
    print(f'Average Real Daily arrivals {arrivals['Date'].value_counts().mean()}')

    #Put arrivals into age bands
    banding = age_bands(arrivals['Age'])
    arrivals['Age Bands'] = [band if int(re.findall(r'\d+', band)[0]) < 90
                                  else '90 and over' for band in banding[0]]
                                  
    #Read in the year on year population change, get a list of years to forecast
    change = pd.read_csv(
             'C:/Users/obriene/Projects/Discrete Event Simulation/MSDEC model/UHPT Population Change.csv'
             ).set_index('Age Bands')
    years = [int(col) for col in change.columns]
    change.columns = years
    years = [int(datetime.today().year)] + years
    
    def inter_arrivals(data, arr_col, change, years):
        #Get the hour and date part of the arrival datetime
        data['Hour'] = data[arr_col].dt.hour
        data['Date'] = data[arr_col].dt.date
        #Get the number of data for each day and hour
        mean_arr = (data.groupby(['Age Bands', 'Date', 'Hour'],
                                        dropna=True, as_index=False)
                                        [arr_col].count())
        #Create a cross tab for all ages, dates and hours.  Join this to data
        #and fill in 0 where no arrivals for that age band on that date/hour.
        crosstab = pd.DataFrame(itertools.product(
                            data['Age Bands'].drop_duplicates(),
                            data['Date'].drop_duplicates(),
                            data['Hour'].drop_duplicates()),
                            columns=['Age Bands', 'Date', 'Hour'])
        mean_arr = crosstab.merge(mean_arr, on=['Age Bands', 'Date', 'Hour'],
                                  how='left').fillna(0)
        #Get the average number of arrivals per hour for each age band. If 0,
        #Replace with forward or back fill, to avoid divion by 0 when getting
        #inter arrival times.
        mean_arr = (mean_arr.groupby(['Age Bands', 'Hour'])[arr_col].mean()
                    .replace(0, np.nan).groupby(['Age Bands', 'Hour'])
                    .ffill().bfill())
        #Loop through each age group and using current arrival numbers, use the
        #population projections to simulate number of arrivals in n years
        change = pd.DataFrame(mean_arr).join(change)
        projections = []
        for row in change.values.tolist():
            start = row[0]
            props = row[1:]
            new_row = [start]
            for prop in props:
                start *= prop
                new_row.append(start)
            projections.append(new_row)
        #Create data frame and transform into inter arrival times
        daily_arrivals = pd.DataFrame(projections, columns=years,
                                      index=change.index)
        daily_arrivals = daily_arrivals.groupby('Hour').sum()
        inter_arr = 60 / daily_arrivals
        return daily_arrivals # itner_arr
    
    #In data, sometimes patients come to ED months before MRU (despite matching
    #prvsp_refno), so need to filter ED arrivals to between the same time period
    #to prevent arrivals being too low.
    ED = arrivals.loc[(arrivals['ED Arrival Datetime']
                       > pd.to_datetime(start_dttm, dayfirst=True))
                    & (arrivals['ED Arrival Datetime']
                       < pd.to_datetime(end_dttm, dayfirst=True))].copy()  
    ED_arr = inter_arrivals(ED, 'ED Arrival Datetime', change, years)
    EXT = arrivals.loc[arrivals['ED Arrival Datetime'].isna()].copy()
    EXT_arr = inter_arrivals(EXT, 'Ward Stay Start Datetime', change, years)

    print(f'ED Average Real Daily arrivals {ED['ED Date'].value_counts().mean()}')
    print(f'EXT Average Real Daily arrivals {EXT['Date'].value_counts().mean()}')
    print(pd.DataFrame([ED_arr.sum(), (EXT_arr).sum(),
                        ED_arr.sum() + (EXT_arr).sum()],
                        index=['ED', 'EXT', 'TOTAL']).T)

    #LoS
    mean_ED_los = 120
    min_MRU_los = 60*1.5
    mean_MRU_los = 60*8
    max_MRU_los = 60*24
    bed_turnaround = 60
    #Discharge times
    dis_start = 8
    dis_end = 22
    #resources
    no_MRU_beds = np.inf
    #lists for storing results
    pat_res = []
    occ_res = []

class spawn_patient:
    def __init__(self, p_id):
        self.id = p_id
        self.age_band = np.nan
        self.arrival = ''
        self.arrival_year = np.nan
        self.ED_arrival_time = np.nan
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
        self.year = self.input_params.start_year #set start year
        #establish resources
        self.MRU_bed = simpy.Resource(self.env,
                                      capacity=input_params.no_MRU_beds)
        self.ED = simpy.Resource(self.env, capacity = np.inf)

    ############################INCREASE YEAR###############################
    def increase_year(self):
        while True:
            yield self.env.timeout(365*24*60)
            self.year += 1

    ###########################MODEL TIME#############################
    def model_time(self, time):
        #Work out what day and time it is in the model.
        day = math.floor(time / (24*60))
        hour = math.floor((time % (day*(24*60)) if day != 0 else time) / 60)
        return day, hour
    
    ########################ARRIVALS################################
    def ED_arrivals(self):
        while True:
            #up patient counter and spawn a new walk-in patient
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter)
            p.arrival = 'ED'
            p.arrival_year = self.year
            #begin patient ED process
            self.env.process(self.ED_to_MRU_journey(p))
            #randomly sample the time until the next patient arrival
            time_of_day = math.floor(self.env.now % (60*24) / 60)
            no_arr = self.input_params.ED_arr.loc[time_of_day, self.year]
            #If more than 1 arrival within an hour, just use this as the time out.
            if no_arr > 1:
                ED_arr = round(60 / self.input_params.ED_arr.loc[time_of_day, self.year])
            else:
                #Otherwise, wait until next hour then check whent he next
                #arrival will be
                ED_arr = 60
                arr_bool = True
                while arr_bool:
                    #Check the number of hourly arrivals for next hour and use
                    #arrival numbers as probability.
                    time_of_day = time_of_day + 1 if time_of_day != 23 else 0
                    no_arr = self.input_params.ED_arr.loc[time_of_day, self.year]
                    if random.random() < no_arr:
                        #there will be an arrival next hour, escape while loop.
                        arr_bool = False
                    else:
                        #If no arrival in the next hour, add to the time out
                        #and check the hour after that
                        ED_arr += 60
            sampled_interarrival = round(random.expovariate(1.0 / ED_arr))
            yield self.env.timeout(sampled_interarrival)
    
    def External_arrivals(self):#, age_band):
        while True:
            #up patient counter and spawn a new walk-in patient
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter)
            p.arrival = 'External'
            p.arrival_year = self.year
            #begin patient ED process
            self.env.process(self.MRU_journey(p))

                        #randomly sample the time until the next patient arrival
            time_of_day = math.floor(self.env.now % (60*24) / 60)
            no_arr = self.input_params.EXT_arr.loc[time_of_day, self.year]
            #If more than 1 arrival within an hour, just use this as the time out.
            if no_arr > 1:
                EXT_arr = round(60 / self.input_params.EXT_arr.loc[time_of_day, self.year])
            else:
                #Otherwise, wait until next hour then check whent he next
                #arrival will be
                EXT_arr = 60
                arr_bool = True
                while arr_bool:
                    #Check the number of hourly arrivals for next hour and use
                    #arrival numbers as probability.
                    time_of_day = time_of_day + 1 if time_of_day != 23 else 0
                    no_arr = self.input_params.EXT_arr.loc[time_of_day, self.year]
                    if random.random() < no_arr:
                        #there will be an arrival next hour, escape while loop.
                        arr_bool = False
                    else:
                        #If no arrival in the next hour, add to the time out
                        #and check the hour after that
                        EXT_arr += 60
            sampled_interarrival = round(random.expovariate(1.0 / EXT_arr))
            yield self.env.timeout(sampled_interarrival)
    
    ##################ED TO MRU PROCESS #########################

    def MRU_journey(self, patient):
        #Enter MRU
        patient.MRU_wait_start_time = self.env.now
        with self.MRU_bed.request() as req:
            yield req
            #Get the arrival time and initial sampled length of stay
            arr_time = self.env.now
            patient.MRU_arrival_time = arr_time
            sampled_MRU_time = round((random.expovariate(1.0
                                                / self.input_params.mean_MRU_los)))
            #If sampled LoS is greater than the max, resample until under.
            while ((sampled_MRU_time < self.input_params.min_MRU_los) 
                   or (sampled_MRU_time > self.input_params.max_MRU_los)):
                sampled_MRU_time = round((random.expovariate(1.0
                                                / self.input_params.mean_MRU_los)))
            #Patient holds MRU bed for the sampled MRU time
            yield self.env.timeout(sampled_MRU_time + self.input_params.bed_turnaround)
        patient.MRU_leave_time = self.env.now - self.input_params.bed_turnaround
        self.store_patient_results(patient)

    def ED_to_MRU_journey(self, patient):
        #Patient comes into ed
        patient.ED_arrival_time = self.env.now 
        with self.ED.request() as req:
            yield req
            #randomly sample the time spent in ED
            sampled_ED_time = round(min(random.expovariate(1.0
                                                / self.input_params.mean_ED_los), 240))
            yield self.env.timeout(sampled_ED_time)

        #Enter MRU
        patient.MRU_wait_start_time = self.env.now
        with self.MRU_bed.request() as req:
            yield req
            #Get the arrival time and initial sampled length of stay
            arr_time = self.env.now
            patient.MRU_arrival_time = arr_time
            sampled_MRU_time = round(random.expovariate(1.0
                                                / self.input_params.mean_MRU_los))
            #If sampled LoS is greater than the max, resample until under.
            while ((sampled_MRU_time < self.input_params.min_MRU_los) 
                   or (sampled_MRU_time > self.input_params.max_MRU_los)):
                sampled_MRU_time = round((random.expovariate(1.0
                                                / self.input_params.mean_MRU_los)))
            #Patient holds MRU bed for the sampled MRU time
            yield self.env.timeout(sampled_MRU_time + self.input_params.bed_turnaround)
        patient.MRU_leave_time = self.env.now - self.input_params.bed_turnaround
        self.store_patient_results(patient)

###################RECORD RESULTS####################
    def store_patient_results(self, patient):
        self.patient_results.append([self.run_number, patient.id,
                                     patient.arrival, patient.arrival_year,
                                     patient.ED_arrival_time,
                                     patient.MRU_wait_start_time,
                                     patient.MRU_arrival_time,
                                     patient.MRU_leave_time])
    
    def store_occupancy(self):
        while True:
            self.mru_occupancy_results.append([self.run_number,
                                               self.ED._env.now,
                                               self.year,
                                               self.ED.count,
                                               len(self.MRU_bed.queue),
                                               self.MRU_bed.count])
            yield self.env.timeout(self.input_params.occ_sample_time)
########################RUN#######################
    def run(self):
        self.env.process(self.increase_year())
        self.env.process(self.ED_arrivals())
        self.env.process(self.External_arrivals())
        self.env.process(self.store_occupancy())
        self.env.run(until = self.input_params.run_time)
        default_params.pat_res += self.patient_results
        default_params.occ_res += self.mru_occupancy_results
        return self.patient_results, self.mru_occupancy_results

def export_results(pat_results, occ_results):
    patient_df = pd.DataFrame(pat_results,
                              columns=['Run', 'Patient ID',
                                       'Arrival Method',
                                       'Arrival Year',
                                       'ED Arrival Time',
                                       'MRU Wait Start Time',
                                       'MRU Arrival Time', 'MRU Leave Time'])
    patient_df['Simulation Arrival Time'] = (patient_df['ED Arrival Time']
                                             .fillna(patient_df['MRU Wait Start Time']))
    patient_df['Simulation Arrival Day'] = patient_df['Simulation Arrival Time'] // (24*60) 
    patient_df['MRU Arrival Hour'] = ((patient_df['MRU Arrival Time'] / 60)
                                      % 24).apply(np.floor)
    patient_df['Wait for MRU Bed Time'] = (patient_df['MRU Arrival Time']
                                           - patient_df['MRU Wait Start Time'])
    patient_df['Simulation Leave Day'] = patient_df['MRU Leave Time']// (24*60)
    patient_df['Simulation Leave Hour'] = ((patient_df['MRU Leave Time'] / 60)
                                           % 24).apply(np.floor)

    
    occupancy_df = pd.DataFrame(occ_results,
                                columns=['Run', 'Time', 'Year', 'ED Occupancy',
                                'MRU Bed Queue', 'MRU Occupancy'])
    occupancy_df['day'] = occupancy_df['Time'] // (24*60)
    occupancy_df['hour'] = ((occupancy_df['Time'] / 60) % 24).apply(np.floor)
    return patient_df, occupancy_df

def run_the_model(input_params):
    #run the model for the number of iterations specified
    for run in range(input_params.iterations):
        print(f"Run {run+1} of {input_params.iterations}")
        model = mru_model(run, input_params)
        model.run()
    patient_df, occ_df = export_results(
                                        input_params.pat_res,
                                        input_params.occ_res)
    return patient_df, occ_df

###############################################################################
#Run and save results
pat, occ = run_the_model(default_params)
pat.to_csv(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/Full Outputs/patients - {default_params.run_name}.csv')
occ.to_csv(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/Full Outputs/occupancy - {default_params.run_name}.csv')

#Quartiles and font size
def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)
def q80(x):
    return x.quantile(0.80)
def q85(x):
    return x.quantile(0.85)
def q90(x):
    return x.quantile(0.90)
def q95(x):
    return x.quantile(0.95)
font_size = 24

#Summary numbers
print('---------------------')
print('Arrivals Summary:')
print(pat.groupby(['Run',  'Simulation Arrival Day', 'Arrival Year'],as_index=False)['Patient ID'].count()
         .groupby(['Arrival Year'])['Patient ID']
         .agg(['min', q25, 'mean', q75, q80, q85, q90, q95, 'max']))

print('---------------------')
print('Daily Occupancy Quartiles:')
#Get only occupancy during opening hours on a week day, as this is higher than
#weekends.  Print the quartiles
occ_sum = occ.groupby('Year')['MRU Occupancy'].agg(
                ['min', q25, 'mean', q75, q80, q85, q90, q95, 'max']).round(2)
print(occ_sum)

print('-----------------------')
print('Arrivals Based on Occupancy Quartiles:')
#Find the days where occupancy reaches those quartiles.
now = occ_sum.iloc[0]
fut = occ_sum.iloc[-1]
def arrivals_for_quartiles(row, label):
    arrs = []
    for quart in row.round():
        #Get where the number of beds occured, and the time 24 hours ago
        days = occ.loc[(occ['MRU Occupancy'] == quart)
                       & (occ['Year'] == int(label)), ['Run', 'Time']].copy()
        days['24hour'] = (days['Time'] - (60*24)).clip(lower=0)
        #remove any negative times
        days = days.loc[days['24hour'] > 0].copy()
        #Add id for each instance, get list of all times in last 24 hours and explode
        days['24hour ID'] = [i for i in range(len(days))]
        days['times'] = [[i for i in range(x, y)] for x , y in zip( days['24hour'], days['Time'])]
        days = days.explode('times')
        #Merge this back onto the patients table to get the average number of
        #patients 24hours before that number was hit.
        arrivals = (days.merge(pat, left_on=['Run', 'times'],
                              right_on=['Run', 'MRU Arrival Time'])
                              .groupby('24hour ID')['Patient ID'].count().mean())
        arrs.append(arrivals)
    out = pd.DataFrame([arrs], columns=row.index, index=[label]).round(2)
    return out

print(arrivals_for_quartiles(now, '2025'))
print(arrivals_for_quartiles(fut, '2034'))
print('-----------------------')







####arrivals by hour of day
MRU_arrivals = (pat.groupby(['Run', 'Simulation Arrival Day', 'MRU Arrival Hour'], as_index=False)
                ['Patient ID'].count()
                .groupby('MRU Arrival Hour')['Patient ID'].mean())
MRU_arrivals.name = 'Arrivals'
#discharged
MRU_discharges = (pat.groupby(['Run', 'Simulation Leave Day', 'Simulation Leave Hour'], as_index=False)
                  ['Patient ID'].count()
                  .groupby('Simulation Leave Hour')['Patient ID'].mean())
MRU_discharges.name = 'Discharges'
#combine
MRU_in_out = pd.DataFrame([MRU_arrivals, MRU_discharges]).T
#occupancy
MRU_occupancy = occ.groupby('hour')['MRU Occupancy'].agg(['min', q25, 'mean', q75, q80, q85, q90, q95, 'max'])
hours = MRU_occupancy.index

#plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
fig.suptitle('MRU by Hour of Day', fontsize=24)
ax1.plot(hours, MRU_occupancy['mean'].fillna(0), '-r', label='Mean')
ax1.fill_between(hours, MRU_occupancy['min'].fillna(0), MRU_occupancy['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
ax1.fill_between(hours, MRU_occupancy['q25'].fillna(0), MRU_occupancy['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
ax1.set_title('Occupancy', fontsize=18)
ax1.set_xlabel('Hour of Day', fontsize=18)
ax1.set_ylabel('No. Beds Occupied', fontsize=18)
ax1.tick_params(axis='both',  which='major', labelsize=18)
ax1.legend(fontsize=18)
MRU_in_out.plot(ax=ax2)
ax2.set_title('Arrivals/Discharges', fontsize=18)
ax2.set_xlabel('Hour of Day', fontsize=18)
ax2.tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/Hour of Day Occ, Arr and Dis - {default_params.run_name}.png', bbox_inches='tight', dpi=1200)
plt.close()

####occupancy by year
#Set formats
whis = (5, 95)
boxprops = dict(linestyle='-', linewidth=3, color='black')
whiskerprops = dict(linestyle='-', linewidth=3, color='black')
capprops = dict(linestyle='-', linewidth=3, color='black')
medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
meanlineprops = dict(linestyle='--', linewidth=2.5, color='firebrick')
flierprops = dict(marker='o', markerfacecolor='grey', markersize=5, markeredgecolor='none')
#Create box plot for each year
columns = occ['Year'].drop_duplicates().to_list()
fig, ax = plt.subplots(figsize=(20,10))
for position, column in enumerate(columns):
    bp = ax.boxplot(occ.loc[occ['Year'] == column, 'MRU Occupancy'], positions=[position],
                    sym='.', widths=0.9, whis=whis, showmeans=True,  meanline=True,
                    boxprops=boxprops, medianprops=medianprops,
                    whiskerprops=whiskerprops,
                    flierprops=flierprops, capprops=capprops,
                    meanprops=meanlineprops)
#Create and save figure
ax.set_yticks(range(occ['MRU Occupancy'].max()+1))
ax.set_xticklabels(columns, fontdict={'fontsize':20})
ax.set_xlim(xmin=-0.5)
ax.set_title(f'{default_params.run_name} MRU Chair Occupancy Box Plots by Year ({whis[0]}% - {whis[1]}%)', fontdict={'fontsize':20})
plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
ax.grid()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/{default_params.run_name} MRU Occupancy Box Plots by Year.png', bbox_inches='tight')
plt.close()

#### MRU Bed queue
queue = occ.groupby('day')['MRU Bed Queue'].mean()
fig, axs = plt.subplots(figsize=(25, 10))
axs.plot(queue.index, queue)
axs.set_title(f'Average Number of Patients in MRU Queue - {default_params.run_name}', fontsize=font_size)
axs.set_xlabel('Simulation Day', fontsize=font_size)
axs.set_ylabel('No. Patients', fontsize=font_size)
axs.tick_params(axis='both',  which='major', labelsize=font_size)
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/Average Number of Patients in MRU Queue - {default_params.run_name}.png', bbox_inches='tight', dpi=1200)
plt.close()

#### MRU Bed Wait Time
wait_for_bed = pat.groupby('Simulation Arrival Day')['Wait for MRU Bed Time'].mean() / 60
fig, axs = plt.subplots(figsize=(25, 10))
axs.plot(wait_for_bed.index, wait_for_bed)
axs.set_title(f'Average Hours Waiting for MRU Bed - {default_params.run_name}', fontsize=font_size)
axs.set_xlabel('Simulation Day', fontsize=font_size)
axs.set_ylabel('Hours Waited', fontsize=font_size)
axs.tick_params(axis='both',  which='major', labelsize=font_size)
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/Average Wait for Bed Time - {default_params.run_name}.png', bbox_inches='tight', dpi=1200)
plt.close()
