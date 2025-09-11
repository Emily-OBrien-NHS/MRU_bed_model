import os
import re
import math
import pandas as pd
from datetime import datetime
os.chdir('C:/Users/obriene/Projects/Discrete Event Simulation/MSDEC model/Population Data')

################################################################################
                            ####FUNCTIONS####
################################################################################
def age_bands(age_col):
    '''Function to group up age column into 5 year age bands'''
    age_band = []
    order = []
    for age in age_col.values:
        try:
            #Divide the age by 5 and round up and down to get upper and lower
            #bounds
            divide = int(age) / 5
            lower = math.floor(divide) * 5
            upper = math.ceil(divide) * 5
            #If upper and lower are the same (i.e. age is a multiple of 5),
            #increase upper
            if lower == upper:
                upper += 5
            #Take 1 away from upper bound so boundaries don't overlap.
            upper -= 1
            #append the correct string and add the upper bound to the order list
            age_band.append(f"{lower} to {upper}")
            order.append(upper)
            #calculate which of the current population age bands would correspond
            #to this age band, for scaling later on.
        except:
            age_band.append(age)
            order.append(900)
    return age_band, order

def proportion_age_bands(age_col, bands):
    '''Function to create an age band column to match the population
    proportions'''
    parsed_bands = []
    for band in bands:
        # Extract numbers from the string
        numbers = re.findall(r'\d+', band)
        if len(numbers) == 2:
            parsed_bands.append((int(numbers[0]), int(numbers[1]), band))
        else:
            parsed_bands.append((int(numbers[0]), float('inf'), band))
    #Function to return which of the porportion age bands an age falls under
    band_label = []
    for age in age_col.values:
        age = int(re.findall(r'\d+', age)[0])
        for min_age, max_age, label in parsed_bands:
                if min_age <= age <= max_age:
                    band_label.append(label)
    return band_label

################################################################################
                        ####POPULATION DATASETS####
################################################################################
#ONS Population forecasts from
# https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/populationprojectionsforsubintegratedcareboardsbysingleyearofageandsexengland
pop_projection = pd.read_csv('2022 SNPP SICB pop persons.csv')
#2022 population by LSOA
#https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimatesnationalstatistics
current_pop = pd.read_excel("C:/Users/obriene/Downloads/sapelsoabroadage20112022.xlsx", sheet_name='Mid-2022 LSOA 2021', skiprows=3)
#List of LSOAs in each ICB
# https://www.data.gov.uk/dataset/5a615795-acb1-44cd-8f1f-4dc33317a73a/lsoa-2011-to-clinical-commissioning-group-to-lad-april-2019-lookup-in-en
icb_lsoas = pd.read_excel("LSOAs to ICBs.xlsx")
#List of UHP LSOAs
#From Hannah
uhp_lsoas = pd.read_csv('SecondaryCatchmentLSOAs.csv')
uhp_lsoas['UHP'] = 'UHP'

################################################################################
    #####Calculate proportion of ICB Data which fall under UHP LSOAs#####
################################################################################
#merge the current population data on the ICB LSOAs to get which ones are in
#which ICB, and filter to Devon ICB
joined = current_pop.merge(icb_lsoas, left_on='LSOA 2021 Code', right_on='LSOA11CD')
joined = joined.loc[joined['CCG19CD'] == 'E38000230'].copy()
#Combine male and female age band totals
bands = ['0 to 15', '16 to 29', '30 to 44', '45 to 64', '65 and over']
for band in bands:
    joined[band] = joined[f'F{band}'] + joined[f'M{band}']
#Merge onto UHP LSOA list to have a column for each LSOA population row to
#determine if its a UHP LSOA or not.  Fill Nans with 'Other'
joined = joined.merge(uhp_lsoas, left_on='LSOA 2021 Code', right_on='LowerSuperOutputArea2011', how='left')
joined['UHP'] = joined['UHP'].fillna('Other')
#Group populations by if they're UHP or not
cur_pop = joined.groupby('UHP')[bands].sum()
#Get the proportions of the Devon ICB population that currently fall under
#UHP for each age band, we will use this to scale the population projections.
proportion = (cur_pop.loc['UHP'] / cur_pop.sum()).reset_index()

################################################################################
    #####Group up population projections, and scale down to UHP proportion#####
################################################################################
#Filter data to devon ICB and only required years
years = [str(i) for i in range(2022, datetime.today().year + 11)]
pop_projection = pop_projection.loc[(pop_projection['AREA_CODE'] == 'E38000230')
                                    & (pop_projection['AGE_GROUP'] != 'All ages'),
                                    ['AGE_GROUP'] + years].copy()
#Group up age column into age bands using functions.  One for 5 year groups
#(including the ordering column for sorting), and one based on the age bands
#in the current population (for scaling the data).
banding = age_bands(pop_projection['AGE_GROUP'])
pop_projection['Age Bands'] = banding[0]
pop_projection['Age Order'] = banding[1]
pop_projection['Prop Age Bands'] = proportion_age_bands(pop_projection['AGE_GROUP'], bands)

#Group up to these age bands and sum the predicted populations
age_projection = pop_projection.groupby(['Age Bands', 'Prop Age Bands'],
                                        as_index=False)[
                                        ['Age Order'] + [str(i) for i in years]
                                        ].sum().sort_values(by='Age Order')
#Merge onto the proportion of the current populaiton that is in UHPT LSOAs,
#and multiply the forecast by this.
age_projection = age_projection.merge(proportion, left_on='Prop Age Bands',
                                      right_on='index', how='left')
age_projection[years] = age_projection[years].multiply(age_projection[0],
                                              axis='index').round().astype(int)
#Re-group by only our 5 year age bands (to prevent splits due to grouping by
#two different age band definitions)
age_projection = (age_projection.sort_values(by='Age Order')
                                .groupby('Age Bands')[
                                ['Age Order'] + [str(i) for i in years]].sum())

################################################################################
                            #####Save to csv#####
################################################################################
#Filter down to only required columns and save to csv for use in the model.
age_projection = age_projection.sort_values(by='Age Order')[years].copy()
age_projection.to_csv('C:/Users/obriene/Projects/Discrete Event Simulation/MSDEC model/UHPT Population Projection.csv')
