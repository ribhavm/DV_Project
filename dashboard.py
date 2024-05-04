import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import plotly.express as px
import seaborn as sns


property_df1=pd.read_csv("./data/indian-crime/10_Property_stolen_and_recovered.csv")
property_df2=pd.read_csv("./data/indian-crime/30_Auto_theft.csv")
property_df3=pd.read_csv("./data/indian-crime/crime/11_Property_stolen_and_recovered_nature_of_property.csv")
data=pd.read_csv("./data/indian-crime/20_Victims_of_rape.csv")
murder_df=pd.read_csv("./data/indian-crime/32_Murder_victim_age_sex.csv")
fraud_data=pd.read_csv("./data/indian-crime/31_Serious_fraud.csv")
crime_df=pd.read_csv("./data/indian-crime/crime/01_District_wise_crimes_committed_IPC_2001_2012.csv")
crime_df = crime_df.rename(columns={'STATE/UT': 'STATE'})
population_df = pd.read_csv('./data/district wise population for year 2001 and 2011.csv')
if 'index' in population_df.columns:
    population_df = population_df.drop(columns=['index'])

population_df.reset_index(inplace=True)

state_grouped = population_df.groupby('State')

def interpolate_population(row):
    years = np.arange(2001, 2011)
    population_2001 = row['Population in 2001']
    population_2011 = row['Population in 2011']
    interpolated_populations = np.linspace(population_2001, population_2011, num=10)
    return interpolated_populations

interpolated_populations = population_df.apply(interpolate_population, axis=1, result_type='expand')
interpolated_populations.columns = [f'Population in {year}' for year in range(2001, 2011)]

interpolated_populations.reset_index(inplace=True)

interpolated_populations['State'] = population_df['State']

melted_populations = pd.melt(interpolated_populations, id_vars='State', var_name='Year', value_name='Population')

melted_populations['Year'] = melted_populations['Year'].str.extract('(\d+)')

melted_populations['Year'] = pd.to_numeric(melted_populations['Year'])

total_population = melted_populations.groupby(['State', 'Year'])['Population'].sum().reset_index()

average_population = total_population[total_population['Year'].between(2001, 2010)].groupby('State')['Population'].mean()

popu_df = pd.DataFrame({
    f'{year}': total_population[total_population['Year'] == year].set_index('State')['Population']
    for year in range(2001, 2011)
})

popu_df['State'] = popu_df.index

popu_df['Average Population'] = average_population.values

popu_df['State'] = popu_df['State'].replace({'Odisha (Orissa)': 'Orissa', 'Puducherry (Pondicherry)': 'Pondicherry'})

popu_df.reset_index(drop=True, inplace=True)

popu_df = popu_df[['State', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', 'Average Population']]


india_population = popu_df.iloc[:, 1:11].sum()


india_df = pd.DataFrame({'Year': india_population.index, 'Total Population': india_population.values})


india_df.reset_index(drop=True, inplace=True)


india_df.columns = ['Year', 'Total Population']

crime_df_final=pd.read_csv("./data/indian-crime/crime/01_District_wise_crimes_committed_IPC_2001_2012.csv")
crime_df_final = crime_df_final.rename(columns={'STATE/UT': 'STATE'})
popu_df_final = popu_df.copy()
popu_df_final.rename(columns={'State': 'STATE'}, inplace=True)
india_map_final = gpd.read_file('./data/indian-shapefile/india_ds.shp')
crime_df_final['STATE'] = crime_df_final['STATE'].str.lower().str.replace('&', 'and').replace('odisha', 'odisha').replace('delhi ut', 'delhi')

# Transform popu_df_final
popu_df_final['STATE'] = popu_df_final['STATE'].str.lower().str.replace('&', 'and').replace('odisha', 'odisha').replace('delhi ut', 'delhi')

# Transform india_map_final
india_map_final['STATE'] = india_map_final['STATE'].str.lower().str.replace('&', 'and').replace('odisha', 'odisha').replace('delhi ut', 'delhi')

def page1_panel1():
        
        total_cases_by_year = crime_df.groupby('YEAR').sum()
        total_cases_by_year.reset_index(inplace=True)


        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(total_cases_by_year['YEAR'], total_cases_by_year['TOTAL IPC CRIMES'], marker='o', linestyle='-')
        ax.set_title('Total IPC Crimes in India (2001-2010)',fontsize=20)
        ax.set_xlabel('Year',fontsize=20)
        ax.set_ylabel('Total Cases',fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.grid(True)
        ax.set_xticks(total_cases_by_year['YEAR'])  
        return st.pyplot(fig)

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def page1_panel2():
    
    total_crimes_by_type = crime_df.drop(columns=['STATE', 'DISTRICT', 'YEAR', 'TOTAL IPC CRIMES']).sum()

    
    small_share_threshold = 0.01
    small_share_mask = total_crimes_by_type / total_crimes_by_type.sum() < small_share_threshold

    
    small_share_total = total_crimes_by_type[small_share_mask].sum()
    total_crimes_by_type = total_crimes_by_type[~small_share_mask]
    total_crimes_by_type['Small Shares'] = small_share_total

    fig, ax = plt.subplots(figsize=(10, 6))
    total_crimes_by_type.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax)
    ax.set_title('Distribution of Total Crimes by Type (2001-2010)')
    ax.axis('equal')
    ax.set_ylabel(None)
    st.pyplot(fig)


def page1_panel4():
    
    total_crimes_by_state = crime_df.groupby('STATE')['TOTAL IPC CRIMES'].sum()

    total_crimes_percentage = total_crimes_by_state / total_crimes_by_state.sum() * 100
    other_states_mask = total_crimes_percentage < 1
    other_states_crimes = total_crimes_by_state[other_states_mask].sum()

    total_crimes_by_state = total_crimes_by_state[~other_states_mask]
    total_crimes_by_state['Other States'] = other_states_crimes

    
    fig, ax = plt.subplots(figsize=(10, 8))
    total_crimes_by_state.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax)
    ax.set_title('Distribution of Total Crimes by State (2001-2010)')
    ax.axis('equal')
    ax.set_ylabel(None)
    plt.tight_layout()

    
    st.pyplot(fig)
def page1_panel4_capita():
    
    crime_df_lower = crime_df.copy()

    crime_df_lower['STATE'] = crime_df_lower['STATE'].str.lower().str.title()

    total_crimes_by_state = crime_df_lower.groupby('STATE')['TOTAL IPC CRIMES'].sum()

    total_population_by_state = popu_df.set_index('State').sum(axis=1)

    crime_rate_by_state = (total_crimes_by_state / total_population_by_state) * 100000

    crime_rate_by_state.index = crime_rate_by_state.index.str.replace('&', 'And')
    crime_rate_by_state.index = crime_rate_by_state.index.str.replace('N ', 'N')
    crime_rate_by_state.index = crime_rate_by_state.index.str.replace('Delhi Ut', 'Delhi')
    crime_rate_by_state.index = crime_rate_by_state.index.str.replace('Orissa', 'Odisha')
    crime_rate_by_state.index = crime_rate_by_state.index.str.replace('Pondicherry', 'Puducherry')

    crime_rate_by_state = crime_rate_by_state.dropna()

    other_states_mask = (crime_rate_by_state / crime_rate_by_state.sum()) < 0.015

    other_states_crime_rate = crime_rate_by_state[other_states_mask].sum()
    crime_rate_by_state = crime_rate_by_state[~other_states_mask]
    crime_rate_by_state['Other States'] = other_states_crime_rate

    
    fig, ax = plt.subplots(figsize=(10, 8))
    crime_rate_by_state.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax)
    ax.set_title('Distribution of Crime Rate by State (2001-2010)')
    ax.axis('equal')
    ax.set_ylabel(None)
    plt.tight_layout()

    
    st.pyplot(fig)

def page1_panel5(crime_df):
   
    india_map = gpd.read_file('./data/indian-shapefile/india_ds.shp')

    
    crime_df_local1=crime_df.copy()
    crime_df_local1['STATE'] = crime_df_local1['STATE'].str.lower().str.replace('&', 'and')
    crime_df_local1 = crime_df_local1.replace({'odisha': 'orissa'})

    # Group by state and find the total crimes
    total_crimes_per_state = crime_df_local1.groupby('STATE')['TOTAL IPC CRIMES'].sum().reset_index()

    # Replace inconsistent state names
    total_crimes_per_state['STATE'] = total_crimes_per_state['STATE'].astype(str)
    total_crimes_per_state['STATE'] = total_crimes_per_state['STATE'].str.replace('&', 'And')
    total_crimes_per_state['STATE'] = total_crimes_per_state['STATE'].str.replace('N ', 'N')
    total_crimes_per_state['STATE'] = total_crimes_per_state['STATE'].str.replace('Delhi Ut', 'Delhi')
    total_crimes_per_state['STATE'] = total_crimes_per_state['STATE'].str.replace('Orissa', 'Odisha')
    total_crimes_per_state['STATE'] = total_crimes_per_state['STATE'].str.replace('Pondicherry', 'Puducherry')
    # print(total_crimes_per_state)
    india_map['STATE'] = india_map['STATE'].str.lower().str.replace('&', 'and')
    # Merge the crime data with the shapefile
    merged_data = india_map.merge(total_crimes_per_state, how='left', left_on='STATE', right_on='STATE')

    # Create the choropleth map
    fig = px.choropleth(merged_data,
                        geojson=merged_data.geometry,
                        locations=merged_data.index,
                        color='TOTAL IPC CRIMES',
                        hover_name='STATE',
                        hover_data={'TOTAL IPC CRIMES': True},
                        color_continuous_scale='Viridis',
                        labels={'TOTAL IPC CRIMES': 'Total Crimes'})

    # Adjust the map boundaries and layout
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title='Total Crimes Heatmap in India', margin={"r": 0, "t": 30, "l": 0, "b": 0})

    # Display the map in Streamlit
    st.plotly_chart(fig, use_container_width=True)
        # Group by state and district to find the total crimes
    # total_crimes_per_district = crime_df.groupby(['STATE', 'DISTRICT'])['TOTAL IPC CRIMES'].sum().reset_index()

    # # Replace inconsistent state names
    # total_crimes_per_district['STATE'] = total_crimes_per_district['STATE'].astype(str)
    # total_crimes_per_district['STATE'] = total_crimes_per_district['STATE'].str.replace('&', 'And')
    # total_crimes_per_district['STATE'] = total_crimes_per_district['STATE'].str.replace('N ', 'N')
    # total_crimes_per_district['STATE'] = total_crimes_per_district['STATE'].str.replace('Delhi Ut', 'Delhi')
    # total_crimes_per_district['STATE'] = total_crimes_per_district['STATE'].str.replace('Orissa', 'Odisha')
    # total_crimes_per_district['STATE'] = total_crimes_per_district['STATE'].str.replace('Pondicherry', 'Puducherry')
    # india_map['DISTRICT'] = india_map['DISTRICT'].str.upper()
    # print(total_crimes_per_district)
    # # Merge the crime data with the shapefile
    # merged_data = india_map.merge(total_crimes_per_district, how='left', left_on=['STATE', 'DISTRICT'], right_on=['STATE', 'DISTRICT'])

    # # Create the choropleth map
    # fig = px.choropleth(merged_data,
    #                     geojson=merged_data.geometry,
    #                     locations=merged_data.index,
    #                     color='TOTAL IPC CRIMES',
    #                     hover_name='DISTRICT',
    #                     hover_data={'TOTAL IPC CRIMES': True, 'STATE': True},
    #                     color_continuous_scale='Viridis',
    #                     labels={'TOTAL IPC CRIMES': 'Total Crimes'},
    #                     projection="mercator")

    # # Adjust the map boundaries and layout
    # fig.update_geos(fitbounds="locations", visible=False)
    # fig.update_layout(title='Total Crimes Heatmap in India', margin={"r": 0, "t": 30, "l": 0, "b": 0})

    # # Display the map in Streamlit
    # st.plotly_chart(fig, use_container_width=True)



import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st

def page1_panel5_capita(crime_df, popu_df):
    
    india_map = india_map_final.copy()
    
    total_crimes_per_state = crime_df_final.groupby('STATE')['TOTAL IPC CRIMES'].sum().reset_index()
    

   
    merged_data = india_map.merge(total_crimes_per_state, how='left', left_on='STATE', right_on='STATE')

    # Calculate per capita crime rate
    merged_data = merged_data.merge(popu_df_final[['STATE', 'Average Population']], how='left', left_on='STATE', right_on='STATE')
    merged_data['Per Capita Crime Rate'] = merged_data['TOTAL IPC CRIMES'] / merged_data['Average Population'] * 100000

    
    fig = px.choropleth(merged_data,
                        geojson=merged_data.geometry,
                        locations=merged_data.index,
                        color='Per Capita Crime Rate',
                        hover_name='STATE',
                        hover_data={'Per Capita Crime Rate': True},
                        color_continuous_scale='Viridis',
                        labels={'Per Capita Crime Rate': 'Per Capita Crime Rate'})

    
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title='Per Capita Crime Rate Heatmap in India', margin={"r": 0, "t": 30, "l": 0, "b": 0})

    
    st.plotly_chart(fig, use_container_width=True)


 
def murder_1(murder_df):
    age_group_totals = murder_df[['Victims_Above_50_Yrs', 'Victims_Upto_30_50_Yrs', 'Victims_Upto_18_30_Yrs',
                              'Victims_Upto_15_18_Yrs', 'Victims_Upto_10_15_Yrs', 'Victims_Upto_10_Yrs']].sum()

    plt.figure(figsize=(12, 6))
    age_group_totals.plot(kind='bar', stacked=True)
    plt.title('Murder Victims by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Murder Victims')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot (plt)

def murder_2(murder_df):
    plt.figure(figsize=(12, 6))
    murder_df_grouped = murder_df.groupby('Year')['Victims_Total'].sum().reset_index()
    sns.lineplot(data=murder_df_grouped, x='Year', y='Victims_Total')
    plt.title('Murder Victims Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Murder Victims')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def murder_6(murder_df):
    india_map = india_map_final.copy()

    india_map['STATE'] = india_map['STATE'].str.lower().str.replace('&', 'and')
    murder_df['Area_Name'] = murder_df['Area_Name'].str.lower().str.replace('&', 'and')
    murder_df['Area_Name'] = murder_df['Area_Name'].replace({'odisha': 'orissa'})

    merged_data = india_map.merge(murder_df.groupby('Area_Name')['Victims_Total'].sum().reset_index(),
                                  how='left', left_on='STATE', right_on='Area_Name')

    fig = px.choropleth(merged_data,
                        geojson=merged_data.geometry,
                        locations=merged_data.index,
                        color='Victims_Total',
                        hover_name='STATE',
                        hover_data={'Victims_Total': True},
                        color_continuous_scale='Viridis',
                        labels={'Victims_Total':'Total Murder Victims'})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title='Murder Victims Heatmap in India', margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig)

def murder_3(murder_df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Group by 'Year' and 'Group_Name', and sum the 'Victims_Total' for each group
    murder_df_grouped = murder_df.groupby(['Year', 'Group_Name'])['Victims_Total'].sum().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=murder_df_grouped, x='Year', y='Victims_Total', hue='Group_Name')
    plt.title('Total Murder Victims by Gender Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Total Number of Murder Victims')
    plt.xticks(rotation=45)
    plt.legend(title='Gender')
    plt.tight_layout()


    st.pyplot(plt)


def murder_5(murder_df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(murder_df.pivot_table(index='Year', columns='Area_Name', values='Victims_Total', aggfunc='sum'), cmap='YlGnBu')
    plt.title('Heatmap of Murder Victims by Age Group and Year')
    plt.xlabel('State')
    plt.ylabel('Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)


def murder_4(murder_df):
    age_group_totals = murder_df[['Victims_Above_50_Yrs', 'Victims_Upto_30_50_Yrs', 'Victims_Upto_18_30_Yrs',
                              'Victims_Upto_15_18_Yrs', 'Victims_Upto_10_15_Yrs', 'Victims_Upto_10_Yrs']].sum()
    plt.figure(figsize=(8, 8))
    plt.pie(age_group_totals, labels=age_group_totals.index, autopct='%1.1f%%', startangle=140)
    plt.title('Murder Victims by Age Group')
    plt.axis('equal')
    plt.tight_layout()
    st.pyplot(plt)



def rape_1(data):
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Year', y='Rape_Cases_Reported', estimator='sum')
    plt.title('Total Rape Cases Reported Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Total Rape Cases Reported')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def rape_2(data):
    

    age_group_totals = data[['Victims_Above_50_Yrs', 'Victims_Between_10-14_Yrs', 'Victims_Between_14-18_Yrs',
                            'Victims_Between_18-30_Yrs', 'Victims_Between_30-50_Yrs', 'Victims_Upto_10_Yrs']].sum()
    plt.figure(figsize=(8, 8))
    plt.pie(age_group_totals, labels=age_group_totals.index, autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of Rape Cases Reported by Age Group')
    plt.axis('equal')
    plt.tight_layout()
    st.pyplot(plt)


def rape_3(data):
    age_groups = ['Victims_Above_50_Yrs', 'Victims_Between_10-14_Yrs', 'Victims_Between_14-18_Yrs',
              'Victims_Between_18-30_Yrs', 'Victims_Between_30-50_Yrs', 'Victims_Upto_10_Yrs']
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Year', y=age_groups[0], estimator='sum', label='Above 50')
    for group in age_groups[1:]:
        sns.lineplot(data=data, x='Year', y=group, estimator='sum', label=group)
    plt.title('Trend of Rape Cases Reported Over the Years by Age Groups')
    plt.xlabel('Year')
    plt.ylabel('Number of Rape Cases Reported')
    plt.xticks(rotation=45)
    plt.legend(title='Age Group')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def rape_4(data):
    def radar_chart(df, title):
        labels=np.array(['Above 50', '10-14', '14-18', '18-30', '30-50', 'Upto 10'])
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        def _plot(ax, values, title, color):
            ax.fill(angles, values, color=color, alpha=0.25)
            ax.plot(angles, values, color=color, linewidth=2)
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=12)  # Add labels for each angle
            ax.set_title(title)
            ax.grid(True)

        colors = ['b', 'r', 'g', 'y', 'm', 'c']
        for i, row in df.iterrows():
            values = row[1:].values.flatten().tolist()
            values += values[:1]
            _plot(ax=ax, values=values, title=row[0], color=colors[i % len(colors)])

        plt.figlegend(labels=df['Year'].tolist(), loc='upper right')
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        st.pyplot(fig)

    radar_data = data.pivot_table(index='Year', values=['Victims_Above_50_Yrs', 'Victims_Between_10-14_Yrs',
                                                        'Victims_Between_14-18_Yrs', 'Victims_Between_18-30_Yrs',
                                                        'Victims_Between_30-50_Yrs', 'Victims_Upto_10_Yrs'],
                                  aggfunc='sum').reset_index()

    radar_chart(radar_data, 'Distribution of Rape Cases Reported by Age Groups')


def rape_5(data):
    india_map = gpd.read_file('./data/indian-shapefile/india_ds.shp')

    india_map['STATE'] = india_map['STATE'].str.lower().str.replace('&', 'and')

    data['Area_Name'] = data['Area_Name'].str.lower().str.replace('&', 'and')
    data['Area_Name'] = data['Area_Name'].replace({'odisha': 'orissa'})

    total_cases_per_state = data.groupby('Area_Name')['Rape_Cases_Reported'].sum().reset_index()
    max_cases_per_state = data.groupby(['Area_Name', 'Year'])['Rape_Cases_Reported'].sum().reset_index()
    max_cases_per_state = max_cases_per_state.loc[max_cases_per_state.groupby('Area_Name')['Rape_Cases_Reported'].idxmax()]
    min_cases_per_state = data[data['Rape_Cases_Reported'] > -1].groupby(['Area_Name', 'Year'])['Rape_Cases_Reported'].sum().reset_index()
    min_cases_per_state = min_cases_per_state.loc[min_cases_per_state.groupby('Area_Name')['Rape_Cases_Reported'].idxmin()]

    merged_data = india_map.merge(total_cases_per_state, how='left', left_on='STATE', right_on='Area_Name')
    merged_data = merged_data.merge(max_cases_per_state[['Area_Name', 'Year', 'Rape_Cases_Reported']], how='left', on='Area_Name', suffixes=('', '_max'))
    merged_data = merged_data.merge(min_cases_per_state[['Area_Name', 'Year', 'Rape_Cases_Reported']], how='left', on='Area_Name', suffixes=('', '_min'))

    fig = px.choropleth(merged_data,
                        geojson=merged_data.geometry,
                        locations=merged_data.index,
                        color='Rape_Cases_Reported',
                        hover_name='STATE',
                        hover_data={'Rape_Cases_Reported': True, 'Year': True, 'Rape_Cases_Reported_max': True, 'Rape_Cases_Reported_min': True, 'Year_min': True},
                        color_continuous_scale='Viridis',
                        labels={'Rape_Cases_Reported':'Total Rape Cases', 'Year': 'Year', 'Rape_Cases_Reported_max': 'Most Cases in Year', 'Rape_Cases_Reported_min': 'Least Cases in Year', 'Year_min': 'Year with Least Cases'})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title='Rape Cases Heatmap in India', margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig)

# Usage
# rape_5(data)









def rape_6(data):
    
    grouped_data = data.groupby('Year').agg({
        'Rape_Cases_Reported': 'sum',
        'Victims_Above_50_Yrs': 'sum'
    }).reset_index()

   
    fig = px.scatter_3d(data_frame=grouped_data, x='Year', y='Rape_Cases_Reported', z='Victims_Above_50_Yrs',
                        color='Victims_Above_50_Yrs', size='Rape_Cases_Reported', opacity=0.7)
    fig.update_layout(title='3D Scatter plot of Rape Cases Reported, Year, and Victims\' Age')
    fig.update_layout(scene=dict(xaxis_title='Year', yaxis_title='Rape Cases Reported', zaxis_title='Victims Above 50 Years'))

    
    st.plotly_chart(fig)


def theft_1(df1, df2, df3):
    df = pd.concat([df1, df2, df3])

    # Group by year and sum cases
    theft_by_year = df.groupby('Year').agg(
        Recovered_Property=('Cases_Property_Recovered', 'sum'),
        Stolen_Property=('Cases_Property_Stolen', 'sum'),
        # Recovered_Auto=('Auto_Theft_Recovered', 'sum'),
        # Stolen_Auto=('Auto_Theft_Stolen', 'sum')
    )

    # Plot Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    theft_by_year.plot(kind='bar', ax=ax, width=0.8)

    plt.title("Recovered vs. Stolen Property in India",fontsize=24)
    plt.xlabel("Year",fontsize=18)
    plt.ylabel("Number of Cases",fontsize=18)
    plt.xticks(rotation=45,fontsize=18)
    plt.yticks(fontsize=18)   # Rotate x-axis labels for readability
    plt.legend(title="Property Type", loc='upper right')  # Add legend title
    plt.tight_layout()

    # Show plot in Streamlit
    st.pyplot(fig)

def theft_2(df1, df2, df3):
    df = pd.concat([df1, df2, df3])
    total_cases_per_state = df.groupby('Area_Name')['Cases_Property_Stolen'].sum().reset_index()

    total_cases = total_cases_per_state['Cases_Property_Stolen'].sum()
    total_cases_per_state['Percentage'] = total_cases_per_state['Cases_Property_Stolen'] / total_cases * 100

    excluded_states = []
    total_cases_per_state_filtered = total_cases_per_state[
        (~total_cases_per_state['Area_Name'].isin(excluded_states)) &
        (total_cases_per_state['Percentage'] >= 2)
    ]

    smaller_shares_percentage = total_cases_per_state[total_cases_per_state['Percentage'] < 2]['Percentage'].sum()

    smaller_shares_df = pd.DataFrame({
        'Area_Name': ['Smaller Shares'],
        'Percentage': [smaller_shares_percentage]
    })

    total_cases_per_state_filtered = pd.concat([total_cases_per_state_filtered, smaller_shares_df])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(total_cases_per_state_filtered['Percentage'],
           labels=total_cases_per_state_filtered['Area_Name'],
           autopct='%1.1f%%')
    ax.set_title('Total Cases Across States')
    ax.set_ylabel(None)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_most_share_state(df1, df2, df3):
    df = pd.concat([df1, df2, df3])
    total_cases_per_state_year = df.groupby(['Year', 'Area_Name']).agg(
        Stolen_Property=('Cases_Property_Stolen', 'sum'),
        Recovered_Property=('Cases_Property_Recovered', 'sum')
    ).reset_index()

    total_stolen_per_state = total_cases_per_state_year.groupby('Area_Name')['Stolen_Property'].sum()
    most_share_state = total_stolen_per_state.idxmax()

    state_data = total_cases_per_state_year[total_cases_per_state_year['Area_Name'] == most_share_state]
    return state_data

def plot_least_share_state(df1, df2, df3):
    df = pd.concat([df1, df2, df3])
    total_cases_per_state_year = df.groupby(['Year', 'Area_Name']).agg(
        Stolen_Property=('Cases_Property_Stolen', 'sum'),
        Recovered_Property=('Cases_Property_Recovered', 'sum')
    ).reset_index()

    total_stolen_per_state = total_cases_per_state_year.groupby('Area_Name')['Stolen_Property'].sum()
    least_share_state = total_stolen_per_state.idxmin()

    state_data = total_cases_per_state_year[total_cases_per_state_year['Area_Name'] == least_share_state]
    return state_data

def plot_most_increase_state(df1, df2, df3):
    df = pd.concat([df1, df2, df3])
    total_cases_per_state_year = df.groupby(['Year', 'Area_Name']).agg(
        Stolen_Property=('Cases_Property_Stolen', 'sum'),
        Recovered_Property=('Cases_Property_Recovered', 'sum')
    ).reset_index()

    total_stolen_per_state = total_cases_per_state_year.groupby('Area_Name')['Stolen_Property'].sum()
    total_recovered_per_state = total_cases_per_state_year.groupby('Area_Name')['Recovered_Property'].sum()
    increase_percentage = (total_recovered_per_state - total_stolen_per_state) / total_stolen_per_state
    most_increase_state = increase_percentage.idxmax()

    state_data = total_cases_per_state_year[total_cases_per_state_year['Area_Name'] == most_increase_state]
    return state_data

def plot_least_increase_state(df1, df2, df3):
    df = pd.concat([df1, df2, df3])
    total_cases_per_state_year = df.groupby(['Year', 'Area_Name']).agg(
        Stolen_Property=('Cases_Property_Stolen', 'sum'),
        Recovered_Property=('Cases_Property_Recovered', 'sum')
    ).reset_index()

    total_stolen_per_state = total_cases_per_state_year.groupby('Area_Name')['Stolen_Property'].sum()
    total_recovered_per_state = total_cases_per_state_year.groupby('Area_Name')['Recovered_Property'].sum()
    increase_percentage = (total_recovered_per_state - total_stolen_per_state) / total_stolen_per_state
    least_increase_state = increase_percentage.idxmin()

    state_data = total_cases_per_state_year[total_cases_per_state_year['Area_Name'] == least_increase_state]
    return state_data

def theft_3(df1, df2, df3):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Most Share State
    state_data = plot_most_share_state(df1, df2, df3)
    axs[0, 0].plot(state_data['Year'], state_data['Stolen_Property'], marker='o', label='Stolen Property')
    axs[0, 0].plot(state_data['Year'], state_data['Recovered_Property'], marker='o', label='Recovered Property')
    axs[0, 0].set_title(f'Property Variation in {state_data["Area_Name"].iloc[0]} (Most Share State)')
    axs[0, 0].set_xlabel('Year')
    axs[0, 0].set_ylabel('Number of Cases')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: Least Share State
    state_data = plot_least_share_state(df1, df2, df3)
    axs[0, 1].plot(state_data['Year'], state_data['Stolen_Property'], marker='o', label='Stolen Property')
    axs[0, 1].plot(state_data['Year'], state_data['Recovered_Property'], marker='o', label='Recovered Property')
    axs[0, 1].set_title(f'Property Variation in {state_data["Area_Name"].iloc[0]} (Least Share State)')
    axs[0, 1].set_xlabel('Year')
    axs[0, 1].set_ylabel('Number of Cases')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Most Increase State
    state_data = plot_most_increase_state(df1, df2, df3)
    axs[1, 0].plot(state_data['Year'], state_data['Stolen_Property'], marker='o', label='Stolen Property')
    axs[1, 0].plot(state_data['Year'], state_data['Recovered_Property'], marker='o', label='Recovered Property')
    axs[1, 0].set_title(f'Property Variation in {state_data["Area_Name"].iloc[0]} (Most Increase State)')
    axs[1, 0].set_xlabel('Year')
    axs[1, 0].set_ylabel('Number of Cases')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: Least Increase State
    state_data = plot_least_increase_state(df1, df2, df3)
    axs[1, 1].plot(state_data['Year'], state_data['Stolen_Property'], marker='o', label='Stolen Property')
    axs[1, 1].plot(state_data['Year'], state_data['Recovered_Property'], marker='o', label='Recovered Property')
    axs[1, 1].set_title(f'Property Variation in {state_data["Area_Name"].iloc[0]} (Least Increase State)')
    axs[1, 1].set_xlabel('Year')
    axs[1, 1].set_ylabel('Number of Cases')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)




import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px

def theft_4(property_df1, property_df2, property_df3):
    india_map = gpd.read_file('./data/indian-shapefile/india_ds.shp')

    df = pd.concat([property_df1, property_df2, property_df3])

    # Preprocessing state names to ensure consistency
    df['Area_Name'] = df['Area_Name'].str.lower().str.replace('&', 'and')
    india_map['STATE'] = india_map['STATE'].str.lower().str.replace('&', 'and')

    df['Area_Name'] = df['Area_Name'].replace({'odisha': 'orissa'})

    # Group by state and sum cases
    total_cases_per_state = df.groupby('Area_Name').agg(
        Stolen_Property=('Cases_Property_Stolen', 'sum'),
        Recovered_Property=('Cases_Property_Recovered', 'sum')
    ).reset_index()

    merged_data = india_map.merge(total_cases_per_state, how='left', left_on='STATE', right_on='Area_Name')

    fig = px.choropleth(merged_data,
                        geojson=merged_data.geometry,
                        locations=merged_data.index,
                        color='Stolen_Property',
                        hover_name='Area_Name',
                        hover_data={'Stolen_Property': True},
                        color_continuous_scale='Viridis',
                        labels={'Stolen_Property':'Stolen Property'})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title='Property Stolen Heatmap', margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig)



def theft_5(property_df1, property_df2, property_df3):
    india_map = gpd.read_file('./data/indian-shapefile/india_ds.shp')

    df = pd.concat([property_df1, property_df2, property_df3])

    # Preprocessing state names to ensure consistency
    df['Area_Name'] = df['Area_Name'].str.lower().str.replace('&', 'and')
    india_map['STATE'] = india_map['STATE'].str.lower().str.replace('&', 'and')

    df['Area_Name'] = df['Area_Name'].replace({'odisha': 'orissa'})

    # Group by state and sum cases
    total_cases_per_state = df.groupby('Area_Name').agg(
        Stolen_Property=('Cases_Property_Stolen', 'sum'),
        Recovered_Property=('Cases_Property_Recovered', 'sum')
    ).reset_index()

    merged_data = india_map.merge(total_cases_per_state, how='left', left_on='STATE', right_on='Area_Name')

    fig = px.choropleth(merged_data,
                    geojson=merged_data.geometry,
                    locations=merged_data.index,
                    color='Recovered_Property',
                    hover_name='Area_Name',
                    hover_data={'Recovered_Property': True},
                    color_continuous_scale='Viridis',
                    labels={'Recovered_Property':'Recovered Property'})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title='Property Recovered Heatmap', margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig)

def theft_6(property_df1):
    # Pivoting the dataframe to create a heatmap data
    heatmap_data = property_df1.pivot_table(index="Year", columns="Sub_Group_Name", values="Cases_Property_Recovered")

    fig = px.imshow(heatmap_data,
                    labels=dict(x="Sub-Group Name", y="Year", color="Cases Property Recovered"),
                    color_continuous_scale='Viridis')  # Change color scale if needed
    fig.update_layout(title="Cases of Property Recovered Across Years and Sub-Groups",
                      xaxis_nticks=len(property_df1["Sub_Group_Name"].unique()))
    st.plotly_chart(fig)


def fraud_1(fraud_data):
    # Combine all columns to get total loss of property cases
    fraud_data['Total_Loss_of_Property'] = fraud_data.iloc[:, 1:].sum(axis=1)

    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=fraud_data, x='Year', y='Total_Loss_of_Property', estimator=sum)
    plt.title('Total Loss of Property Cases Across Years')
    plt.xlabel('Year')
    plt.ylabel('Total Loss of Property Cases')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def fraud_2(fraud_data):
    columns = ['Loss_of_Property_1_10_Crores', 'Loss_of_Property_10_25_Crores',
               'Loss_of_Property_25_50_Crores', 'Loss_of_Property_50_100_Crores',
               ]

    num_categories = len(columns)
    num_rows = (num_categories + 1) // 2
    num_cols = min(2, num_categories)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 12))

    # Plot each category in a subplot
    for i, column in enumerate(columns):
        if num_cols == 1:
            sns.barplot(data=fraud_data, x='Year', y=column, estimator=sum, ax=axes[i])
            axes[i].set_title(f'Total Loss of Property Cases ({column.replace("_", " ")}) Across Years')
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel(f'Total Loss of Property Cases ({column.replace("_", " ")})')
            axes[i].tick_params(axis='x', rotation=45)
        else:
            row = i // 2
            col = i % 2
            sns.barplot(data=fraud_data, x='Year', y=column, estimator=sum, ax=axes[row, col])
            axes[row, col].set_title(f'Total Loss of Property Cases ({column.replace("_", " ")}) Across Years')
            axes[row, col].set_xlabel('Year')
            axes[row, col].set_ylabel(f'Total Loss of Property Cases ({column.replace("_", " ")})')
            axes[row, col].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(plt)

def fraud_3(fraud_data):
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=fraud_data, x='Year', y='Loss_of_Property_1_10_Crores', hue='Group_Name', estimator=sum)
    plt.title('Loss of Property Categories Over Years')
    plt.xlabel('Year')
    plt.ylabel('Total Loss of Property (1-10 Crores)')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    
    st.pyplot(plt)

def fraud_4(fraud_data):
    pivot_table = fraud_data.pivot_table(index='Area_Name', columns='Year', values='Loss_of_Property_1_10_Crores', aggfunc='sum')

    vmin = pivot_table.values.min()
    vmax = pivot_table.values.max()

    # Plot heatmap with adjusted color range
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='Blues', linewidths=0.5, linecolor='gray', vmin=vmin, vmax=vmax, ax=ax)
    ax.set_title('Loss of Property by State and Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('State')
    plt.tight_layout()
    
    
    st.pyplot(fig)

# def fraud_6(fraud_data):
    # india_map = gpd.read_file('./data/indian-shapefile/india_ds.shp')

    
    # india_map['STATE'] = india_map['STATE'].str.lower().str.replace('&', 'and')
    # fraud_data['Area_Name'] = fraud_data['Area_Name'].str.lower().str.replace('&', 'and')
    # fraud_data['Area_Name'] = fraud_data['Area_Name'].replace({'odisha': 'orissa'})

    
    # rajasthan_data = pd.DataFrame({'Area_Name': ['rajasthan'], 'Total_Fraud_Cases': [0]})
    # fraud_data = pd.concat([fraud_data, rajasthan_data], ignore_index=True)

    # merged_data = india_map.merge(fraud_data.groupby('Area_Name')['Total_Fraud_Cases'].sum().reset_index(),
    #                               how='left', left_on='STATE', right_on='Area_Name')

    # fig = px.choropleth(merged_data,
    #                     geojson=merged_data.geometry,
    #                     locations=merged_data.index,
    #                     color='Total_Fraud_Cases',
    #                     hover_name='STATE',
    #                     hover_data={'Total_Fraud_Cases': True},
    #                     color_continuous_scale='Viridis',
    #                     labels={'Total_Fraud_Cases': 'Total Fraud Cases'})
    # fig.update_geos(fitbounds="locations", visible=False)
    # fig.update_layout(title='Total Fraud Cases in India', margin={"r":0,"t":30,"l":0,"b":0})
    
    
    # st.plotly_chart(fig)    
def fraud_6(fraud_data):
    india_map = gpd.read_file('./data/indian-shapefile/india_ds.shp')

    
    fraud_data['Total_Fraud_Cases'] = fraud_data[['Loss_of_Property_1_10_Crores',
                                              'Loss_of_Property_10_25_Crores',
                                              'Loss_of_Property_25_50_Crores',
                                              'Loss_of_Property_50_100_Crores',
                                              'Loss_of_Property_Above_100_Crores']].sum(axis=1)
    india_map['STATE'] = india_map['STATE'].str.lower().str.replace('&', 'and')
    fraud_data['Area_Name'] = fraud_data['Area_Name'].str.lower().str.replace('&', 'and')
    fraud_data['Area_Name'] = fraud_data['Area_Name'].replace({'odisha': 'orissa'})

    # Create a DataFrame for Rajasthan with 0 fraud cases
    rajasthan_data = pd.DataFrame({'Area_Name': ['rajasthan'], 'Total_Fraud_Cases': [0]})
    fraud_data = pd.concat([fraud_data, rajasthan_data], ignore_index=True)

    merged_data = india_map.merge(fraud_data.groupby('Area_Name')['Total_Fraud_Cases'].sum().reset_index(),
                                  how='left', left_on='STATE', right_on='Area_Name')

    merged_data['Total_Fraud_Cases'] = np.log1p(merged_data['Total_Fraud_Cases'])

    fig = px.choropleth(merged_data,
                        geojson=merged_data.geometry,
                        locations=merged_data.index,
                        color='Total_Fraud_Cases',
                        hover_name='STATE',
                        hover_data={'Total_Fraud_Cases': True},
                        color_continuous_scale='Viridis',
                        labels={'Total_Fraud_Cases': 'Total Fraud Cases (Log Scale)'})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title='Total Fraud Cases in India (Log Scale)', margin={"r":0,"t":30,"l":0,"b":0})

    # Display the plot in Streamlit
    st.plotly_chart(fig)
def fraud_5(fraud_data):
    categories = ['Loss_of_Property_1_10_Crores', 'Loss_of_Property_10_25_Crores',
           'Loss_of_Property_25_50_Crores', 'Loss_of_Property_50_100_Crores',
           'Loss_of_Property_Above_100_Crores']
    total_cases_per_year = fraud_data.pivot_table(index='Year', values=categories, aggfunc='sum').sum(axis=1)

   
    plt.figure(figsize=(10, 6))
    plt.plot(total_cases_per_year.index, total_cases_per_year.values, marker='o', linestyle='-')
    plt.title('Trend of Total Cases Across All Categories Across India (Per Year)')
    plt.xlabel('Year')
    plt.ylabel('Total Cases')
    plt.grid(True)
    plt.tight_layout()
    
    
    st.pyplot(plt) 

def load_data():
    crime_df_lower = crime_df.copy()
    crime_df_lower['STATE'] = crime_df_lower['STATE'].str.lower().str.title()
    crime_df_2001 = crime_df_lower[crime_df_lower['YEAR'] == 2001]
    crime_df_2010 = crime_df_lower[crime_df_lower['YEAR'] == 2010]

    total_crimes_by_state_2001 = crime_df_2001.groupby('STATE')['TOTAL IPC CRIMES'].sum()
    total_crimes_by_state_2010 = crime_df_2010.groupby('STATE')['TOTAL IPC CRIMES'].sum()

    total_population_by_state_2001 = popu_df.set_index('State')['2001']
    total_population_by_state_2010 = popu_df.set_index('State')['2010']

    crime_rate_by_state_2001 = (total_crimes_by_state_2001 / total_population_by_state_2001) * 100000
    crime_rate_by_state_2010 = (total_crimes_by_state_2010 / total_population_by_state_2010) * 100000

    crime_rate_by_state_2001.index = crime_rate_by_state_2001.index.str.replace('&', 'And').str.replace('N ', 'N').str.replace('Delhi Ut', 'Delhi').str.replace('Orissa', 'Odisha').str.replace('Pondicherry', 'Puducherry')
    crime_rate_by_state_2010.index = crime_rate_by_state_2010.index.str.replace('&', 'And').str.replace('N ', 'N').str.replace('Delhi Ut', 'Delhi').str.replace('Orissa', 'Odisha').str.replace('Pondicherry', 'Puducherry')

    crime_rate_by_state_2001 = crime_rate_by_state_2001.dropna()
    crime_rate_by_state_2010 = crime_rate_by_state_2010.dropna()

    other_states_mask_2001 = (crime_rate_by_state_2001 / crime_rate_by_state_2001.sum()) < 0.018
    other_states_mask_2010 = (crime_rate_by_state_2010 / crime_rate_by_state_2010.sum()) < 0.018

    other_states_crime_rate_2001 = crime_rate_by_state_2001[other_states_mask_2001].sum()
    other_states_crime_rate_2010 = crime_rate_by_state_2010[other_states_mask_2010].sum()

    crime_rate_by_state_2001 = crime_rate_by_state_2001[~other_states_mask_2001]
    crime_rate_by_state_2010 = crime_rate_by_state_2010[~other_states_mask_2010]

    crime_rate_by_state_2001['Other States'] = other_states_crime_rate_2001
    crime_rate_by_state_2010['Other States'] = other_states_crime_rate_2010

    crime_rate_percentage_change = ((crime_rate_by_state_2010 - crime_rate_by_state_2001) / crime_rate_by_state_2001) * 100
    
    return crime_rate_by_state_2001, crime_rate_by_state_2010


    
def f1():
    # st.subheader("Distribution of Crime Rate by State in 2001")
    crime_rate_by_state_2001, _ = load_data()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(crime_rate_by_state_2001, labels=crime_rate_by_state_2001.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Distribution of Crime Rate by State in 2001')
    ax.axis('equal')
    st.pyplot(fig)

def f2():
    # st.subheader("Distribution of Crime Rate by State in 2010")
    _, crime_rate_by_state_2010 = load_data()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(crime_rate_by_state_2010, labels=crime_rate_by_state_2010.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Distribution of Crime Rate by State in 2010')
    ax.axis('equal')
    st.pyplot(fig)
# page1_panel5(crime_df, popu_df)    

# DATASET MANIPULATION TILL HERE
# DASHBOARD LAYOUT BELOW
# DATASET MANIPULATION TILL HERE
# DASHBOARD LAYOUT BELOW
# DATASET MANIPULATION TILL HERE
# DASHBOARD LAYOUT BELOW
# DATASET MANIPULATION TILL HERE
# DASHBOARD LAYOUT BELOW

#st.title('Crime Data Analysis Dashboard')
st.set_page_config(page_title="Crime Data Analysis", layout="wide")

# Sidebar
st.sidebar.title("Crime Data Analysis Dashboard")
selected_tab = st.sidebar.radio("Choose a tab", ["National Crime Stats", "Data by Type of Crime","2001-2010 comparison"])

# # Checkbox
# crime_type = st.sidebar.checkbox("Per Capita Crime", False)

# Create tabs
if selected_tab == "National Crime Stats":
    st.title("National Crime Statistics")
    col1, col2= st.columns(2)
    with col1:
        st.header("Year-on-Year Trends")
        page1_panel1()

        
       

    with col2:
        st.header("Distribution by Type of Crime")
       
        page1_panel2()

    # with col3:
    #     st.header("Panel 3")
        # Insert code to create graph 3 here

    col4, col5 = st.columns(2)

    with col4:
        st.header("Distribution by State")
        selected_tab_p1p4 = st.radio("Format:", ["Total", "Per Capita"])
        if selected_tab_p1p4 == "Total":
            page1_panel4()
        elif selected_tab_p1p4 == "Per Capita":   
            page1_panel4_capita() 
        

    with col5:
        st.header("National Heatmaps")
        selected_tab_p1p5= st.radio("Data Format:", ["Total", "Per Capita"])
        if selected_tab_p1p5== "Total":
            page1_panel5(crime_df)
        elif selected_tab_p1p5== "Per Capita":   
            page1_panel5_capita(crime_df,popu_df) 
        # page1_panel5(crime_df)
        

    # with col6:
    #     st.header("Panel 6")
    
    # col6, col7 = st.columns(2)

    # with col6:
    #     st.header("Panel 6")
    #     page1_panel1()
    #     # Insert code to create graph 4 here

    # with col7:
    #     st.header("Panel 7")
    #     page1_panel1()
   

elif selected_tab == "Data by Type of Crime":
    st.title("Crime Data by Type of Crime")
    selected_crime = st.radio("Choose a tab", ["Murder","Rape","Theft","Fraud"])
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Panel 1")
        if selected_crime== "Murder":
            murder_1(murder_df)
        elif selected_crime== "Rape":   
            rape_1(data)
        elif selected_crime== "Theft":   
            theft_1(property_df1, property_df2, property_df3)
        elif selected_crime== "Fraud":   
            fraud_1(fraud_data)        
        
        

    with col2:
        st.header("Panel 2")
        if selected_crime== "Murder":
            murder_2(murder_df)
        elif selected_crime== "Rape":   
            rape_2(data)
        elif selected_crime== "Theft":   
            theft_2(property_df1, property_df2, property_df3)
        elif selected_crime== "Fraud":   
            fraud_2(fraud_data) 

    with col3:
        st.header("Panel 3")
        if selected_crime== "Murder":
            murder_3(murder_df)
        elif selected_crime== "Rape":   
            rape_3(data)
        elif selected_crime== "Theft":   
            theft_3(property_df1, property_df2, property_df3)
        elif selected_crime== "Fraud":   
            fraud_3(fraud_data)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.header("Panel 4")
        if selected_crime== "Murder":
            murder_4(murder_df)
        elif selected_crime== "Rape":   
            rape_4(data)
        elif selected_crime== "Theft":   
            theft_4(property_df1, property_df2, property_df3)
        elif selected_crime== "Fraud":   
            fraud_4(fraud_data)

    with col5:
        st.header("Panel 5")
        if selected_crime== "Murder":
            murder_5(murder_df)
        elif selected_crime== "Rape":   
            rape_5(data)
        elif selected_crime== "Theft":   
            theft_5(property_df1, property_df2, property_df3)
        elif selected_crime== "Fraud":   
            fraud_5(fraud_data)

    with col6:
        st.header("Panel 6")
        if selected_crime== "Murder":
            murder_6(murder_df)
        elif selected_crime== "Rape":   
            rape_6(data)
        elif selected_crime== "Theft":   
            theft_6(property_df1)
        elif selected_crime== "Fraud":   
            fraud_6(fraud_data)


elif selected_tab == "2001-2010 comparison":
    st.title("Crime Data by Decadal Comparison")
    col1, col2= st.columns(2)

    with col1:
        
        f1()
        

    with col2:
        
        f2()
       

   
        

   