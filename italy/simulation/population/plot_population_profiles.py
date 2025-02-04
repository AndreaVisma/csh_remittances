import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv('c:\\data\\migration\\italy\\estimated_stocks.csv')
df['age'] = df.age_group.astype(str).apply(lambda x: int(re.findall(r'\d+', x)[0]))
df.sort_values(['citizenship', 'year', 'age'], inplace = True)
sns.set(style="whitegrid")

def plot_bar_for_year(df, citizenship='Afghanistan', year=2010):
    """
    Create a bar plot for a specific year and citizenship.
    Displays count by age_group, with bars colored by sex.
    """
    # Filter the data for the selected citizenship and year
    df_year = df[(df['citizenship'] == citizenship) & (df['year'] == year)]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='age', y='count', hue='sex', data=df_year)
    plt.title(f'Migrant Count by Age Group and Sex in {year} ({citizenship})')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show(block = True)

def plot_time_series(df, citizenship='Afghanistan', age_group='From 10 to 14 years', sex='female'):
    """
    Create a time series plot of counts over years for a given citizenship, age_group, and sex.
    """
    # Filter the data for the selected parameters
    df_series = df[(df['citizenship'] == citizenship) &
                   (df['age_group'] == age_group) &
                   (df['sex'] == sex)]

    # Create the line plot
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(x='year', y='count', data=df_series, marker="o")
    plt.title(f'Time Series for {age_group} - {sex} ({citizenship})')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.xticks(df_series['year'].unique())  # Ensure all years are shown
    plt.tight_layout()
    plt.show(block = True)

plot_bar_for_year(df, citizenship='Germany', year=2020)

# Visualize a time series for a specific group
plot_time_series(df, citizenship='Afghanistan', age_group='From 10 to 14 years', sex='female')


## population pyramids
def prepare_pyramid_data(df, citizenship, year):
    """
    Filter the data for a given citizenship and year, and then
    aggregate counts by age_group and sex.

    Returns:
        DataFrame with index age_group and columns for each sex.
    """
    # Filter by citizenship and year
    df_filtered = df[(df['citizenship'] == citizenship) & (df['year'] == year)]

    # Aggregate counts by age_group and sex
    agg = df_filtered.groupby(['age', 'sex'])['count'].sum().unstack(fill_value=0)

    # Optionally, you might want to sort the age groups.
    # This sorting assumes age_group is categorical and in a logical order.
    # If not, you may need to define a custom order.
    agg = agg.sort_index()

    return agg

def plot_population_pyramid(ax, pyramid_data, title):
    """
    Given an axis (ax) and pyramid_data (DataFrame with index=age_group, columns e.g., ['male', 'female']),
    plot a horizontal bar chart representing the population pyramid.

    Males will be plotted as negative values on the left and females as positive on the right.
    """
    # Ensure the DataFrame has columns for 'male' and 'female'
    # If one is missing, add it with zeros.
    for sex in ['male', 'female']:
        if sex not in pyramid_data.columns:
            pyramid_data[sex] = 0

    # To create a pyramid, make the male counts negative.
    pyramid_data = pyramid_data.copy()
    pyramid_data['male'] = -pyramid_data['male']

    # Get age groups
    age_groups = pyramid_data.index.tolist()

    # Plot horizontal bars for males and females
    ax.barh(age_groups, pyramid_data['male'], color='royalblue', label='Male', height = 5)
    ax.barh(age_groups, pyramid_data['female'], color='coral', label='Female', height = 5)

    # Draw a vertical line at 0 for clarity.
    ax.axvline(0, color='black', linewidth=0.5)

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel('Population Count')
    ax.set_ylabel('Age Group')

    # Optionally, add a legend.
    ax.legend()

def compare_population_pyramids(df, citizenship1, citizenship2, year):
    """
    Create side-by-side population pyramids for two diaspora groups.
    """
    # Prepare the aggregated data for each group
    pyramid1 = prepare_pyramid_data(df, citizenship1, year)
    pyramid2 = prepare_pyramid_data(df, citizenship2, year)

    # Create two subplots side by side
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8), sharey=True)

    plot_population_pyramid(axes[0], pyramid1, f"{citizenship1} - {year}")
    plot_population_pyramid(axes[1], pyramid2, f"{citizenship2} - {year}")

    # Adjust layout and display
    fig.savefig(f"C:\\git-projects\\csh_remittances\\italy\\plots\\diaspora\\comparisons\\{citizenship1}_{citizenship2}_{year}.svg")
    plt.show(block = True)


citizenship1 = 'Pakistan'
citizenship2 = 'Philippines'
year = 2015

# Display the comparison plot.
compare_population_pyramids(df, 'Philippines', 'Bangladesh', 2010)


