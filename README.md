
- [Project Description](#project-description)
- [Project Goal](#project-goal)
- [Initial Thoughts](#initial-thoughts)
- [Plan](#the-plan)
- [Data Dictionary](#data-dictionary)
- [Steps to Reproduce](#steps-to-reproduce) 
- [Conclusions](#conclusions)
	- [Takeaway and Key Findings](#takeaways-and-key-findings)
	- [Reccomendations and next steps](#recommendations-and-next-steps)
    
# Wine Project


<div align="center">
  <img src="https://github.com/Marc-Aradillas/Wine_Project/assets/106922826/bd7c8b8c-70a9-4334-a416-4005c805d4e7" alt="Alt Text" width="500">
</div>

To forecast a wine's excellence by harnessing the nuanced insights of unsupervised learning algorithms.

# Project Description

Elevating the art of winemaking through data-driven alchemy, this project fuses chemical analytics and machine learning to distill the essence of wine quality, employing cluster analysis and predictive modeling to refine the vintner's craft.

# Project Goal

- Uncover crucial drivers affecting wine quality.
- Utilize K-means clustering to delve into chemical intricacies.
- Gauge cluster efficacy via linear and polynomial regression, deploying the ultimate model for future quality projections.

# Initial Thoughts

- Elevated alcohol levels correlate with quality.
- Volatile acidity influences quality.
- Chloride concentration impacts quality.
- Density sways quality.

# The Plan
- Acquired data from  https://data.world/food/wine-quality
  - Files were concated with a column added to distinguish red from white wines

- Prepare data


# Data Dictionary

| Feature                                  | Data Type | Definition                                                     |
| ---------------------------------------- | :-------: | -------------------------------------------------------------- |
| Fixed Acidity                            |   Float   | The amount of tartaric acid in wine, which is found in grapes (measured in g/dm³). |
| Volatile Acidity                         |   Float   | The amount of acetic acid in wine, which at high levels can lead to an unpleasant, vinegar-like taste. |
| Citric Acid                              |   Float   | Found in small quantities, citric acid can add 'freshness' and flavor to wines. |
| Residual Sugar                           |   Float   | The amount of sugar remaining after fermentation stops (measured in g/dm³). |
| Chloride                                 |   Float   | The amount of salt in the wine (measured in g/dm³). |
| Free Sulfur Dioxide                      |   Float   | The free form of SO₂ present in the wine (measured in mg/dm³). |
| Total Sulfur Dioxide                     |   Float   | Amount of free and bound forms of SO₂ (measured in mg/dm³). |
| Density                                  |   Float   | The density of wine is close to that of water depending on the alcohol and sugar content (measured in g/cm³). |
| pH                                       |   Float   | Describes the acidity or basicity of the wine on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4. |
| Sulphates                                |   Float   | A wine additive that contributes to SO₂ gas levels, which acts as an antimicrobial and antioxidant (measured in g/dm³). |
| Alcohol                                  |   Float   | The percentage of alcohol present in the wine. |
| Quality                                  |   Float   | A score between 0 and 10 given to the wine sample. |
| total_sulfur_ratio                       |   Float   | The ratio of total sulfur dioxide to some other factor. |
| acidity_level                            |   Float   | Represents the level of acidity in the wine. |
| sugar_alcohol_ratio                      |   Float   | The ratio of sugar content to alcohol content in the wine. |
| chlorides_ph_ratio                       |   Float   | The ratio of chlorides to pH level in the wine. |
| density_ph_ratio                         |   Float   | The ratio of wine density to pH level. |
| sulfur_dioxide_level                     |   Float   | The level of sulfur dioxide in the wine. |
| sulfates_chlorides_ratio                 |   Float   | The ratio of sulfates to chlorides in the wine. |
| ph_bins                                  |   Float   | A bin or category representing the pH level of the wine. |
| total_acid                               |   Float   | The total acidity of the wine. |
| sulfur_dioxide_chlorides_ratio           |   Float   | The ratio of sulfur dioxide to chlorides in the wine. |
| residual_sugar_ph_ratio                  |   Float   | The ratio of residual sugar content to pH level in the wine. |
| acid_ratio                               |   Float   | A ratio related to the acidity of the wine. |
| alcohol_ph_ratio                         |   Float   | The ratio of alcohol content to pH level in the wine. |
| chlorides_density_ratio                  |   Float   | The ratio of chlorides to wine density. |
| total_sulfur_residual_sugar_ratio        |   Float   | The ratio of total sulfur dioxide to residual sugar. |
| sulfur_dioxide_percentage                |   Float   | The percentage of sulfur dioxide in the wine. |
| ph_chlorides_ratio                       |   Float   | The ratio of pH level to chlorides in the wine. |
| alcohol_sugar_ratio                      |   Float   | The ratio of alcohol content to sugar content in the wine. |
| density_sulfates_ratio                   |   Float   | The ratio of wine density to sulfates content. |
| chlorides_sulfates_ratio                 |   Float   | The ratio of chlorides to sulfates in the wine. |
| residual_sugar_percentage                |   Float   | The percentage of residual sugar in the wine. |
| alcohol_chlorides_ratio                  |   Float   | The ratio of alcohol content to chlorides in the wine. |
| density_sulfur_dioxide_ratio             |   Float   | The ratio of wine density to sulfur dioxide level. |
| ph_sulfur_dioxide_ratio                  |   Float   | The ratio of pH level to sulfur dioxide level in the wine. |
| sulfur_dioxide_sugar_ratio              |   Float   | The ratio of sulfur dioxide to sugar content in the wine. |
| composite_cluster_1                      |   Float   | A cluster or grouping assigned to the wine sample (Cluster 1). |
| composite_cluster_2                      |   Float   | A cluster or grouping assigned to the wine sample (Cluster 2). |
| composite_cluster_3                      |   Float   | A cluster or grouping assigned to the wine sample (Cluster 3). |
| composite_cluster_4                      |   Float   | A cluster or grouping assigned to the wine sample (Cluster 4). |
| alcohol_bins_low                         |   Float   | A bin or category representing low alcohol content. |
| alcohol_bins_medium                      |   Float   | A bin or category representing medium alcohol content. |
| alcohol_bins_high                        |   Float   | A bin or category representing high alcohol content. |
| ph_bins_low                              |   Float   | A bin or category representing low pH level. |
| ph_bins_medium                           |   Float   | A bin or category representing medium pH level. |
| ph_bins_high                             |   Float   | A bin or category representing high pH level. |
| quality_bins_poor                        |   Float   | A bin or category representing poor wine quality. |
| quality_bins_average                     |   Float   | A bin or category representing average wine quality. |
| quality_bins_excellent                   |   Float   | A bin or category representing excellent wine quality. |


# Steps to Reproduce

- Fetch wine quality data from Data.World.
- Merge red and white wine CSVs, adding a color identifier.
- Save as 'winequality_red_white.csv' for repo use.
- Clone repo and insert data.
- Execute notebook.

# Conclusion
Takeaways and Key Findings
- The chemical properties in the collection of red and white wines taht seem to drive quality are aclohol content, citric acid, and fixed acidity.

- The wines on average contain around 10.5% alcohol content and are mostly composed of average quality wine.

- The clusters when hued by density tells us how they are present within our data



# Model Improvement

- The model still requires further improvement either hyperparamater tuning or revisiting prepare and feature engineering.

# Recommendations and Next Steps

- I would recommend to gather data on wine shelf life, aged wine over time may play a part in quality of wine. The organics used to create the wine would possibily help in predicting quality.

- Given more time, the following actions could be considered:
  - Gather more data to improve model performance.
  - Feature engineer new variables to enhance model understanding.
  - Fine-tune model parameters for better performance.


