
- Project Description
- Project Goal
- Initial Thoughts
- Plan
- Data Dictionary
- Steps to Reproduce
- Conclusions
  - Takeaway and Key Findings
  - Reccomendations
  - Next Step

    
# Wine_Project

<img width="400" alt="Screenshot 2023-09-20 at 12 11 45 PM" src="https://github.com/Marc-Aradillas/Wine_Project/assets/136507682/e3b58b6e-1037-4a5e-a858-860b3fd513ed">
s

To forecast a wine's excellence by harnessing the nuanced insights of unsupervised learning algorithms.

# Project Description

Elevating the art of winemaking through data-driven alchemy, this project fuses chemical analytics and machine learning to distill the essence of wine quality, employing cluster analysis and predictive modeling to refine the vintner's craft.

# Project Goal

- Uncover crucial drivers affecting wine quality.
- Utilize K-means clustering to delve into chemical intricacies.
- Gauge cluster efficacy via linear and polynomial regression, deploying the ultimate model for future quality projections. Tools & Tech:

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

|Feature            | Data Type | Defination |
| :---------------- | :------: | :---- |
|Fixed Acidity       |   Float  | The amount of tartaric acid in wine, which is found in grapes (measured in g/dm³). |
| Volatile Acidity          | Float  | The amount of acetic acid in wine, which at high levels can lead to an unpleasant, vinegar-like taste. |
| Citric Acid   |  Float    | Found in small quantities, citric acid can add 'freshness' and flavor to wines.|
| Residual Sugar|  Float    | The amount of sugar remaining after fermentation stops (measured in g/dm³). |
|Chloride      |   Float  | The amount of salt in the wine (measured in g/dm³). |
| Free Sulfur dioxide         |   Float    | The free form of SO₂ present in the wine (measured in mg/dm³). |
| Total sulfar dioxide | Float | Amount of free and bound forms of SO₂ (measured in mg/dm³).|
| Density  | Float  | The density of wine is close to that of water depending on the alcohol and sugar content (measured in g/cm³).|
| PH|  Float   |  Describes the acidity or basicity of the wine on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4.
 |
|Sulphates      |  Float   |A wine additive that contributes to SO₂ gas levels, which acts as an antimicrobial and antioxidant (measured in g/dm³).|
| Alcohol        |   Float  | The percentage of alcohol present in the wine.|
| Quality  |  Float   | A score between 0 and 10 given to the wine sample. |

# Steps to Reproduce

- Fetch wine quality data from Data.World.
- Merge red and white wine CSVs, adding a color identifier.
- Save as 'combined_wine.csv' for repo use.
- Clone repo and insert data.
- Execute notebook.

# Conclusion
Takeaways and Key Findings



# Model Improvement

# Recommendations and Next Steps



