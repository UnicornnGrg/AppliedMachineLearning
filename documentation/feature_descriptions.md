# PUMS Data Feature Descriptions

## Common PUMS Person-Level Variables

### Demographics

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| **SERIALNO** | Housing unit/GQ person serial number | ID | Unique identifier |
| **SPORDER** | Person number | Integer | Order within household |
| **PUMA** | Public use microdata area code | Categorical | Geographic area |
| **ST** | State code | Categorical | Two-digit state FIPS code |
| **AGEP** | Age | Integer | 0-99 years |
| **SEX** | Sex | Categorical | 1=Male, 2=Female |
| **RAC1P** | Race | Categorical | Detailed race categories |
| **HISP** | Hispanic origin | Categorical | 01=Not Hispanic, 02-24=Hispanic origins |
| **MAR** | Marital status | Categorical | 1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never married |
| **CIT** | Citizenship status | Categorical | 1-5 representing different citizenship statuses |

### Education

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| **SCHL** | Educational attainment | Categorical | 01-24 representing different education levels |
| **SCH** | School enrollment | Categorical | 1=No, 2=Yes, public, 3=Yes, private |
| **SCHG** | Grade level attending | Categorical | For those enrolled in school |

### Employment & Income

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| **ESR** | Employment status | Categorical | 1=Employed, 2=Employed not at work, 3=Unemployed, 4-6=Not in labor force |
| **WKHP** | Usual hours worked per week | Integer | 1-99 hours |
| **WKW** | Weeks worked during past 12 months | Categorical | Ranges of weeks |
| **COW** | Class of worker | Categorical | Private, government, self-employed, etc. |
| **WAGP** | Wages or salary income | Integer | In dollars (past 12 months) |
| **PERNP** | Total person's earnings | Integer | In dollars (past 12 months) |
| **PINCP** | Total person's income | Integer | In dollars (past 12 months) |
| **POVPIP** | Income-to-poverty ratio | Integer | Percentage |
| **JWMNP** | Travel time to work | Integer | Minutes |
| **JWTR** | Means of transportation to work | Categorical | Car, bus, walk, etc. |

### Health & Disability

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| **DIS** | Disability status | Binary | 1=Yes, 2=No |
| **DEAR** | Hearing difficulty | Binary | 1=Yes, 2=No |
| **DEYE** | Vision difficulty | Binary | 1=Yes, 2=No |
| **DREM** | Cognitive difficulty | Binary | 1=Yes, 2=No |
| **DPHY** | Ambulatory difficulty | Binary | 1=Yes, 2=No |
| **DDRS** | Self-care difficulty | Binary | 1=Yes, 2=No |
| **DOUT** | Independent living difficulty | Binary | 1=Yes, 2=No |
| **HICOV** | Health insurance coverage | Binary | 1=Yes, 2=No |

### Language

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| **LANX** | Language other than English spoken at home | Binary | 1=Yes, 2=No |
| **ENG** | Ability to speak English | Categorical | 1=Very well, 2=Well, 3=Not well, 4=Not at all |

### Housing (if applicable)

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| **RELP** | Relationship to householder | Categorical | Various family relationships |
| **NOC** | Number of own children | Integer | 0+ children |
| **NPF** | Number of persons in family | Integer | 2+ persons |

### Weighting Variables

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| **PWGTP** | Person's weight | Integer | For population estimates |
| **PWGTP1-80** | Person's weight replicate factors | Integer | For statistical variance estimation |

## Missing Value Codes

**Important**: PUMS data uses specific codes for missing values:
- **Blank/Empty**: Not applicable or valid skip
- **Negative values**: Various types of missing data (check data dictionary)
- Specific codes vary by variable

## Data Dictionary Resources

For complete and up-to-date variable definitions:
- [Census PUMS Data Dictionary](https://www.census.gov/programs-surveys/acs/microdata/documentation.html)
- Check the year-specific data dictionary for your dataset
- Variable codes and meanings may change between survey years

## Notes for Machine Learning

1. **Categorical Variables**: Most codes are categorical - convert appropriately
2. **Weights**: Use PWGTP for accurate population estimates
3. **Inflation**: Income variables may need adjustment for inflation
4. **Privacy**: Data is anonymized; geographic detail is limited
5. **Sampling**: Remember this is sample data, not full population
