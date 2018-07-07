### 特徴を見ていく

### application_train

* SK_ID_CURR

ID of loan.Key for other tables?

* TARGET

target variable.1-0 value(x日の滞納があれば1, なければ0)

* NAME_CONTRACT_TYPE

Categorical (cash/revolving)

* CODE_GENDER

M/F/XNA

* FLAG_OWN_CAR

Y/N
about 66% of them don't have

* FLAG_OWN_REALTY

Y/N
* CNT_CHILDREN

How many children does the client have
Many children but young -> dangerous?
(1-12, 14, 19)

* AMT_INCOME_TOTAL

real number
depends on the country or region?

* AMT_CREDIT

real number

* AMT_ANNUITY

real number
has null

* AMT_GOODS_PRICE

real number
has null

* NAME_TYPE_SUITE

category
some record shows that they come with children though they do not have any children
has null
8types

* NAME_INCOME_TYPE

category
8types

* NAME_EDUCATION_TYPE

category
5types

* NAME_FAMILY_STATUS

category
6types

* NAME_HOUSING_TYPE

category
6types

* REGION_POPULATION_RELATIVE

real number
81 regions
can get the region information from this

* DAYS_BIRTH

integer
0 > value
can get the year of the applicants

* DAYS_EMPLOYED

has a overflowed value
integer

* DAYS_REGISTRATION

integer
0 >

* DAYS_ID_PUBLISHED

integer
0 >

* OWN_CAR_AGE

integer
has null
63 unique values(include null)
year but 91 ???

* FLAG_MOBIL

1/0
almost all the record is 1

* FLAG_EMP_PHONE

1/0
81.9% is 1

* FLAG_WORK_PHONE

1/0
19.9% is 1

* FLAG_CONT_MOBILE

1/0
99.8% is 1

* FLAG_PHONE

1/0
28.1% is 1

* FLAG_EMAIL

1/0
5.7% is 1

* OCCUPATION_TYPE

category
has null
19types

* CNT_FAM_MEMBERS

integer
has null
1-16 20

* REGION_RATING_CLIENT

1-3 value
which is good?

* REGION_RATING_CLIENT_W_CITY

1-3 value

* WEEKDAY_APPR_PROCESS_START

category

* HOUR_APPR_PROCESS_START

integer
24 hours

* REG_REGION_NOT_LIVE_REGION

1/0
1.5%

* REG_REGION_NOT_WORK_REGION

1/0
5%

* LIVE_REGION_NOT_WORK_REGION

1/0
4%

* REG_CITY_NOT_LIVE_CITY

1/0
7%

* REG_CITY_NOT_WORK_CITY

1/0
23%

* LIVE_CITY_NOT_WORK_CITY

1/0
17%

* ORGANIZATION_TYPE

category
58types

* EXT_SOURCE_1

0-1 real
has null

* EXT_SOURCE_2

0-1 real
has null

* EXT_SOURCE_3

0-1 real
has null

* APARTMENT_AVG - NON_LIVING_AREA_MEDI

real
has null
the record which has values are almost the same

* FONDKAPREMONT_MODE

category
has null
5types with nan

* HOUSETYPE_MODE

category
has null
5types with nan

* TOTALAREA_MODE

real
0.0-1.0 value
has null

* WALLSMATERIAL_MODE

category
has null
8types with null

* EMERGENCYSTATE_MODE

yes/no/nan
has null

* OBS_30_CNT_SOCIAL_CIRCLE

integer
max348 min0

* DEF_30_CNT_SOCIAL_CIRCLE

integer
max34, min0

* OBS_60_CNT_SOCIAL_CIRCLE

integer
max344, min0

* DEF_60_CNT_SOCIAL_CIRCLE

integer
max24, min0

* DAYS_LAST_PHONE_CHANGE

integer <= 0
max 0.0 min -4292

* FLAG_DOCUMENT_*

0/1

* AMT_REQ_CREDIT_BUREAU_HOUR

0-4 nan

* AMT_REQ_CREDIT_BUREAU_DAY

0-6 8, 9, nan

* AMT_REQ_CREDIT_BUREAU_WEEK

0-7, nan

* AMT_REQ_CREDIT_BUREAU_MON

almost the same

* AMT_REQ_CREDIT_BUREAU_YEAR

almost the same
